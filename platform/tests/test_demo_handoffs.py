"""Public demos and staff handoff boundaries, with Runtime HTTP stubbed."""
import os
from datetime import timedelta

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient
from src.platform import calls, reaper
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import Handoff, Message, VoiceSession, utcnow
from test_interpreter_participants import runtime, livekit  # noqa: F401


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.delenv("PLATFORM_DEMO_AGENTS", raising=False)
    monkeypatch.delenv("PLATFORM_STAFF_KEY", raising=False)
    monkeypatch.delenv("PLATFORM_API_KEYS", raising=False)
    monkeypatch.setenv("PLATFORM_PUBLIC_CALLS", "1")
    calls.reset_limits()
    yield
    calls.reset_limits()


def bearer(token):
    return {"authorization": f"Bearer {token}"}


def demo(client, monkeypatch):
    agent = client.post("/agents", json={"name": "Demo", "enabled_languages": ["en", "ko"]}).json()
    monkeypatch.setenv("PLATFORM_DEMO_AGENTS", f"sample={agent['id']}")
    return agent


def source(client, monkeypatch):
    demo(client, monkeypatch)
    response = client.post("/demo/agents/sample/sessions", json={"name": "Alex"})
    assert response.status_code == 201, response.text
    result = response.json()
    # Runtime acknowledges creation; its SessionStarted event normally marks running.
    with SessionLocal() as db:
        db.get(VoiceSession, result["session_id"]).status = "running"
        db.commit()
    return result


def handoff(client, session):
    return client.post("/handoffs", json={"session_id": session["session_id"], "name": "Alex", "reason": "Help"}, headers=bearer(session["session_token"]))


def test_demo_gating_unknown_and_metadata(monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEYS", "admin")
    with TestClient(app) as client:
        assert client.get("/demo/agents/sample").status_code == 404
        assert client.post("/demo/agents/sample/sessions", json={}).status_code == 404
        monkeypatch.delenv("PLATFORM_API_KEYS")
        demo(client, monkeypatch)
        monkeypatch.setenv("PLATFORM_API_KEYS", "admin")
        assert client.get("/demo/agents/missing").json()["detail"] == "demo_agent_not_found"
        assert client.post("/demo/agents/missing/sessions", json={}).status_code == 404
        assert client.get("/demo/agents/sample").json() == {"key": "sample", "name": "Demo", "description": "", "enabled_languages": ["en", "ko"]}


def test_demo_response_limits_and_absolute_duration(monkeypatch, runtime, livekit):
    with TestClient(app) as client:
        session = source(client, monkeypatch)
        assert session["session_token"].startswith("vls_")
        assert session["connection"]["url"] == "ws://livekit.test"
        with SessionLocal() as db:
            record = db.get(VoiceSession, session["session_id"])
            record.started_at = utcnow() - timedelta(seconds=301)
            record.last_event_at = utcnow()
            db.commit()
            assert record.id in reaper.reap_stale_sessions(db)
        calls.reset_limits()
        monkeypatch.setenv("PLATFORM_CALLS_PER_IP_HOUR", "1")
        assert client.post("/demo/agents/sample/sessions", json={}).status_code == 201
        assert client.post("/demo/agents/sample/sessions", json={}).status_code == 429
        calls.reset_limits()
        monkeypatch.setenv("PLATFORM_CALL_OPEN_PER_IP_MINUTE", "1")
        assert client.get("/demo/agents/sample").status_code == 200
        assert client.get("/demo/agents/sample").status_code == 429


@pytest.mark.parametrize("admin_auth", [False, True])
def test_handoff_auth_staff_accept_transcript(monkeypatch, runtime, livekit, admin_auth):
    with TestClient(app) as client:
        session = source(client, monkeypatch)
        if admin_auth:
            monkeypatch.setenv("PLATFORM_API_KEYS", "admin")
        body = {"session_id": session["session_id"]}
        assert client.post("/handoffs", json=body).status_code == 401
        assert client.post("/handoffs", json=body, headers=bearer("vls_wrong")).status_code == 403
        created = handoff(client, session)
        assert created.status_code == 201, created.text
        result = created.json()
        assert "invite_code" not in result and result["session_token"].startswith("vls_")
        assert handoff(client, session).status_code == 409
        assert client.get("/staff/handoffs").status_code == 404
        monkeypatch.setenv("PLATFORM_STAFF_KEY", "staff-secret")
        assert client.get("/staff/handoffs").status_code == 401
        assert client.get("/staff/handoffs", headers={"X-Staff-Key": "wrong"}).status_code == 401
        staff = {"X-Staff-Key": "staff-secret"}
        items = client.get("/staff/handoffs", headers=staff).json()
        item = next(i for i in items if i["id"] == result["handoff_id"])
        assert "invite_code" not in item and item["requester_name"] == "Alex"
        with SessionLocal() as db:
            db.add(Message(conversation_id=session["conversation_id"], role="user", content="Need help"))
            db.commit()
            assert db.get(VoiceSession, session["session_id"]).status == "ended"
        path = f"/staff/handoffs/{item['id']}"
        transcript = client.get(path + "/transcript", headers=staff).json()
        assert transcript[0]["content"] == "Need help"
        accepted = client.post(path + "/accept", headers=staff)
        assert accepted.status_code == 200
        assert client.post(path + "/accept", headers=staff).status_code == 409
        joined = client.post(f"/calls/invites/{accepted.json()['invite_code']}/join", json={})
        assert joined.status_code == 200, joined.text
        assert client.post(f"/calls/{result['session_id']}/end", headers=bearer(result["session_token"])).status_code == 200
        assert item["id"] not in [i["id"] for i in client.get("/staff/handoffs", headers=staff).json()]
        with SessionLocal() as db:
            assert db.get(Handoff, item["id"]).status == "ended"


def test_expired_handoff_cannot_be_accepted(monkeypatch, runtime, livekit):
    with TestClient(app) as client:
        session = source(client, monkeypatch)
        result = handoff(client, session).json()
        with SessionLocal() as db:
            call = db.get(VoiceSession, result["session_id"])
            contract = dict(call.runtime_contract)
            contract["call_invite"] = {**contract["call_invite"], "expires_at": 1}
            call.runtime_contract = contract
            db.commit()
        monkeypatch.setenv("PLATFORM_STAFF_KEY", "staff")
        assert client.post(f"/staff/handoffs/{result['handoff_id']}/accept", headers={"X-Staff-Key": "staff"}).status_code == 409
        with SessionLocal() as db:
            assert db.get(Handoff, result["handoff_id"]).status == "expired"


def test_foreign_session_token_and_inactive_source_rejected(monkeypatch, runtime, livekit):
    with TestClient(app) as client:
        first = source(client, monkeypatch)
        second = source(client, monkeypatch)
        assert client.post("/handoffs", json={"session_id": second["session_id"]}, headers=bearer(first["session_token"])).status_code == 403
        assert client.post("/handoffs", json={"session_id": first["session_id"], "language": "xx"}, headers=bearer(first["session_token"])).status_code == 422
        with SessionLocal() as db:
            assert db.get(VoiceSession, first["session_id"]).status == "running"
        client.post(f"/voice/sessions/{first['session_id']}/end", headers=bearer(first["session_token"]))
        assert handoff(client, first).status_code == 409


def test_reaper_ends_handoff_lazily(monkeypatch, runtime, livekit):
    with TestClient(app) as client:
        result = handoff(client, source(client, monkeypatch)).json()
        with SessionLocal() as db:
            record = db.get(VoiceSession, result["session_id"])
            record.started_at = utcnow() - timedelta(hours=1)
            record.last_event_at = None
            db.commit()
            assert record.id in reaper.reap_stale_sessions(db)
        monkeypatch.setenv("PLATFORM_STAFF_KEY", "staff")
        client.get("/staff/handoffs", headers={"X-Staff-Key": "staff"})
        with SessionLocal() as db:
            assert db.get(Handoff, result["handoff_id"]).status == "ended"


def test_staff_header_cors():
    with TestClient(app) as client:
        response = client.options("/staff/handoffs", headers={
            "Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Staff-Key",
        })
        assert response.status_code == 200


def test_handoff_rate_limit_and_configured_staff_language(monkeypatch, runtime, livekit):
    with TestClient(app) as client:
        session = source(client, monkeypatch)
        calls.reset_limits()
        monkeypatch.setenv("PLATFORM_CALLS_PER_IP_HOUR", "1")
        monkeypatch.setenv("PLATFORM_HANDOFF_STAFF_LANGUAGE", "fr")
        response = handoff(client, session)
        assert response.status_code == 201, response.text
        with SessionLocal() as db:
            item = db.get(Handoff, response.json()["handoff_id"])
            assert item.staff_language == "fr"
            contract = db.get(VoiceSession, item.call_session_id).runtime_contract
            assert contract["interpreter"]["languages"] == ["en", "fr"]
        assert handoff(client, session).status_code == 429


def test_fixed_calls_organisation_cannot_be_overridden(monkeypatch, runtime, livekit):
    from uuid import uuid4
    from src.platform.models import Agent, Organization

    with TestClient(app) as client:
        session = source(client, monkeypatch)
        result = handoff(client, session).json()
        foreign_org = str(uuid4())
        with SessionLocal() as db:
            db.add(Organization(id=foreign_org, name="Other", slug=f"other-{foreign_org}"))
            db.flush()
            agent = Agent(organization_id=foreign_org, name="Private")
            db.add(agent)
            db.flush()
            monkeypatch.setenv("PLATFORM_DEMO_AGENTS", f"private={agent.id}")
            db.get(Handoff, result["handoff_id"]).organization_id = foreign_org
            db.get(VoiceSession, session["session_id"]).organization_id = foreign_org
            db.commit()
        headers = {"X-Organization-Id": foreign_org, "X-Staff-Key": "staff"}
        assert client.get("/demo/agents/private", headers=headers).status_code == 404
        assert client.post("/demo/agents/private/sessions", json={}, headers=headers).status_code == 404
        monkeypatch.setenv("PLATFORM_STAFF_KEY", "staff")
        path = f"/staff/handoffs/{result['handoff_id']}"
        assert client.post(path + "/accept", headers=headers).status_code == 404
        assert client.get(path + "/transcript", headers=headers).status_code == 404
        assert result["handoff_id"] not in [i["id"] for i in client.get("/staff/handoffs", headers=headers).json()]
        assert handoff(client, session).status_code == 403
