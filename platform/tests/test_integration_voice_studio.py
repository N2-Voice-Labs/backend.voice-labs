import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient

from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import VoiceSession

TOKEN = "voice-studio-token"
HEADERS = {"X-Runtime-Token": TOKEN}


@pytest.fixture(autouse=True)
def runtime_token(monkeypatch):
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", TOKEN)
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)


def phone() -> str:
    return f"+1{uuid4().int % 10**10:010d}"


def make_agent(client: TestClient, **extra) -> dict:
    response = client.post("/agents", json={"name": f"agent-{uuid4()}", **extra})
    assert response.status_code == 201, response.text
    return response.json()


def make_tts_provider(client: TestClient) -> dict:
    return client.post(
        "/providers", json={"kind": "tts", "provider": "cartesia"}
    ).json()


def make_ready_voice(client: TestClient, provider_voice_id: str) -> dict:
    response = client.post(
        "/voices",
        json={
            "name": f"voice-{uuid4()}",
            "tts_provider_id": make_tts_provider(client)["id"],
            "provider_voice_id": provider_voice_id,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_session(client: TestClient, agent_id: str, **extra):
    return client.post("/voice/sessions", json={"agent_id": agent_id, **extra})


def runtime_contract(session_id: str) -> dict:
    with SessionLocal() as db:
        return db.get(VoiceSession, session_id).runtime_contract


def send_event(client, session, agent_id, event_type, payload=None, event_id=None):
    event = {
        "event_id": event_id or str(uuid4()),
        "event_type": event_type,
        "session_id": session["session_id"],
        "conversation_id": session["conversation_id"],
        "agent_id": agent_id,
        "payload": payload or {},
    }
    return client.post("/runtime/events", json=event, headers=HEADERS)


def test_agent_voice_profile_validation_and_runtime_voice_id() -> None:
    with TestClient(app) as client:
        missing = client.post(
            "/agents", json={"name": "x", "voice_profile_id": str(uuid4())}
        )
        assert missing.status_code == 422
        assert missing.json()["detail"] == "invalid_voice_profile_id"

        pending = client.post(
            "/voices/design",
            json={"name": "pending", "design_prompt": "calm and warm"},
        ).json()
        assert pending["status"] == "pending"
        agent = make_agent(client, voice_id="legacy-voice", voice_profile_id=pending["id"])
        assert agent["voice_profile_id"] == pending["id"]
        blocked = make_session(client, agent["id"])
        assert blocked.status_code == 409
        assert blocked.json()["detail"] == "selected_voice_unavailable"

        ready = make_ready_voice(client, "provider-voice-42")
        patched = client.patch(
            f"/agents/{agent['id']}", json={"voice_profile_id": ready["id"]}
        )
        assert patched.status_code == 200
        assert patched.json()["voice_profile_id"] == ready["id"]
        created = make_session(client, agent["id"])
        assert created.status_code == 201
        contract = runtime_contract(created.json()["session_id"])
        assert contract["voice_id"] == "provider-voice-42"
        assert "voice_profile_id" not in contract

        client.delete(f"/voices/{ready['id']}")
        invalid_patch = client.patch(
            f"/agents/{agent['id']}", json={"voice_profile_id": ready["id"]}
        )
        assert invalid_patch.status_code == 422
        assert invalid_patch.json()["detail"] == "invalid_voice_profile_id"
        # disabled voice already attached is unavailable at session time
        assert make_session(client, agent["id"]).status_code == 409

        cleared = client.patch(
            f"/agents/{agent['id']}", json={"voice_profile_id": None}
        ).json()
        assert cleared["voice_profile_id"] is None
        legacy = make_session(client, agent["id"])
        assert runtime_contract(legacy.json()["session_id"])["voice_id"] == "legacy-voice"


def test_deployment_session_validation_and_channel() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        other = make_agent(client)
        sip = client.post(
            "/deployments",
            json={"agent_id": agent["id"], "name": "sip", "channel": "sip", "phone_number": phone()},
        ).json()
        foreign = client.post(
            "/deployments",
            json={"agent_id": other["id"], "name": "web", "channel": "web"},
        ).json()
        paused = client.post(
            "/deployments",
            json={"agent_id": agent["id"], "name": "paused", "channel": "web"},
        ).json()
        client.patch(f"/deployments/{paused['id']}", json={"status": "paused"})

        before = len(client.get("/conversations", params={"agent_id": agent["id"]}).json())
        missing = make_session(client, agent["id"], deployment_id=str(uuid4()))
        assert (missing.status_code, missing.json()["detail"]) == (
            404,
            "deployment_not_found",
        )
        inactive = make_session(client, agent["id"], deployment_id=paused["id"])
        assert (inactive.status_code, inactive.json()["detail"]) == (
            409,
            "deployment_not_active",
        )
        mismatch = make_session(client, agent["id"], deployment_id=foreign["id"])
        assert (mismatch.status_code, mismatch.json()["detail"]) == (
            409,
            "deployment_agent_mismatch",
        )
        after = len(client.get("/conversations", params={"agent_id": agent["id"]}).json())
        assert after == before  # nothing persisted on validation failure

        ok = make_session(client, agent["id"], deployment_id=sip["id"])
        assert ok.status_code == 201
        conversation = client.get(f"/conversations/{ok.json()['conversation_id']}").json()
        assert conversation["deployment_id"] == sip["id"]
        assert conversation["channel"] == "sip"

        default = make_session(client, agent["id"]).json()
        default_conv = client.get(f"/conversations/{default['conversation_id']}").json()
        assert default_conv["deployment_id"] is None
        assert default_conv["channel"] == "web"
        assert default_conv["outcome"] is None
        assert default_conv["actions"] == []


def test_handoff_and_outcome_events_with_idempotency() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        session = make_session(client, agent["id"]).json()
        conv_url = f"/conversations/{session['conversation_id']}"

        event_id = str(uuid4())
        payload = {"reason": "caller asked for a human", "target": "sales"}
        first = send_event(client, session, agent["id"], "HandoffRequested", payload, event_id)
        assert first.status_code == 202 and first.json() == {"status": "accepted"}
        again = send_event(client, session, agent["id"], "HandoffRequested", payload, event_id)
        assert again.json() == {"status": "duplicate"}
        conversation = client.get(conv_url).json()
        assert conversation["outcome"] == "escalated"
        assert conversation["outcome_recorded_at"] is not None
        assert conversation["actions"] == [
            {"type": "handoff_requested", "reason": "caller asked for a human", "target": "sales"}
        ]
        assert client.get(f"{conv_url}/runtime-state").json()["session_status"] == "created"

        bad = send_event(client, session, agent["id"], "OutcomeRecorded", {"outcome": "nope"})
        assert (bad.status_code, bad.json()["detail"]) == (422, "invalid_outcome")
        bad_actions = send_event(
            client,
            session,
            agent["id"],
            "OutcomeRecorded",
            {"outcome": "booked", "actions": ["not-a-dict"]},
        )
        assert bad_actions.status_code == 422
        assert client.get(conv_url).json()["outcome"] == "escalated"

        outcome_id = str(uuid4())
        outcome = {
            "outcome": "booked",
            "summary": "Booked a demo",
            "intent": "book_demo",
            "actions": [{"type": "booking", "id": "b-1"}],
        }
        assert send_event(
            client, session, agent["id"], "OutcomeRecorded", outcome, outcome_id
        ).json() == {"status": "accepted"}
        assert send_event(
            client, session, agent["id"], "OutcomeRecorded", outcome, outcome_id
        ).json() == {"status": "duplicate"}
        conversation = client.get(conv_url).json()
        assert conversation["outcome"] == "booked"
        assert conversation["summary"] == "Booked a demo"
        assert conversation["intent"] == "book_demo"
        assert conversation["actions"] == [{"type": "booking", "id": "b-1"}]

        # a later handoff appends to the actions list exactly once
        send_event(client, session, agent["id"], "HandoffRequested", {})
        actions = client.get(conv_url).json()["actions"]
        assert [item["type"] for item in actions] == ["booking", "handoff_requested"]
        assert actions[1] == {"type": "handoff_requested", "reason": None, "target": None}

        invalid_handoff = send_event(
            client, session, agent["id"], "HandoffRequested", {"reason": 5}
        )
        assert invalid_handoff.status_code == 422


def test_tool_called_event_is_recorded_without_messages() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        session = make_session(client, agent["id"]).json()
        missing = send_event(client, session, agent["id"], "ToolCalled", {"arguments": {}})
        assert (missing.status_code, missing.json()["detail"]) == (
            422,
            "invalid_tool_event",
        )
        bad_args = send_event(
            client, session, agent["id"], "ToolCalled", {"tool_name": "t", "arguments": "x"}
        )
        assert bad_args.status_code == 422
        event_id = str(uuid4())
        ok = send_event(
            client,
            session,
            agent["id"],
            "ToolCalled",
            {"tool_name": "lookup", "arguments": {"q": 1}, "status": "ok"},
            event_id,
        )
        assert ok.status_code == 202
        assert (
            send_event(client, session, agent["id"], "ToolCalled", {"tool_name": "lookup"}, event_id).json()
            == {"status": "duplicate"}
        )
        conv_url = f"/conversations/{session['conversation_id']}"
        assert client.get(f"{conv_url}/messages").json() == []
        events = client.get(f"{conv_url}/runtime-state").json()["events"]
        assert [e["event_type"] for e in events] == ["ToolCalled"]
        assert events[0]["payload"]["tool_name"] == "lookup"
        assert client.get(conv_url).json()["outcome"] is None


def test_put_outcome_and_conversation_filters() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        dep = client.post(
            "/deployments",
            json={"agent_id": agent["id"], "name": "api", "channel": "api"},
        ).json()
        first = make_session(client, agent["id"]).json()
        second = make_session(client, agent["id"], deployment_id=dep["id"]).json()

        assert client.put(
            f"/conversations/{first['conversation_id']}/outcome",
            json={"outcome": "bogus"},
        ).json()["detail"] == "invalid_outcome"
        assert client.put(
            f"/conversations/{uuid4()}/outcome", json={"outcome": "resolved"}
        ).status_code == 404
        assert client.put(
            f"/conversations/{first['conversation_id']}/outcome",
            json={"outcome": "resolved", "actions": "nope"},
        ).status_code == 422

        response = client.put(
            f"/conversations/{first['conversation_id']}/outcome",
            json={
                "outcome": "resolved",
                "summary": "All good",
                "intent": "faq",
                "actions": [{"type": "note"}],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == first["conversation_id"]
        assert (body["outcome"], body["summary"], body["intent"]) == ("resolved", "All good", "faq")
        assert body["actions"] == [{"type": "note"}]
        assert body["outcome_recorded_at"] is not None

        def ids(**params):
            rows = client.get("/conversations", params={"agent_id": agent["id"], **params}).json()
            return [row["id"] for row in rows]

        assert set(ids()) == {first["conversation_id"], second["conversation_id"]}
        assert ids(outcome="resolved") == [first["conversation_id"]]
        assert ids(channel="api") == [second["conversation_id"]]
        assert ids(channel="web") == [first["conversation_id"]]
        assert ids(status="failed") == []
        assert set(ids(status="active")) == {first["conversation_id"], second["conversation_id"]}
        assert ids(outcome="booked") == []


def test_analytics_summary() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        dep = client.post(
            "/deployments",
            json={"agent_id": agent["id"], "name": "sip", "channel": "sip", "phone_number": phone()},
        ).json()
        one = make_session(client, agent["id"]).json()
        two = make_session(client, agent["id"], deployment_id=dep["id"]).json()
        client.put(
            f"/conversations/{one['conversation_id']}/outcome", json={"outcome": "booked"}
        )
        for session, content, language in (
            (one, "hello", "en"),
            (one, "privet", "ru"),
            (two, "???", None),
        ):
            send_event(
                client,
                session,
                agent["id"],
                "TranscriptFinal",
                {"content": content, "primary_language": language},
            )
        for value in (100, 200, 300, 400):
            send_event(
                client,
                one,
                agent["id"],
                "LatencyMeasured",
                {"metric": "stt", "value_ms": value},
            )
        send_event(client, two, agent["id"], "LatencyMeasured", {"metric": "llm", "value_ms": 50})
        send_event(client, two, agent["id"], "LatencyMeasured", {"metric": "llm", "value_ms": "bad"})
        send_event(client, two, agent["id"], "LatencyMeasured", {"value_ms": 5})

        summary = client.get("/analytics/summary", params={"agent_id": agent["id"]}).json()
        assert summary["conversations"]["total"] == 2
        assert summary["conversations"]["by_status"] == {"active": 2}
        assert summary["conversations"]["by_outcome"] == {"booked": 1, "unknown": 1}
        assert summary["conversations"]["by_channel"] == {"web": 1, "sip": 1}
        assert summary["languages"] == {"en": 1, "ru": 1, "unknown": 1}
        assert summary["latency"] == {
            "stt": {"count": 4, "p50": 200.0, "p95": 400.0},
            "llm": {"count": 1, "p50": 50.0, "p95": 50.0},
        }

        future = client.get(
            "/analytics/summary",
            params={"agent_id": agent["id"], "since": "2999-01-01T00:00:00Z"},
        ).json()
        assert future["conversations"]["total"] == 0
        assert future["languages"] == {} and future["latency"] == {}
        past = client.get(
            "/analytics/summary",
            params={"agent_id": agent["id"], "since": "2000-01-01T00:00:00Z"},
        ).json()
        assert past["conversations"]["total"] == 2
        empty = client.get("/analytics/summary", params={"agent_id": str(uuid4())}).json()
        assert empty["conversations"]["total"] == 0
