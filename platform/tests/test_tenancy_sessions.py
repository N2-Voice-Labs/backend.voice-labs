"""Organisation isolation for sessions, deployments, campaigns, SIP and runtime ingest.

Two organisations (A and B) with admin and member ``vlk_`` keys, ``PLATFORM_API_KEYS``
set (the self-HTTP-call regression: SIP code must not need credentials), and no HTTP
between the Platform and itself.
"""

import asyncio
import os
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient
from livekit.protocol.sip import SIPParticipantInfo

from src.platform import auth, campaigns, sip
from src.platform.context import AuthContext
from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import (
    ApiKey,
    AuditLog,
    Campaign,
    CampaignTarget,
    Conversation,
    Deployment,
    Message,
    Organization,
    RuntimeEvent,
    Voice,
    VoiceSession,
)
from src.platform.models import Agent
from src.platform.organizations import ensure_default_org
from src.platform.tenancy import scoped_session_for

SUPER_KEY = "tenancy-sessions-super"
RUNTIME_TOKEN = "runtime-secret"

# (method, path) of every route covered here; a later completeness check reads this.
ISOLATION_CASES = [
    ("POST", "/voice/sessions"),
    ("GET", "/voice/sessions/{session_id}"),
    ("POST", "/voice/sessions/{session_id}/participants"),
    ("POST", "/voice/sessions/{session_id}/phone-participants"),
    ("POST", "/voice/sessions/{session_id}/ensure-running"),
    ("POST", "/voice/sessions/{session_id}/end"),
    ("POST", "/deployments"),
    ("GET", "/deployments"),
    ("GET", "/deployments/{deployment_id}"),
    ("PATCH", "/deployments/{deployment_id}"),
    ("DELETE", "/deployments/{deployment_id}"),
    ("POST", "/deployments/{deployment_id}/bridge-call"),
    ("POST", "/campaigns"),
    ("GET", "/campaigns"),
    ("GET", "/campaigns/{campaign_id}"),
    ("PATCH", "/campaigns/{campaign_id}"),
    ("POST", "/campaigns/{campaign_id}/targets"),
    ("GET", "/campaigns/{campaign_id}/targets"),
    ("POST", "/campaigns/{campaign_id}/status"),
    ("POST", "/campaigns/{campaign_id}/dial-next"),
    ("POST", "/runtime/events"),
]


def phone() -> str:
    return f"+1{uuid4().int % 10**10:010d}"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER_KEY)
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", RUNTIME_TOKEN)
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)
    for name in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_SIP_TRUNK_ID",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def make_org() -> str:
    org_id = str(uuid4())
    with SessionLocal() as db:
        db.add(Organization(id=org_id, name="Org", slug=f"org-{org_id}", status="active"))
        db.commit()
    return org_id


def make_key(org_id: str, role: str) -> dict[str, str]:
    plaintext, prefix, digest = auth.generate_api_key()
    with SessionLocal() as db:
        db.add(
            ApiKey(
                id=str(uuid4()),
                organization_id=org_id,
                name="tenancy-sessions",
                key_prefix=prefix,
                key_hash=digest,
                role=role,
            )
        )
        db.commit()
    return {"Authorization": f"Bearer {plaintext}"}


class Org:
    def __init__(self) -> None:
        self.id = make_org()
        self.admin = make_key(self.id, "admin")
        self.member = make_key(self.id, "member")


@pytest.fixture
def org_a() -> Org:
    return Org()


@pytest.fixture
def org_b() -> Org:
    return Org()


def make_agent(client, org: Org, **extra) -> dict:
    response = client.post(
        "/agents", json={"name": f"agent-{uuid4()}", **extra}, headers=org.admin
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_interpreter(client, org: Org, **interpreter) -> dict:
    return make_agent(
        client,
        org,
        mode="interpreter",
        interpreter={"languages": ["en", "ru"], "direction": "auto", **interpreter},
    )


def make_deployment(client, org: Org, agent_id: str, channel="web", **extra) -> dict:
    body = {"agent_id": agent_id, "name": f"dep-{uuid4()}", "channel": channel, **extra}
    if channel == "sip":
        body.setdefault("phone_number", phone())
    response = client.post("/deployments", json=body, headers=org.admin)
    assert response.status_code == 201, response.text
    return response.json()


def make_session(client, org: Org, agent_id: str, **extra) -> dict:
    response = client.post(
        "/voice/sessions", json={"agent_id": agent_id, **extra}, headers=org.member
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_campaign(client, org: Org, targets=1, **extra) -> dict:
    agent = make_agent(client, org)
    deployment = make_deployment(client, org, agent["id"], channel="sip")
    response = client.post(
        "/campaigns",
        json={
            "name": f"camp-{uuid4()}",
            "agent_id": agent["id"],
            "deployment_id": deployment["id"],
            **extra,
        },
        headers=org.admin,
    )
    assert response.status_code == 201, response.text
    campaign = response.json()
    if targets:
        added = client.post(
            f"/campaigns/{campaign['id']}/targets",
            json={"targets": [{"phone_number": phone()} for _ in range(targets)]},
            headers=org.admin,
        )
        assert added.status_code == 200, added.text
    return campaign


def make_voice(org_id: str, source="clone", status="ready") -> str:
    voice_id = str(uuid4())
    with SessionLocal() as db:
        db.add(
            Voice(
                id=voice_id,
                organization_id=org_id,
                name="voice",
                source=source,
                status=status,
                provider_voice_id=f"pv-{voice_id[:8]}",
            )
        )
        db.commit()
    return voice_id


# --------------------------------------------------------------------- sessions


def test_session_routes_are_isolated(client, org_a, org_b):
    agent_a = make_agent(client, org_a)
    session_a = make_session(client, org_a, agent_a["id"])
    sid = session_a["session_id"]

    # Org B cannot read, end, ensure-running or add participants to org A's session.
    assert client.get(f"/voice/sessions/{sid}", headers=org_b.admin).status_code == 404
    for suffix, body in (
        ("end", None),
        ("ensure-running", None),
        ("participants", {"language": "en"}),
        ("phone-participants", {"language": "en", "phone_number": phone()}),
    ):
        response = client.post(f"/voice/sessions/{sid}/{suffix}", json=body, headers=org_b.admin)
        assert response.status_code == 404, (suffix, response.text)
        assert response.json()["detail"] == "voice_session_not_found"
    # Untouched by the attempts.
    ok = client.get(f"/voice/sessions/{sid}", headers=org_a.member)
    assert ok.status_code == 200 and ok.json()["status"] == "created"

    # Org A's own routes work, and the rows carry org A.
    assert client.post(f"/voice/sessions/{sid}/end", headers=org_a.member).status_code == 200
    with SessionLocal() as db:
        session = db.get(VoiceSession, sid)
        assert session.organization_id == org_a.id
        assert db.get(Conversation, session.conversation_id).organization_id == org_a.id
        assert session.runtime_contract["organization_id"] == org_a.id


def test_create_session_rejects_foreign_references(client, org_a, org_b):
    agent_a = make_agent(client, org_a)
    dep_a = make_deployment(client, org_a, agent_a["id"])
    agent_b = make_agent(client, org_b)
    dep_b = make_deployment(client, org_b, agent_b["id"])

    r = client.post("/voice/sessions", json={"agent_id": agent_a["id"]}, headers=org_b.member)
    assert (r.status_code, r.json()["detail"]) == (404, "agent_not_found")
    r = client.post(
        "/voice/sessions",
        json={"agent_id": agent_b["id"], "deployment_id": dep_a["id"]},
        headers=org_b.member,
    )
    assert (r.status_code, r.json()["detail"]) == (404, "deployment_not_found")
    r = client.post(
        "/voice/sessions",
        json={"agent_id": agent_b["id"], "deployment_id": dep_b["id"]},
        headers=org_b.member,
    )
    assert r.status_code == 201


def test_foreign_voice_is_unavailable(client, org_a, org_b):
    voice_a = make_voice(org_a.id)
    interpreter_b = make_interpreter(client, org_b, direction="one_way", engine="cascaded")
    response = client.post(
        "/voice/sessions",
        json={
            "agent_id": interpreter_b["id"],
            "participant": {"language": "en", "voice_profile_id": voice_a},
        },
        headers=org_b.member,
    )
    assert (response.status_code, response.json()["detail"]) == (
        409,
        "selected_voice_unavailable",
    )
    # An agent whose voice_profile_id points at a foreign voice is unavailable too.
    agent_b = make_agent(client, org_b)
    with SessionLocal() as db:
        db.get(Agent, agent_b["id"]).voice_profile_id = voice_a
        db.commit()
    response = client.post(
        "/voice/sessions", json={"agent_id": agent_b["id"]}, headers=org_b.member
    )
    assert (response.status_code, response.json()["detail"]) == (
        409,
        "selected_voice_unavailable",
    )


def test_voice_used_audited_for_clone_and_design_only(client, org_a):
    results = {}
    for source in ("clone", "design", "library"):
        voice_id = make_voice(org_a.id, source=source)
        agent = make_agent(client, org_a)
        with SessionLocal() as db:
            db.get(Agent, agent["id"]).voice_profile_id = voice_id
            db.commit()
        session = make_session(client, org_a, agent["id"])
        results[source] = (voice_id, session["session_id"])
    with SessionLocal() as db:
        rows = db.query(AuditLog).filter(AuditLog.action == "voice.used").all()
        by_voice = {row.resource_id: row for row in rows}
    assert results["clone"][0] in by_voice and results["design"][0] in by_voice
    assert results["library"][0] not in by_voice
    row = by_voice[results["clone"][0]]
    assert row.organization_id == org_a.id
    assert row.resource_type == "voice"
    assert row.metadata_json["session_id"] == results["clone"][1]


def test_session_token_still_reaches_its_own_session_only(client, org_a, org_b):
    session_a = make_session(client, org_a, make_agent(client, org_a)["id"])
    session_b = make_session(client, org_b, make_agent(client, org_b)["id"])
    token = {"Authorization": f"Bearer {session_a['session_token']}"}
    own = client.get(f"/voice/sessions/{session_a['session_id']}", headers=token)
    assert own.status_code == 200
    assert client.get(f"/voice/sessions/{session_b['session_id']}", headers=token).status_code == 403
    assert client.post(f"/voice/sessions/{session_a['session_id']}/end", headers=token).status_code == 200
    # Session tokens never create sessions.
    assert (
        client.post("/voice/sessions", json={"agent_id": make_agent(client, org_a)["id"]}, headers=token).status_code
        == 403
    )


def test_session_roles(client, org_a):
    agent = make_agent(client, org_a)
    # Members create and end sessions; unauthenticated callers do not.
    session = make_session(client, org_a, agent["id"])
    assert client.post("/voice/sessions", json={"agent_id": agent["id"]}).status_code == 401
    assert client.post(f"/voice/sessions/{session['session_id']}/end", headers=org_a.member).status_code == 200


# ------------------------------------------------------------------ deployments


def test_deployment_routes_are_isolated(client, org_a, org_b):
    agent_a = make_agent(client, org_a)
    dep_a = make_deployment(client, org_a, agent_a["id"], channel="sip")
    did = dep_a["id"]

    assert client.get("/deployments", headers=org_b.admin).json() == []
    assert [d["id"] for d in client.get("/deployments", headers=org_a.member).json()] == [did]
    assert client.get(f"/deployments/{did}", headers=org_b.admin).status_code == 404
    assert client.patch(f"/deployments/{did}", json={"name": "x"}, headers=org_b.admin).status_code == 404
    assert client.delete(f"/deployments/{did}", headers=org_b.admin).status_code == 404
    bridge = client.post(
        f"/deployments/{did}/bridge-call",
        json={"from_number": phone(), "to_number": phone()},
        headers=org_b.member,
    )
    assert (bridge.status_code, bridge.json()["detail"]) == (404, "deployment_not_found")
    # Untouched.
    assert client.get(f"/deployments/{did}", headers=org_a.admin).json()["status"] == "active"

    # A body referencing another org's agent is a 422, same as a nonexistent one.
    r = client.post(
        "/deployments",
        json={"agent_id": agent_a["id"], "name": "x", "channel": "web"},
        headers=org_b.admin,
    )
    assert (r.status_code, r.json()["detail"]) == (422, "invalid_agent_id")
    missing = client.post(
        "/deployments",
        json={"agent_id": str(uuid4()), "name": "x", "channel": "web"},
        headers=org_b.admin,
    )
    assert (missing.status_code, missing.json()) == (r.status_code, r.json())
    # A missing agent is still the old 422.
    r = client.post(
        "/deployments",
        json={"agent_id": str(uuid4()), "name": "x", "channel": "web"},
        headers=org_b.admin,
    )
    assert (r.status_code, r.json()["detail"]) == (422, "invalid_agent_id")


def test_deployment_role_requirements(client, org_a):
    agent = make_agent(client, org_a)
    dep = make_deployment(client, org_a, agent["id"])
    body = {"agent_id": agent["id"], "name": "x", "channel": "web"}
    for response in (
        client.post("/deployments", json=body, headers=org_a.member),
        client.patch(f"/deployments/{dep['id']}", json={"name": "y"}, headers=org_a.member),
        client.delete(f"/deployments/{dep['id']}", headers=org_a.member),
    ):
        assert response.status_code == 403
        assert response.json()["detail"] == {"code": "insufficient_role"}
    assert client.get("/deployments", headers=org_a.member).status_code == 200
    assert client.get(f"/deployments/{dep['id']}", headers=org_a.member).status_code == 200


def test_active_phone_number_in_use_is_409(client, org_a, org_b):
    agent_a = make_agent(client, org_a)
    agent_b = make_agent(client, org_b)
    number = phone()
    first = make_deployment(client, org_a, agent_a["id"], channel="sip", phone_number=number)
    for org, agent in ((org_a, agent_a), (org_b, agent_b)):
        r = client.post(
            "/deployments",
            json={"agent_id": agent["id"], "name": "dup", "channel": "sip", "phone_number": number},
            headers=org.admin,
        )
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == {"code": "phone_number_in_use"}
    # Archiving frees the number; moving another deployment onto a taken one is 409.
    other = make_deployment(client, org_a, agent_a["id"], channel="sip")
    r = client.patch(
        f"/deployments/{other['id']}", json={"phone_number": number}, headers=org_a.admin
    )
    assert (r.status_code, r.json()["detail"]) == (409, {"code": "phone_number_in_use"})
    assert client.delete(f"/deployments/{first['id']}", headers=org_a.admin).status_code == 204
    r = client.patch(
        f"/deployments/{other['id']}", json={"phone_number": number}, headers=org_a.admin
    )
    assert r.status_code == 200


# -------------------------------------------------------------------- campaigns


def test_campaign_routes_are_isolated(client, org_a, org_b):
    campaign = make_campaign(client, org_a, targets=2)
    cid = campaign["id"]

    assert client.get("/campaigns", headers=org_b.admin).json() == []
    assert [c["id"] for c in client.get("/campaigns", headers=org_a.member).json()] == [cid]
    for method, path, body in (
        ("GET", f"/campaigns/{cid}", None),
        ("PATCH", f"/campaigns/{cid}", {"name": "x"}),
        ("POST", f"/campaigns/{cid}/targets", {"targets": [{"phone_number": phone()}]}),
        ("GET", f"/campaigns/{cid}/targets", None),
        ("POST", f"/campaigns/{cid}/status", {"status": "scheduled"}),
        ("POST", f"/campaigns/{cid}/dial-next", None),
    ):
        response = client.request(method, path, json=body, headers=org_b.admin)
        assert response.status_code == 404, (method, path, response.text)
        assert response.json()["detail"] == "campaign_not_found"
    assert len(client.get(f"/campaigns/{cid}/targets", headers=org_a.member).json()) == 2
    assert client.get(f"/campaigns/{cid}", headers=org_a.admin).json()["status"] == "draft"

    # Foreign agent / deployment in the body -> 422, identical to a nonexistent id.
    agent_b = make_agent(client, org_b)
    dep_b = make_deployment(client, org_b, agent_b["id"], channel="sip")
    base = {"name": "x", "agent_id": campaign["agent_id"], "deployment_id": campaign["deployment_id"]}
    r = client.post("/campaigns", json=base, headers=org_b.admin)
    assert (r.status_code, r.json()["detail"]) == (422, "invalid_agent_id")
    missing = client.post(
        "/campaigns", json={**base, "agent_id": str(uuid4())}, headers=org_b.admin
    )
    assert (missing.status_code, missing.json()) == (r.status_code, r.json())
    r = client.post(
        "/campaigns", json={**base, "agent_id": agent_b["id"]}, headers=org_b.admin
    )
    assert (r.status_code, r.json()["detail"]) == (422, "invalid_deployment_id")
    missing = client.post(
        "/campaigns",
        json={**base, "agent_id": agent_b["id"], "deployment_id": str(uuid4())},
        headers=org_b.admin,
    )
    assert (missing.status_code, missing.json()) == (r.status_code, r.json())
    r = client.post(
        "/campaigns",
        json={**base, "agent_id": agent_b["id"], "deployment_id": dep_b["id"]},
        headers=org_b.admin,
    )
    assert r.status_code == 201


def test_campaign_role_requirements(client, org_a):
    campaign = make_campaign(client, org_a)
    cid = campaign["id"]
    for method, path, body in (
        ("POST", "/campaigns", {"name": "x", "agent_id": campaign["agent_id"], "deployment_id": campaign["deployment_id"]}),
        ("PATCH", f"/campaigns/{cid}", {"name": "x"}),
        ("POST", f"/campaigns/{cid}/targets", {"targets": [{"phone_number": phone()}]}),
        ("POST", f"/campaigns/{cid}/status", {"status": "scheduled"}),
        ("POST", f"/campaigns/{cid}/dial-next", None),
    ):
        response = client.request(method, path, json=body, headers=org_a.member)
        assert response.status_code == 403, (method, path)
        assert response.json()["detail"] == {"code": "insufficient_role"}
    for path in ("/campaigns", f"/campaigns/{cid}", f"/campaigns/{cid}/targets"):
        assert client.get(path, headers=org_a.member).status_code == 200


# ------------------------------------------------------- SIP / in-process sessions


class FakeLiveKit:
    def __init__(self) -> None:
        self.created: list = []
        self.removed: list = []
        outer = self

        class Sip:
            async def create_sip_participant(self, request):
                outer.created.append(request)
                return SIPParticipantInfo(
                    participant_id=f"PA_{len(outer.created)}",
                    participant_identity=request.participant_identity,
                    room_name=request.room_name,
                )

        class Room:
            async def remove_participant(self, request):
                outer.removed.append((request.room, request.identity))

        self.sip = Sip()
        self.room = Room()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def livekit_sip(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://lk.example.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("LIVEKIT_SIP_TRUNK_ID", "ST_trunk")
    monkeypatch.setattr(campaigns, "POLL_INTERVAL_S", 0)
    monkeypatch.setattr(campaigns, "CALL_WAIT_TIMEOUT_S", 0)
    started: list = []
    monkeypatch.setattr(campaigns, "start_dispatcher", lambda *a: started.append(a))
    fake = FakeLiveKit()
    monkeypatch.setattr(sip, "_new_client", lambda: fake)
    return fake


def test_sip_module_makes_no_http_self_calls():
    assert not hasattr(sip, "httpx") and not hasattr(sip, "http_transport")


def test_bridge_call_works_with_api_keys_set_and_stays_in_org(client, org_a, org_b, livekit_sip):
    interpreter = make_interpreter(client, org_a)
    dep = make_deployment(client, org_a, interpreter["id"], channel="sip")
    body = {"from_number": "+14155550100", "to_number": "+74955550100"}

    foreign = client.post(f"/deployments/{dep['id']}/bridge-call", json=body, headers=org_b.member)
    assert foreign.status_code == 404
    assert livekit_sip.created == []

    response = client.post(f"/deployments/{dep['id']}/bridge-call", json=body, headers=org_a.member)
    assert response.status_code == 201, response.text
    result = response.json()
    assert len(livekit_sip.created) == 2
    with SessionLocal() as db:
        session = db.get(VoiceSession, result["session_id"])
        assert session.organization_id == org_a.id
        conversation = db.get(Conversation, result["conversation_id"])
        assert conversation.organization_id == org_a.id
        assert conversation.deployment_id == dep["id"]


def test_bridge_call_cleans_up_in_org_when_second_leg_fails(
    client, org_a, livekit_sip, monkeypatch
):
    interpreter = make_interpreter(client, org_a)
    dep = make_deployment(client, org_a, interpreter["id"], channel="sip")
    calls = []

    async def dial(number, room, identity, **kwargs):
        calls.append(number)
        if len(calls) == 2:
            raise sip.SipDialError("busy")
        return "PA_1"

    monkeypatch.setattr(sip, "dial", dial)
    response = client.post(
        f"/deployments/{dep['id']}/bridge-call",
        json={"from_number": "+14155550100", "to_number": "+74955550100"},
        headers=org_a.member,
    )
    assert response.status_code == 502
    with SessionLocal() as db:
        sessions = db.query(VoiceSession).filter(VoiceSession.organization_id == org_a.id).all()
        assert [s.status for s in sessions if s.runtime_contract["agent_id"] == interpreter["id"]] == ["ended"]


def run_batch(campaign_id: str, **kwargs):
    asyncio.run(campaigns.run_campaign_batch(campaign_id, **kwargs))


def test_dispatcher_and_dial_next_run_under_campaign_org(client, org_a, org_b, livekit_sip):
    campaign = make_campaign(client, org_a, targets=2)
    cid = campaign["id"]
    for step in ("scheduled", "running"):
        r = client.post(f"/campaigns/{cid}/status", json={"status": step}, headers=org_a.admin)
        assert r.status_code == 200, r.text

    assert client.post(f"/campaigns/{cid}/dial-next", headers=org_b.admin).status_code == 404
    assert livekit_sip.created == []

    response = client.post(f"/campaigns/{cid}/dial-next", headers=org_a.admin)
    assert response.status_code == 200, response.text
    assert response.json()["result"] == "completed"

    run_batch(cid)  # dispatcher drains the remaining target under the campaign's org
    assert len(livekit_sip.created) == 2
    with SessionLocal() as db:
        assert db.get(Campaign, cid).status == "completed"
        targets = db.query(CampaignTarget).filter(CampaignTarget.campaign_id == cid).all()
        assert [t.status for t in targets] == ["completed", "completed"]
        assert {t.organization_id for t in targets} == {org_a.id}
        for target in targets:
            conversation = db.get(Conversation, target.conversation_id)
            assert conversation is not None and conversation.organization_id == org_a.id
            assert conversation.deployment_id == campaign["deployment_id"]
        sessions = db.query(VoiceSession).filter(
            VoiceSession.conversation_id.in_([t.conversation_id for t in targets])
        )
        assert {s.organization_id for s in sessions} == {org_a.id}


def test_dispatcher_org_is_the_campaign_org_not_the_default(
    client, org_a, livekit_sip, monkeypatch
):
    seen: list = []
    from src.platform import sessions

    original = sessions.open_voice_session

    async def spy(db, ctx, **kwargs):
        seen.append((ctx.org_id, ctx.principal, ctx.role, db.info.get("org_id")))
        return await original(db, ctx, **kwargs)

    monkeypatch.setattr(sessions, "open_voice_session", spy)
    campaign = make_campaign(client, org_a)
    for step in ("scheduled", "running"):
        client.post(f"/campaigns/{campaign['id']}/status", json={"status": step}, headers=org_a.admin)
    run_batch(campaign["id"])
    assert seen == [(org_a.id, "system", "admin", org_a.id)]


def test_session_service_system_ctx_is_org_bound(client, org_a, org_b):
    """open_voice_session with a system ctx cannot reach another org's agent."""

    from fastapi import HTTPException

    from src.platform import sessions

    agent_a = make_agent(client, org_a)
    with scoped_session_for(org_b.id) as db:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                sessions.open_voice_session(
                    db, sip.system_ctx(org_b.id), agent_id=agent_a["id"]
                )
            )
    assert (exc.value.status_code, exc.value.detail) == (404, "agent_not_found")


# ---------------------------------------------------------------- runtime ingest


def runtime_event(session: dict, agent_id: str, **overrides) -> dict:
    return {
        "event_id": str(uuid4()),
        "event_type": "TranscriptFinal",
        "session_id": session["session_id"],
        "conversation_id": session["conversation_id"],
        "agent_id": agent_id,
        "payload": {"role": "user", "content": "hello"},
        **overrides,
    }


def post_event(client, event: dict):
    return client.post("/runtime/events", json=event, headers={"X-Runtime-Token": RUNTIME_TOKEN})


def test_runtime_event_is_stored_with_the_session_org(client, org_a, org_b):
    agent_a = make_agent(client, org_a)
    session_a = make_session(client, org_a, agent_a["id"])
    # An organization_id in the body is ignored.
    event = runtime_event(session_a, agent_a["id"], organization_id=org_b.id)
    event["payload"]["organization_id"] = org_b.id
    response = post_event(client, event)
    assert response.status_code == 202, response.text
    with SessionLocal() as db:
        stored = db.query(RuntimeEvent).filter(RuntimeEvent.event_id == event["event_id"]).one()
        assert stored.organization_id == org_a.id
        message = db.query(Message).filter(Message.conversation_id == session_a["conversation_id"]).one()
        assert message.organization_id == org_a.id
    # The conversation is readable by org A only.
    path = f"/conversations/{session_a['conversation_id']}/messages"
    assert client.get(path, headers=org_a.member).status_code == 200
    assert client.get(path, headers=org_b.member).status_code == 404


def test_runtime_event_rejects_cross_org_mismatches(client, org_a, org_b):
    agent_a = make_agent(client, org_a)
    agent_b = make_agent(client, org_b)
    session_a = make_session(client, org_a, agent_a["id"])
    session_b = make_session(client, org_b, agent_b["id"])

    # Event naming org A's session but org B's conversation.
    mixed = runtime_event(session_a, agent_a["id"], conversation_id=session_b["conversation_id"])
    assert post_event(client, mixed).status_code == 409
    # Event for org A's session naming an agent of org B.
    wrong_agent = runtime_event(session_a, agent_b["id"])
    assert post_event(client, wrong_agent).status_code == 409

    # Corrupt rows: A's session+conversation, but the conversation's agent is B's.
    with SessionLocal() as db:
        conversation = db.get(Conversation, session_a["conversation_id"])
        conversation.agent_id = agent_b["id"]
        db.commit()
    crossed = runtime_event(session_a, agent_b["id"])
    response = post_event(client, crossed)
    assert (response.status_code, response.json()["detail"]) == (409, "session_agent_mismatch")
    # Corrupt rows: A's session whose conversation was stamped with org B.
    with SessionLocal() as db:
        conversation = db.get(Conversation, session_a["conversation_id"])
        conversation.agent_id = agent_a["id"]
        conversation.organization_id = org_b.id
        db.commit()
    response = post_event(client, runtime_event(session_a, agent_a["id"]))
    assert (response.status_code, response.json()["detail"]) == (409, "session_conversation_mismatch")
    with SessionLocal() as db:
        assert db.query(RuntimeEvent).filter(RuntimeEvent.session_id == session_a["session_id"]).count() == 0


def test_runtime_event_requires_runtime_token(client, org_a):
    agent = make_agent(client, org_a)
    session = make_session(client, org_a, agent["id"])
    event = runtime_event(session, agent["id"])
    assert client.post("/runtime/events", json=event, headers=org_a.admin).status_code == 401
    assert (
        client.post("/runtime/events", json=event, headers={"X-Runtime-Token": "nope"}).status_code
        == 401
    )
