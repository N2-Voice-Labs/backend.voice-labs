"""Organisation isolation for /conversations/* and /analytics/summary (T5e)."""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient

from src.platform import auth
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import (
    ApiKey,
    Agent,
    Conversation,
    Message,
    Organization,
    RuntimeEvent,
    VoiceSession,
)
from src.platform.tenancy import bypass

ISOLATION_CASES = [
    ("GET", "/conversations"),
    ("GET", "/conversations/{conversation_id}"),
    ("PUT", "/conversations/{conversation_id}/outcome"),
    ("GET", "/conversations/{conversation_id}/runtime-state"),
    ("GET", "/conversations/{conversation_id}/messages"),
    ("GET", "/analytics/summary"),
]

SUPER = "t5e-superadmin-key"


def hdr(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def make_org() -> str:
    org_id = str(uuid4())
    with SessionLocal() as db:
        db.add(Organization(id=org_id, name="Org", slug=f"org-{org_id}", status="active"))
        db.commit()
    return org_id


def make_key(org_id: str, role: str = "member") -> str:
    plaintext, prefix, digest = auth.generate_api_key()
    with SessionLocal() as db:
        db.add(
            ApiKey(
                id=str(uuid4()),
                organization_id=org_id,
                name="t",
                key_prefix=prefix,
                key_hash=digest,
                role=role,
            )
        )
        db.commit()
    return plaintext


def seed(org_id: str, *, latency: float, engine: str, token: str | None = None) -> dict:
    """One agent/conversation/session/message/events set in ``org_id``."""

    agent_id, conv_id, sess_id = (str(uuid4()) for _ in range(3))
    with SessionLocal() as db, bypass(db):
        db.add(Agent(id=agent_id, organization_id=org_id, name="a"))
        db.flush()
        db.add(
            Conversation(
                id=conv_id,
                organization_id=org_id,
                agent_id=agent_id,
                channel="web",
                status="active",
            )
        )
        db.flush()
        db.add(
            VoiceSession(
                id=sess_id,
                organization_id=org_id,
                conversation_id=conv_id,
                room_name=f"room-{sess_id}",
                session_token_hash=auth.hash_token(token) if token else None,
            )
        )
        db.add(
            Message(
                organization_id=org_id,
                conversation_id=conv_id,
                role="user",
                content=f"hello from {org_id}",
                primary_language="fr" if latency > 1000 else "en",
            )
        )
        db.add(
            RuntimeEvent(
                organization_id=org_id,
                event_id=str(uuid4()),
                event_type="LatencyMeasured",
                conversation_id=conv_id,
                session_id=sess_id,
                payload={"metric": "llm_ttfb", "value_ms": latency},
            )
        )
        db.add(
            RuntimeEvent(
                organization_id=org_id,
                event_id=str(uuid4()),
                event_type="LatencyMeasured",
                conversation_id=conv_id,
                session_id=sess_id,
                payload={
                    "metric": "utterance_timeline",
                    "value_ms": latency,
                    "engine": engine,
                    "marks_ms": {"vad": latency / 2},
                },
            )
        )
        db.add(
            RuntimeEvent(
                organization_id=org_id,
                event_id=str(uuid4()),
                event_type="SessionStarted",
                conversation_id=conv_id,
                session_id=sess_id,
                payload={},
            )
        )
        db.commit()
    return {"conversation_id": conv_id, "session_id": sess_id, "agent_id": agent_id}


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER)
    with TestClient(app) as client:  # startup creates the schema
        org_a, org_b = make_org(), make_org()
        guest_token = "vls_" + uuid4().hex
        yield {
            "client": client,
            "org_a": org_a,
            "org_b": org_b,
            "a": seed(org_a, latency=100.0, engine="engine-a", token=guest_token),
            "b": seed(org_b, latency=5000.0, engine="engine-b"),
            "key_a": make_key(org_a, "member"),
            "key_b": make_key(org_b, "member"),
            "guest_a": guest_token,
        }


def test_get_conversation_cross_org_404(world):
    c = world["client"]
    cid_a = world["a"]["conversation_id"]
    assert c.get(f"/conversations/{cid_a}", headers=hdr(world["key_a"])).status_code == 200
    r = c.get(f"/conversations/{cid_a}", headers=hdr(world["key_b"]))
    assert r.status_code == 404
    assert r.json()["detail"] == "conversation_not_found"


def test_list_excludes_other_org(world):
    c = world["client"]
    ids_b = [x["id"] for x in c.get("/conversations", headers=hdr(world["key_b"])).json()]
    assert ids_b == [world["b"]["conversation_id"]]
    ids_a = [x["id"] for x in c.get("/conversations", headers=hdr(world["key_a"])).json()]
    assert ids_a == [world["a"]["conversation_id"]]
    # filters still work, and cannot reach across orgs via agent_id
    r = c.get(
        "/conversations",
        params={"agent_id": world["a"]["agent_id"]},
        headers=hdr(world["key_b"]),
    )
    assert r.status_code == 200 and r.json() == []


def test_outcome_cross_org_404_and_untouched(world):
    c = world["client"]
    cid_a = world["a"]["conversation_id"]
    body = {"outcome": "resolved", "summary": "hijack"}
    r = c.put(f"/conversations/{cid_a}/outcome", json=body, headers=hdr(world["key_b"]))
    assert r.status_code == 404
    assert r.json()["detail"] == "conversation_not_found"
    with SessionLocal() as db:
        assert db.get(Conversation, cid_a).outcome is None
    # same-org member may record an outcome
    ok = c.put(f"/conversations/{cid_a}/outcome", json=body, headers=hdr(world["key_a"]))
    assert ok.status_code == 200 and ok.json()["summary"] == "hijack"


def test_runtime_state_and_messages_cross_org_404(world):
    c = world["client"]
    cid_a = world["a"]["conversation_id"]
    for suffix in ("runtime-state", "messages"):
        assert (
            c.get(f"/conversations/{cid_a}/{suffix}", headers=hdr(world["key_b"])).status_code
            == 404
        )
        assert (
            c.get(f"/conversations/{cid_a}/{suffix}", headers=hdr(world["key_a"])).status_code
            == 200
        )
    state = c.get(
        f"/conversations/{cid_a}/runtime-state", headers=hdr(world["key_a"])
    ).json()
    assert state["session_id"] == world["a"]["session_id"]
    msgs = c.get(f"/conversations/{cid_a}/messages", headers=hdr(world["key_a"])).json()
    assert [m["content"] for m in msgs] == [f"hello from {world['org_a']}"]


def test_superadmin_with_org_header_is_scoped(world):
    c = world["client"]
    cid_a = world["a"]["conversation_id"]
    r = c.get(
        f"/conversations/{cid_a}",
        headers={**hdr(SUPER), "X-Organization-Id": world["org_b"]},
    )
    assert r.status_code == 404
    r = c.get(
        f"/conversations/{cid_a}",
        headers={**hdr(SUPER), "X-Organization-Id": world["org_a"]},
    )
    assert r.status_code == 200


def test_guest_session_token_reads_own_conversation(world):
    c = world["client"]
    token = world["guest_a"]
    cid_a = world["a"]["conversation_id"]
    for suffix in ("", "/messages", "/runtime-state"):
        assert c.get(f"/conversations/{cid_a}{suffix}", headers=hdr(token)).status_code == 200
    # ... and nothing else
    cid_b = world["b"]["conversation_id"]
    assert c.get(f"/conversations/{cid_b}", headers=hdr(token)).status_code == 403
    assert c.get("/conversations", headers=hdr(token)).status_code == 403
    assert c.get("/analytics/summary", headers=hdr(token)).status_code == 403
    r = c.put(
        f"/conversations/{cid_a}/outcome", json={"outcome": "resolved"}, headers=hdr(token)
    )
    assert r.status_code == 403


def test_analytics_isolated_between_orgs(world):
    c = world["client"]
    sum_a = c.get("/analytics/summary", headers=hdr(world["key_a"])).json()
    sum_b = c.get("/analytics/summary", headers=hdr(world["key_b"])).json()
    assert sum_a["conversations"]["total"] == 1
    assert sum_a["latency"]["llm_ttfb"] == {"count": 1, "p50": 100.0, "p95": 100.0}
    assert list(sum_a["timeline"]) == ["engine-a"]
    assert sum_a["languages"] == {"en": 1}
    assert sum_b["conversations"]["total"] == 1
    assert sum_b["latency"]["llm_ttfb"] == {"count": 1, "p50": 5000.0, "p95": 5000.0}
    assert list(sum_b["timeline"]) == ["engine-b"]
    assert sum_b["languages"] == {"fr": 1}


def test_analytics_unchanged_by_other_org_data(world):
    c = world["client"]
    before = c.get("/analytics/summary", headers=hdr(world["key_a"])).json()
    # Org B grows: more conversations, slower latency, a new engine.
    for latency in (9000.0, 9500.0):
        seed(world["org_b"], latency=latency, engine="engine-c")
    assert c.get("/analytics/summary", headers=hdr(world["key_a"])).json() == before
    after_b = c.get("/analytics/summary", headers=hdr(world["key_b"])).json()
    assert after_b["conversations"]["total"] == 3
    # agent_id of another org yields an empty summary, not that org's numbers
    r = c.get(
        "/analytics/summary",
        params={"agent_id": world["a"]["agent_id"]},
        headers=hdr(world["key_b"]),
    ).json()
    assert r["conversations"]["total"] == 0 and r["latency"] == {} and r["timeline"] == {}


def test_analytics_since_filter_still_applies(world):
    c = world["client"]
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    r = c.get("/analytics/summary", params={"since": future}, headers=hdr(world["key_a"]))
    assert r.status_code == 200 and r.json()["conversations"]["total"] == 0
