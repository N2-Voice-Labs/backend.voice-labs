"""Races around session termination: end vs. Runtime ingest, ensure-running capacity, reaper."""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient
from sqlalchemy import select

from src.platform import auth, reaper
from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import (
    Agent,
    ApiKey,
    Conversation,
    Organization,
    RuntimeEvent,
    VoiceSession,
)
from src.platform.organizations import ensure_default_org

RUNTIME_TOKEN = "races-runtime-token"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", RUNTIME_TOKEN)
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.delenv("PLATFORM_STALE_SESSION_SECS", raising=False)
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    yield


def make_org_and_headers() -> tuple[str, dict[str, str]]:
    org_id = str(uuid4())
    plaintext, prefix, digest = auth.generate_api_key()
    with SessionLocal() as db:
        db.add(Organization(id=org_id, name="Org", slug=f"org-{org_id}", status="active"))
        db.flush()
        db.add(
            ApiKey(
                id=str(uuid4()),
                organization_id=org_id,
                name="races",
                key_prefix=prefix,
                key_hash=digest,
                role="admin",
            )
        )
        db.commit()
    return org_id, {"Authorization": f"Bearer {plaintext}"}


def seed_session(
    org_id: str,
    *,
    status: str = "created",
    started_at: datetime | None = None,
    last_event_at: datetime | None = None,
) -> str:
    with SessionLocal() as db:
        agent = Agent(name="seed", organization_id=org_id)
        db.add(agent)
        db.flush()
        conversation = Conversation(
            agent_id=agent.id, organization_id=org_id, status="active"
        )
        db.add(conversation)
        db.flush()
        session_id = str(uuid4())
        db.add(
            VoiceSession(
                id=session_id,
                conversation_id=conversation.id,
                room_name=f"room-{session_id}",
                organization_id=org_id,
                status=status,
                started_at=started_at or datetime.now(timezone.utc),
                last_event_at=last_event_at,
                runtime_contract={"session_id": session_id},
            )
        )
        db.commit()
    return session_id


def load(session_id: str) -> tuple[str, str]:
    with SessionLocal() as db:
        session = db.get(VoiceSession, session_id)
        conversation = db.get(Conversation, session.conversation_id)
        return session.status, conversation.status


def set_status(session_id: str, status: str) -> None:
    """Commit a terminal status the way Runtime ingest does (session + conversation)."""

    with SessionLocal() as db:
        session = db.get(VoiceSession, session_id)
        session.status = status
        session.ended_at = datetime.now(timezone.utc)
        db.get(Conversation, session.conversation_id).status = status
        db.commit()


class Resp:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body
        self.request = httpx.Request("POST", "http://runtime.test")

    def json(self):
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=self.request,
                response=httpx.Response(
                    self.status_code, json=self._body, request=self.request
                ),
            )


# ------------------------------------------------------------------ M1: end vs ingest


def test_end_does_not_downgrade_failed_committed_during_runtime_call(monkeypatch) -> None:
    org_id, headers = make_org_and_headers()
    session_id = seed_session(org_id, status="running")

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, **_):
            # Runtime ingest commits SessionFailed while the /end call is in flight.
            set_status(session_id, "failed")
            return Resp(200, {"status": "ended"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: Client())
    with TestClient(app) as client:
        response = client.post(f"/voice/sessions/{session_id}/end", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "failed"}
    assert load(session_id) == ("failed", "failed")


def test_end_still_ends_an_active_session(monkeypatch) -> None:
    org_id, headers = make_org_and_headers()
    session_id = seed_session(org_id, status="running")

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, **_):
            return Resp(200, {"status": "ended"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: Client())
    with TestClient(app) as client:
        response = client.post(f"/voice/sessions/{session_id}/end", headers=headers)
    assert response.json() == {"status": "ended"}
    assert load(session_id) == ("ended", "ended")


# ------------------------------------------------- M2: ensure-running capacity mapping


@pytest.mark.parametrize(
    "body",
    [
        {"code": "runtime_at_capacity", "message": "full"},
        {"detail": {"code": "runtime_at_capacity"}},
    ],
)
def test_ensure_running_maps_runtime_at_capacity(monkeypatch, body) -> None:
    org_id, headers = make_org_and_headers()
    session_id = seed_session(org_id, status="created")

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, *_, **__):
            return Resp(200, {"status": "not_running"})

        async def post(self, *_, **__):
            return Resp(503, body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: Client())
    with TestClient(app) as client:
        response = client.post(
            f"/voice/sessions/{session_id}/ensure-running", headers=headers
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "runtime_at_capacity"
    assert detail["message"]
    assert load(session_id)[0] == "created"  # a retry may still succeed


def test_ensure_running_other_runtime_errors_stay_unavailable(monkeypatch) -> None:
    org_id, headers = make_org_and_headers()
    session_id = seed_session(org_id, status="created")

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, *_, **__):
            return Resp(200, {"status": "not_running"})

        async def post(self, *_, **__):
            return Resp(500, {"detail": "boom"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: Client())
    with TestClient(app) as client:
        response = client.post(
            f"/voice/sessions/{session_id}/ensure-running", headers=headers
        )
    assert response.status_code == 503
    assert "runtime_at_capacity" not in response.text


# ----------------------------------------------------------------------- L3: reaper


def stale_time() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=1)


def test_reaper_skips_session_ended_after_it_was_judged_stale(monkeypatch) -> None:
    org_id, _ = make_org_and_headers()
    session_id = seed_session(org_id, status="running", started_at=stale_time())

    original = reaper._reap_one

    def end_first(db, now, sid, *args):
        if sid == session_id:
            set_status(session_id, "ended")  # another worker/user ends it meanwhile
        return original(db, now, sid, *args)

    monkeypatch.setattr(reaper, "_reap_one", end_first)
    with SessionLocal() as db:
        reaped = reaper.reap_stale_sessions(db)
    assert session_id not in reaped
    assert load(session_id) == ("ended", "ended")
    with SessionLocal() as db:
        assert db.scalar(
            select(RuntimeEvent).where(
                RuntimeEvent.event_id == f"platform-stale-{session_id}"
            )
        ) is None


def test_reaper_skips_session_that_received_an_event_meanwhile(monkeypatch) -> None:
    org_id, _ = make_org_and_headers()
    session_id = seed_session(org_id, status="running", started_at=stale_time())
    original = reaper._reap_one

    def event_first(db, now, sid, *args):
        if sid == session_id:
            with SessionLocal() as other:
                other.get(VoiceSession, session_id).last_event_at = datetime.now(
                    timezone.utc
                )
                other.commit()
        return original(db, now, sid, *args)

    monkeypatch.setattr(reaper, "_reap_one", event_first)
    with SessionLocal() as db:
        assert session_id not in reaper.reap_stale_sessions(db)
    assert load(session_id)[0] == "running"


def test_reaper_event_integrity_error_does_not_abort_the_pass() -> None:
    org_id, _ = make_org_and_headers()
    first = seed_session(org_id, started_at=stale_time())
    second = seed_session(org_id, started_at=stale_time())
    with SessionLocal() as db:
        # A concurrent worker already wrote the first session's stale event.
        s = db.get(VoiceSession, first)
        db.add(
            RuntimeEvent(
                organization_id=org_id,
                event_id=f"platform-stale-{first}",
                event_type="SessionFailed",
                session_id=first,
                conversation_id=s.conversation_id,
                agent_id=db.get(Conversation, s.conversation_id).agent_id,
                payload={"reason": "stale"},
            )
        )
        db.commit()
    with SessionLocal() as db:
        reaped = reaper.reap_stale_sessions(db)
    assert {first, second} <= set(reaped)
    assert load(first) == ("failed", "failed")
    assert load(second) == ("failed", "failed")
    with SessionLocal() as db:
        events = db.scalars(
            select(RuntimeEvent).where(
                RuntimeEvent.event_id.in_(
                    [f"platform-stale-{first}", f"platform-stale-{second}"]
                )
            )
        ).all()
    assert len(events) == 2
