"""ST-13 platform half: stale-session reaper, admission limits, runtime capacity mapping."""

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from src.platform import auth, limits, reaper
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

SUPER_KEY = "lifecycle-superadmin"
RUNTIME_TOKEN = "lifecycle-runtime-token"
# Other tests leak fresh sessions into the shared DB; a real-time NOW keeps them non-stale.
NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER_KEY)
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", RUNTIME_TOKEN)
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)
    monkeypatch.delenv("PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG", raising=False)
    monkeypatch.delenv("PLATFORM_SESSION_CREATE_RATE", raising=False)
    monkeypatch.delenv("PLATFORM_STALE_SESSION_SECS", raising=False)
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    limits.session_create_bucket.reset()
    yield
    limits.session_create_bucket.reset()


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


def make_key(org_id: str, role: str = "admin") -> dict[str, str]:
    plaintext, prefix, digest = auth.generate_api_key()
    with SessionLocal() as db:
        db.add(
            ApiKey(
                id=str(uuid4()),
                organization_id=org_id,
                name="lifecycle",
                key_prefix=prefix,
                key_hash=digest,
                role=role,
            )
        )
        db.commit()
    return {"Authorization": f"Bearer {plaintext}"}


def make_agent(client, headers) -> str:
    response = client.post("/agents", json={"name": f"a-{uuid4()}"}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def open_session(client, headers, agent_id):
    return client.post("/voice/sessions", json={"agent_id": agent_id}, headers=headers)


def seed_session(
    org_id: str,
    *,
    status: str = "created",
    started_at: datetime = NOW,
    last_event_at: datetime | None = None,
) -> str:
    """A session row (with agent + conversation) inserted directly."""

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
                started_at=started_at,
                last_event_at=last_event_at,
            )
        )
        db.commit()
    return session_id


def load(session_id: str):
    with SessionLocal() as db:
        session = db.get(VoiceSession, session_id)
        conversation = db.get(Conversation, session.conversation_id)
        db.expunge_all()
        return session.status, conversation.status


# ----------------------------------------------------------------------- reaper


def test_reaper_marks_only_stale_sessions_across_orgs() -> None:
    org_a, org_b = make_org(), make_org()
    old = NOW - timedelta(seconds=600)
    fresh = NOW - timedelta(seconds=30)
    stale_a = seed_session(org_a, started_at=old)
    stale_b = seed_session(org_b, status="running", started_at=old)
    fresh_a = seed_session(org_a, started_at=fresh)
    already_ended = seed_session(org_b, status="ended", started_at=old)

    with SessionLocal() as db:
        reaped = reaper.reap_stale_sessions(db, NOW)

    assert set(reaped) == {stale_a, stale_b}
    assert load(stale_a) == ("failed", "failed")
    assert load(stale_b) == ("failed", "failed")
    assert load(fresh_a)[0] == "created"
    assert load(already_ended)[0] == "ended"
    with SessionLocal() as db:
        event = db.query(RuntimeEvent).filter_by(session_id=stale_b).one()
        assert event.event_type == "SessionFailed"
        assert event.payload["reason"] == "stale"
        assert event.organization_id == org_b


def test_reaper_respects_last_event_at_and_is_idempotent() -> None:
    org = make_org()
    old = NOW - timedelta(seconds=3600)
    alive = seed_session(
        org, status="running", started_at=old, last_event_at=NOW - timedelta(seconds=10)
    )
    silent = seed_session(
        org, status="running", started_at=old, last_event_at=NOW - timedelta(seconds=400)
    )
    with SessionLocal() as db:
        assert reaper.reap_stale_sessions(db, NOW) == [silent]
        assert reaper.reap_stale_sessions(db, NOW) == []
    assert load(alive)[0] == "running"
    assert load(silent)[0] == "failed"


def test_reaper_threshold_is_configurable(monkeypatch) -> None:
    session_id = seed_session(make_org(), started_at=NOW - timedelta(seconds=100))
    with SessionLocal() as db:
        assert reaper.reap_stale_sessions(db, NOW) == []
    monkeypatch.setenv("PLATFORM_STALE_SESSION_SECS", "60")
    with SessionLocal() as db:
        assert reaper.reap_stale_sessions(db, NOW) == [session_id]


def test_reaper_interval_env(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_REAPER_INTERVAL_SECS", raising=False)
    assert reaper.reaper_interval() == reaper.DEFAULT_INTERVAL_SECS
    monkeypatch.setenv("PLATFORM_REAPER_INTERVAL_SECS", "15")
    assert reaper.reaper_interval() == 15
    monkeypatch.setenv("PLATFORM_REAPER_INTERVAL_SECS", "0")
    assert reaper.reaper_interval() == 0
    assert reaper.start_reaper() is None


@pytest.mark.asyncio
async def test_reap_once_asks_runtime_to_end_and_ignores_failures(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", RUNTIME_TOKEN)
    session_id = seed_session(make_org(), started_at=datetime.now(timezone.utc) - timedelta(hours=1))
    urls: list[str] = []

    class FakeClient:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, **_):
            urls.append(url)
            raise httpx.ConnectError("runtime down")

    monkeypatch.setattr("src.platform.reaper.httpx.AsyncClient", FakeClient)
    reaped = await reaper.reap_once()
    assert session_id in reaped
    assert f"http://runtime.test/sessions/{session_id}/end" in urls
    assert load(session_id)[0] == "failed"


def test_runtime_events_refresh_last_event_at(client) -> None:
    org = make_org()
    session_id = seed_session(org, started_at=NOW - timedelta(hours=1))
    with SessionLocal() as db:
        session = db.get(VoiceSession, session_id)
        conversation = db.get(Conversation, session.conversation_id)
        body = {
            "event_id": f"evt-{uuid4()}",
            "event_type": "SpeechStarted",
            "session_id": session_id,
            "conversation_id": conversation.id,
            "agent_id": conversation.agent_id,
            "payload": {},
        }
    response = client.post(
        "/runtime/events", json=body, headers={"X-Runtime-Token": RUNTIME_TOKEN}
    )
    assert response.status_code == 202, response.text
    with SessionLocal() as db:
        last = db.get(VoiceSession, session_id).last_event_at
    assert last is not None
    assert datetime.now(timezone.utc) - last.replace(tzinfo=last.tzinfo or timezone.utc) < timedelta(minutes=1)
    with SessionLocal() as db:
        assert reaper.reap_stale_sessions(db) == []


# ----------------------------------------------------------------------- limits


def test_active_session_cap_is_per_org(client, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG", "2")
    org_a, org_b = make_org(), make_org()
    key_a, key_b = make_key(org_a), make_key(org_b)
    agent_a, agent_b = make_agent(client, key_a), make_agent(client, key_b)

    assert open_session(client, key_a, agent_a).status_code == 201
    second = open_session(client, key_a, agent_a)
    assert second.status_code == 201
    third = open_session(client, key_a, agent_a)
    assert third.status_code == 429
    assert third.json()["detail"]["code"] == "too_many_active_sessions"
    assert third.json()["detail"]["message"]
    with SessionLocal() as db:
        assert limits.count_active_sessions(db, org_a) == 2

    # The other org is unaffected.
    assert open_session(client, key_b, agent_b).status_code == 201

    # Ending a session frees a slot.
    end = client.post(f"/voice/sessions/{second.json()['session_id']}/end", headers=key_a)
    assert end.status_code == 200
    assert open_session(client, key_a, agent_a).status_code == 201


def test_failed_sessions_do_not_count_toward_cap(client, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG", "1")
    org = make_org()
    key = make_key(org)
    agent = make_agent(client, key)
    seed_session(org, status="failed")
    seed_session(org, status="ended")
    assert open_session(client, key, agent).status_code == 201
    assert open_session(client, key, agent).status_code == 429


def test_create_rate_limit_returns_retry_after(client, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_SESSION_CREATE_RATE", "2/min")
    org = make_org()
    key, other_key = make_key(org), make_key(org)
    agent = make_agent(client, key)

    assert open_session(client, key, agent).status_code == 201
    assert open_session(client, key, agent).status_code == 201
    limited = open_session(client, key, agent)
    assert limited.status_code == 429
    assert limited.json()["detail"]["code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1
    # Buckets are per key.
    assert open_session(client, other_key, agent).status_code == 201


def test_token_bucket_refills_over_time() -> None:
    bucket = limits.TokenBucket()
    assert bucket.take("k", 60, now=0.0) == 0.0
    for _ in range(59):
        assert bucket.take("k", 60, now=0.0) == 0.0
    wait = bucket.take("k", 60, now=0.0)
    assert 0 < wait <= 1.0
    assert bucket.take("k", 60, now=1.1) == 0.0


# ------------------------------------------------------- runtime capacity mapping


class _Resp:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self._body = body
        self.request = httpx.Request("POST", "http://runtime.test/sessions")

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


def _patch_runtime(monkeypatch, status: int, body: dict) -> list[str]:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", RUNTIME_TOKEN)
    calls: list[str] = []

    class FakeClient:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url, **_):
            raise httpx.ConnectError("no health in tests")

        async def post(self, url, **_):
            calls.append(url)
            return _Resp(status, body)

    monkeypatch.setattr("src.platform.main.httpx.AsyncClient", FakeClient)
    return calls


@pytest.mark.parametrize(
    "body",
    [
        {"code": "runtime_at_capacity"},
        {"detail": {"code": "runtime_at_capacity", "message": "full"}},
    ],
)
def test_runtime_at_capacity_maps_to_503_without_active_session(
    client, monkeypatch, body
) -> None:
    org = make_org()
    key = make_key(org)
    agent = make_agent(client, key)
    calls = _patch_runtime(monkeypatch, 503, body)

    response = open_session(client, key, agent)

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "runtime_at_capacity",
        "message": "All voice workers are busy, please try again shortly.",
    }
    assert len(calls) == 1  # no retry against a full runtime
    with SessionLocal() as db:
        assert limits.count_active_sessions(db, org) == 0
        statuses = {
            s.status for s in db.query(VoiceSession).filter_by(organization_id=org)
        }
    assert statuses == {"failed"}


def test_other_runtime_503_stays_runtime_unavailable(client, monkeypatch) -> None:
    key = make_key(make_org())
    agent = make_agent(client, key)
    _patch_runtime(monkeypatch, 503, {"detail": "boom"})
    response = open_session(client, key, agent)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "runtime_unavailable"


# -------------------------------------------------------------------- migration


def run_alembic(db_url: str, *args: str) -> None:
    platform_dir = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PLATFORM_DATABASE_URL": db_url, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PLATFORM_API_KEYS", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=platform_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_migration_0013_upgrade_and_downgrade() -> None:
    cache = Path.home() / ".cache" / "voice-labs-test"
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / f"lifecycle-migration-{uuid4().hex}.db"
    url = f"sqlite:///{db_path}"
    try:
        run_alembic(url, "upgrade", "head")
        db_engine = create_engine(url)
        insp = inspect(db_engine)
        assert "last_event_at" in {c["name"] for c in insp.get_columns("voice_sessions")}
        assert "ix_voice_sessions_last_event_at" in {
            i["name"] for i in insp.get_indexes("voice_sessions")
        }
        db_engine.dispose()

        run_alembic(url, "downgrade", "0012_problem_reports")
        db_engine = create_engine(url)
        insp = inspect(db_engine)
        assert "last_event_at" not in {
            c["name"] for c in insp.get_columns("voice_sessions")
        }
        db_engine.dispose()

        run_alembic(url, "upgrade", "head")
    finally:
        db_path.unlink(missing_ok=True)
