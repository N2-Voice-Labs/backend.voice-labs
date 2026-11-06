"""PostgreSQL-only: /translations under concurrency (hidden agent reuse, slow preflight).

Run this module on its own against a migrated disposable database::

    PLATFORM_TEST_POSTGRES_URL=postgresql://... python -m pytest tests/test_postgres_translations.py
"""

import asyncio
import os
import time
from uuid import uuid4
import types
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

_TEST_DATABASE_URL = os.getenv("PLATFORM_TEST_POSTGRES_URL")
if not _TEST_DATABASE_URL:
    pytest.skip(
        "Set PLATFORM_TEST_POSTGRES_URL to a disposable, migrated PostgreSQL database",
        allow_module_level=True,
    )
if not _TEST_DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg://")):
    raise ValueError("PLATFORM_TEST_POSTGRES_URL must use PostgreSQL")

os.environ["PLATFORM_DATABASE_URL"] = _TEST_DATABASE_URL
os.environ["PLATFORM_AUTO_CREATE_SCHEMA"] = "0"

from src.platform import health  # noqa: E402
from src.platform.db import SessionLocal, engine  # noqa: E402
from src.platform.main import app  # noqa: E402
from src.platform.models import Agent, VoiceSession  # noqa: E402

if engine.dialect.name != "postgresql":
    pytest.skip(
        "Run this module separately so Platform initializes with PostgreSQL",
        allow_module_level=True,
    )

def fresh_body() -> dict:
    """A request whose hidden configuration has never existed (unique glossary term)."""

    term = f"term {uuid4().hex[:10]}"
    return {
        "languages": ["ko", "en"],
        "creator": {"language": "ko"},
        "glossary": [{"source": term, "translations": {"ko": term, "en": term}}],
    }


def hidden_agents() -> int:
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(Agent).where(Agent.system.is_(True))) or 0


def agent_ids(session_ids: list[str]) -> set[str]:
    with SessionLocal() as db:
        return {db.get(VoiceSession, sid).runtime_contract["agent_id"] for sid in session_ids}


def test_concurrent_creates_reuse_one_hidden_configuration(monkeypatch) -> None:
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)
    before = hidden_agents()
    workers = 8
    body = fresh_body()

    def create():
        with TestClient(app) as client:
            return client.post("/translations", json=body)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(lambda _: create(), range(workers)))
    assert [r.status_code for r in responses] == [201] * workers, [r.text for r in responses]
    assert hidden_agents() - before == 1  # one new configuration, however many raced
    assert len(agent_ids([r.json()["session_id"] for r in responses])) == 1


def test_slow_provider_check_does_not_block_the_api(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")
    delay = 1.0

    async def slow_health(*, force: bool = False):
        await asyncio.sleep(delay)  # a Runtime provider check that takes a second
        return None

    class Reply:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "accepted"}

    class FakeRuntime:
        def __init__(self, *_, **__) -> None: ...
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def post(self, *_, **__): return Reply()
        async def delete(self, *_, **__): return Reply()

    monkeypatch.setattr(health, "fetch_runtime_health", slow_health)
    monkeypatch.setattr(
        "src.platform.sessions.httpx",
        types.SimpleNamespace(**{**vars(httpx), "AsyncClient": FakeRuntime}),
    )

    body = fresh_body()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://platform.test") as api:
            started = time.monotonic()
            creates = [asyncio.create_task(api.post("/translations", json=body)) for _ in range(2)]
            await asyncio.sleep(0.2)  # both creates are now waiting on the slow check
            probe_start = time.monotonic()
            probe = await api.get("/health")
            probe_took = time.monotonic() - probe_start
            replies = await asyncio.gather(*creates)
            return probe, probe_took, replies, time.monotonic() - started

    probe, probe_took, replies, total = asyncio.run(scenario())
    assert probe.status_code == 200
    assert probe_took < 0.5, f"API unresponsive during provider check: {probe_took:.2f}s"
    assert [r.status_code for r in replies] == [201, 201], [r.text for r in replies]
    # The two checks overlap instead of queueing behind a lock.
    assert total < delay * 1.8, f"creates serialised: {total:.2f}s"
    assert len(agent_ids([r.json()["session_id"] for r in replies])) == 1
