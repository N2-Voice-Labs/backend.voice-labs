"""PostgreSQL-only tests for transaction races in Platform event ingestion."""

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session as OrmSession

_TEST_DATABASE_URL = os.getenv("PLATFORM_TEST_POSTGRES_URL")
if not _TEST_DATABASE_URL:
    pytest.skip(
        "Set PLATFORM_TEST_POSTGRES_URL to a disposable PostgreSQL database",
        allow_module_level=True,
    )
if not _TEST_DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg://")):
    raise ValueError("PLATFORM_TEST_POSTGRES_URL must use PostgreSQL")

os.environ["PLATFORM_DATABASE_URL"] = _TEST_DATABASE_URL
os.environ["PLATFORM_AUTO_CREATE_SCHEMA"] = "0"

from src.platform.db import engine  # noqa: E402
from src.platform.main import app  # noqa: E402

if engine.dialect.name != "postgresql":
    pytest.skip(
        "Run this module separately so Platform initializes with PostgreSQL",
        allow_module_level=True,
    )


def test_concurrent_duplicate_transcripts_are_persisted_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_token = "postgres-concurrency-test-token"
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", runtime_token)
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)

    with TestClient(app) as client:
        agent_response = client.post("/agents", json={"name": "Concurrent event test"})
        assert agent_response.status_code == 201
        agent = agent_response.json()
        session_response = client.post(
            "/voice/sessions", json={"agent_id": agent["id"]}
        )
        assert session_response.status_code == 201
        session = session_response.json()

    transcript = "concurrent transcript once"
    event_payload = {
        "event_id": str(uuid4()),
        "event_type": "TranscriptFinal",
        "session_id": session["session_id"],
        "conversation_id": session["conversation_id"],
        "agent_id": agent["id"],
        "payload": {"role": "user", "content": transcript},
    }
    commit_barrier = threading.Barrier(2)

    def synchronize_commits(_: OrmSession) -> None:
        commit_barrier.wait(timeout=20)

    def post_event():
        with TestClient(app) as client:
            return client.post(
                "/runtime/events",
                json=event_payload,
                headers={"X-Runtime-Token": runtime_token},
            )

    event.listen(OrmSession, "before_commit", synchronize_commits)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(post_event) for _ in range(2)]
            responses = [future.result(timeout=30) for future in futures]
    finally:
        event.remove(OrmSession, "before_commit", synchronize_commits)

    assert all(response.status_code == 202 for response in responses)
    assert sorted(response.json()["status"] for response in responses) == [
        "accepted",
        "duplicate",
    ]
    with TestClient(app) as client:
        messages_response = client.get(
            f"/conversations/{session['conversation_id']}/messages"
        )
    assert messages_response.status_code == 200
    messages = messages_response.json()
    assert [message["content"] for message in messages] == [transcript]
