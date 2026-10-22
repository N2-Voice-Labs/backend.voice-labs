"""RM-003: the ``timeline`` section of the analytics summary."""

import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient

from src.platform.analytics import timeline_sample
from src.platform.main import app

TOKEN = "timeline-token"
HEADERS = {"X-Runtime-Token": TOKEN}


@pytest.fixture(autouse=True)
def runtime_token(monkeypatch):
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", TOKEN)
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)


def send(client, session, agent_id, payload, event_type="LatencyMeasured"):
    return client.post(
        "/runtime/events",
        json={
            "event_id": str(uuid4()),
            "event_type": event_type,
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent_id,
            "payload": payload,
        },
        headers=HEADERS,
    )


def timeline_event(engine, first_audio, **marks):
    return {
        "metric": "utterance_timeline",
        "value_ms": first_audio,
        "engine": engine,
        "marks_ms": {"vad": 200.0, **marks, "first_audio": first_audio},
    }


def test_timeline_section_aggregates_by_engine_and_mark() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": f"tl-{uuid4()}"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        for first_audio, stable in ((1000, 1500), (2000, 2500), (3000, 3500), (4000, 4500)):
            event = timeline_event(
                "cascaded", first_audio, first_transcript=first_audio / 4, stable=stable
            )
            assert send(client, session, agent["id"], event).status_code in (200, 201, 202)
        realtime = timeline_event("realtime", 900.0, stable=1400.0)
        send(client, session, agent["id"], realtime)
        # Malformed or unrelated events are ignored by the timeline section.
        send(client, session, agent["id"], {"metric": "utterance_timeline", "value_ms": "bad", "engine": "x"})
        send(client, session, agent["id"], {"metric": "utterance_timeline", "engine": "x"})
        send(client, session, agent["id"], {"metric": "stt_ttfb", "value_ms": 120})

        summary = client.get("/analytics/summary", params={"agent_id": agent["id"]}).json()

    timeline = summary["timeline"]
    assert set(timeline) == {"cascaded", "realtime"}
    cascaded = timeline["cascaded"]
    assert (cascaded["count"], cascaded["p50"], cascaded["p95"]) == (4, 2000.0, 4000.0)
    assert cascaded["marks"]["first_audio"] == {"count": 4, "p50": 2000.0, "p95": 4000.0}
    assert cascaded["marks"]["stable"] == {"count": 4, "p50": 2500.0, "p95": 4500.0}
    assert cascaded["marks"]["vad"] == {"count": 4, "p50": 200.0, "p95": 200.0}
    assert cascaded["marks"]["first_transcript"]["p50"] == 500.0
    assert "translation_start" not in cascaded["marks"]  # marks nobody reported are absent
    assert timeline["realtime"] == {
        "count": 1,
        "p50": 900.0,
        "p95": 900.0,
        "marks": {
            "first_audio": {"count": 1, "p50": 900.0, "p95": 900.0},
            "stable": {"count": 1, "p50": 1400.0, "p95": 1400.0},
            "vad": {"count": 1, "p50": 200.0, "p95": 200.0},
        },
    }
    # Existing response keys are untouched, and the other latency metrics still aggregate.
    assert {"conversations", "languages", "latency"} <= set(summary)
    assert summary["latency"]["stt_ttfb"] == {"count": 1, "p50": 120.0, "p95": 120.0}
    assert summary["latency"]["utterance_timeline"]["count"] == 5


def test_timeline_section_is_empty_without_timeline_events() -> None:
    with TestClient(app) as client:
        summary = client.get("/analytics/summary", params={"agent_id": str(uuid4())}).json()
    assert summary["timeline"] == {}


def test_timeline_sample_parsing() -> None:
    assert timeline_sample(timeline_event("agent_translation", 1200.0)) == (
        "agent_translation", 1200.0, {"vad": 200.0, "first_audio": 1200.0},
    )  # fmt: skip
    # Missing engine => "unknown"; bad marks are dropped, not the whole sample.
    sample = timeline_sample(
        {"metric": "utterance_timeline", "value_ms": 5, "marks_ms": {"vad": "x", "stable": -1, "first_audio": 5}}
    )
    assert sample == ("unknown", 5.0, {"first_audio": 5.0})
    assert timeline_sample({"metric": "utterance_timeline", "value_ms": True}) is None
    assert timeline_sample({"metric": "utterance_timeline", "value_ms": float("nan")}) is None
    assert timeline_sample({"metric": "stt_ttfb", "value_ms": 5}) is None
    assert timeline_sample("nope") is None
