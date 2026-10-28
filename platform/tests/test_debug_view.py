"""ST-02b: per-call debug view and problem reports."""

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from src.platform import auth, debug
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import (
    Agent,
    ApiKey,
    AuditLog,
    Conversation,
    Organization,
    ProblemReport,
    RuntimeEvent,
    VoiceSession,
)
from src.platform.tenancy import bypass

SUPER = "st02b-superadmin-key"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)

ISOLATION_CASES = [
    ("GET", "/conversations/{conversation_id}/debug"),
    ("GET", "/conversations/{conversation_id}/reports"),
    ("POST", "/conversations/{conversation_id}/reports"),
]


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


def seed(org_id: str, token: str | None = None, contract: dict | None = None) -> dict:
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
                runtime_contract=contract,
            )
        )
        db.commit()
    return {"conversation_id": conv_id, "session_id": sess_id, "agent_id": agent_id}


def add_events(ids: dict, org_id: str, events: list[tuple[str, dict]]) -> None:
    with SessionLocal() as db, bypass(db):
        for index, (event_type, payload) in enumerate(events):
            db.add(
                RuntimeEvent(
                    organization_id=org_id,
                    event_id=str(uuid4()),
                    event_type=event_type,
                    conversation_id=ids["conversation_id"],
                    session_id=ids["session_id"],
                    payload=payload,
                    received_at=BASE + timedelta(milliseconds=index),
                )
            )
        db.commit()


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER)
    with TestClient(app) as client:
        org_a, org_b = make_org(), make_org()
        guest = "vls_" + uuid4().hex
        contract = {
            "system_prompt": "be nice",
            "providers": {"llm": {"api_key": "sk-live-123", "model": "x"}},
            "provider_settings": {"stt": {"secret_reference": "ref", "language": "en"}},
        }
        yield {
            "client": client,
            "org_a": org_a,
            "org_b": org_b,
            "a": seed(org_a, token=guest, contract=contract),
            "b": seed(org_b),
            "key_a": make_key(org_a, "member"),
            "key_b": make_key(org_b, "member"),
            "guest": guest,
        }


def debug_of(world, key="key_a", side="a"):
    r = world["client"].get(
        f"/conversations/{world[side]['conversation_id']}/debug", headers=hdr(world[key])
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_turns_are_grouped_by_turn_id(world):
    add_events(
        world["a"],
        world["org_a"],
        [
            ("RuntimeStarted", {}),
            ("SpeechStarted", {"turn_id": 1}),
            ("TranscriptFinal", {"turn_id": 1, "role": "user", "content": "hi", "confidence": 0.9}),
            ("LatencyMeasured", {"turn_id": 1, "metric": "llm_ttfb", "value_ms": 300.0}),
            ("LatencyMeasured", {"turn_id": 1, "metric": "turn_to_first_audio", "value_ms": 900.0}),
            (
                "LatencyMeasured",
                {
                    "turn_id": 1,
                    "metric": "utterance_timeline",
                    "value_ms": 1000.0,
                    "engine": "agent",
                    "marks_ms": {"vad": 50.0},
                },
            ),
            (
                "TurnDebug",
                {
                    "turn_id": 1,
                    "stt": {"text": "hi", "language": "en", "confidence": 0.9},
                    "llm": {"model": "m", "input_tokens": 10, "output_tokens": 5, "first_token_ms": 250},
                    "tts": {"provider": "cartesia", "voice_id": "v", "text": "hello", "first_audio_ms": 400},
                    "input": {"rms_dbfs": -30, "peak_dbfs": -10, "voiced_ms": 800},
                },
            ),
            ("TranscriptFinal", {"turn_id": 2, "role": "user", "content": "again"}),
            ("LatencyMeasured", {"turn_id": 2, "metric": "turn_to_first_audio", "value_ms": 1500.0}),
            ("SessionEnded", {"outcome": "resolved", "reason": "hangup"}),
        ],
    )
    body = debug_of(world)
    assert [t["turn_id"] for t in body["turns"]] == [1, 2]
    one, two = body["turns"]
    assert one["stt"]["text"] == "hi"
    # token *counts* must survive redaction
    assert one["llm"]["input_tokens"] == 10 and one["llm"]["output_tokens"] == 5
    assert one["tts"]["provider"] == "cartesia"
    assert one["input"]["voiced_ms"] == 800
    assert one["latency"]["llm_ttfb"] == 300.0
    assert one["latency"]["turn_to_first_audio"] == 900.0
    assert one["latency"]["utterance_timeline"]["marks_ms"] == {"vad": 50.0}
    assert [t["content"] for t in one["transcripts"]] == ["hi"]
    assert [e["event_type"] for e in one["events"]] == ["SpeechStarted"]
    assert two["stt"] is None and two["transcripts"][0]["content"] == "again"
    assert {e["event_type"] for e in body["unassigned_events"]} == {"RuntimeStarted", "SessionEnded"}
    summary = body["summary"]
    assert summary["turn_count"] == 2 and summary["error_count"] == 0
    assert summary["failed_stage"] is None
    assert summary["turn_to_first_audio_ms"] == {"count": 2, "p50": 900.0, "p95": 1500.0}
    assert body["session"]["runtime_status"] == "ended"
    assert body["session"]["end_reason"] == "hangup"
    assert body["conversation"]["id"] == world["a"]["conversation_id"]
    assert body["truncated"] is False


def test_failed_stage_is_first_terminal_failure(world):
    add_events(
        world["a"],
        world["org_a"],
        [
            # retryable errors are not failures
            ("ProviderError", {"turn_id": 1, "kind": "stt", "category": "timeout", "stage": "x",
                               "retryable": True, "attempt": 1, "message": "m"}),
            ("TranslationEmpty", {"turn_id": 1, "stage": "translate"}),
            ("ProviderError", {"turn_id": 2, "kind": "tts", "category": "auth", "stage": "synthesis",
                               "retryable": False, "attempt": 1, "message": "m"}),
            ("SessionFailed", {"reason": "provider_failed"}),
        ],
    )
    body = debug_of(world)
    assert body["summary"]["failed_stage"] == "translation"
    assert body["summary"]["error_count"] == 2
    assert [e["event_type"] for e in body["errors"]] == ["ProviderError", "ProviderError"]
    turn_one = body["turns"][0]
    assert [e["event_type"] for e in turn_one["errors"]] == ["ProviderError", "TranslationEmpty"]
    assert body["session"]["runtime_status"] == "failed"
    assert body["session"]["end_reason"] == "provider_failed"


@pytest.mark.parametrize(
    "events, stage",
    [
        ([("ProviderError", {"kind": "stt", "retryable": False})], "stt"),
        ([("ProviderError", {"kind": "llm", "retryable": False})], "llm"),
        ([("AnswerEmpty", {"turn_id": 1})], "llm"),
        ([("ProviderError", {"kind": "tts", "retryable": False})], "tts"),
        ([("ProviderError", {"kind": "realtime", "retryable": False})], "translation"),
        ([("SessionFailed", {"reason": "livekit_disconnected"})], "transport"),
        ([("ProviderError", {"kind": "llm", "retryable": True})], None),
    ],
)
def test_failed_stage_detection(world, events, stage):
    add_events(world["a"], world["org_a"], events)
    assert debug_of(world)["summary"]["failed_stage"] == stage


def test_config_is_redacted(world):
    add_events(
        world["a"],
        world["org_a"],
        [("ToolCalled", {"turn_id": 1, "tool_name": "t", "arguments": {"api_key": "sk-leak", "q": 1}})],
    )
    body = debug_of(world)
    config = body["config"]
    assert config["system_prompt"] == "be nice"
    assert config["providers"]["llm"] == {"api_key": "[redacted]", "model": "x"}
    assert config["provider_settings"]["stt"]["secret_reference"] == "[redacted]"
    assert config["provider_settings"]["stt"]["language"] == "en"
    assert body["turns"][0]["events"][0]["payload"]["arguments"] == {"api_key": "[redacted]", "q": 1}
    assert "sk-live-123" not in str(body) and "sk-leak" not in str(body)


def test_config_falls_back_to_agent_without_contract(world):
    body = debug_of(world, "key_b", "b")
    assert body["config_source"] == "agent"
    assert body["config"]["agent_id"] == world["b"]["agent_id"]


def test_cross_org_is_404(world):
    c = world["client"]
    cid = world["a"]["conversation_id"]
    for method, path in (
        ("get", f"/conversations/{cid}/debug"),
        ("get", f"/conversations/{cid}/reports"),
        ("post", f"/conversations/{cid}/reports"),
    ):
        kwargs = {"json": {"note": "x"}} if method == "post" else {}
        r = getattr(c, method)(path, headers=hdr(world["key_b"]), **kwargs)
        assert r.status_code == 404, (method, path)
        assert r.json()["detail"] == "conversation_not_found"
    with SessionLocal() as db, bypass(db):
        assert db.query(ProblemReport).count() == 0


def test_unauthenticated_and_unknown(world):
    c = world["client"]
    cid = world["a"]["conversation_id"]
    assert c.get(f"/conversations/{cid}/debug").status_code == 401
    assert c.get(f"/conversations/{uuid4()}/debug", headers=hdr(world["key_a"])).status_code == 404


def test_session_token_can_report_but_not_read_debug(world):
    c = world["client"]
    cid = world["a"]["conversation_id"]
    guest = hdr(world["guest"])
    assert c.get(f"/conversations/{cid}/debug", headers=guest).status_code == 403
    assert c.get(f"/conversations/{cid}/reports", headers=guest).status_code == 403
    r = c.post(f"/conversations/{cid}/reports", json={"note": "  audio cut out  "}, headers=guest)
    assert r.status_code == 201, r.text
    assert set(r.json()) == {"id", "created_at"}
    # a token for another conversation is refused
    other = c.post(
        f"/conversations/{world['b']['conversation_id']}/reports",
        json={"note": "x"},
        headers=guest,
    )
    assert other.status_code == 403
    listed = c.get(f"/conversations/{cid}/reports", headers=hdr(world["key_a"])).json()
    assert len(listed) == 1
    assert listed[0]["note"] == "audio cut out"
    assert listed[0]["session_id"] == world["a"]["session_id"]
    assert listed[0]["reporter"] == f"session:{world['a']['session_id']}"


def test_member_report_audit_and_validation(world):
    c = world["client"]
    cid = world["a"]["conversation_id"]
    url = f"/conversations/{cid}/reports"
    assert c.post(url, json={"note": ""}, headers=hdr(world["key_a"])).status_code == 422
    assert c.post(url, json={"note": "   "}, headers=hdr(world["key_a"])).status_code == 422
    assert c.post(url, json={"note": "x" * 2001}, headers=hdr(world["key_a"])).status_code == 422
    assert c.post(url, json={"note": "x" * 2000}, headers=hdr(world["key_a"])).status_code == 201
    r = c.post(url, json={"note": "robotic voice"}, headers=hdr(world["key_a"]))
    assert r.status_code == 201
    with SessionLocal() as db, bypass(db):
        entry = (
            db.query(AuditLog)
            .filter(AuditLog.action == "problem.reported", AuditLog.organization_id == world["org_a"])
            .all()
        )
        assert len(entry) == 2
        assert {e.organization_id for e in entry} == {world["org_a"]}
        assert any(e.resource_id == r.json()["id"] for e in entry)
        report = db.get(ProblemReport, r.json()["id"])
        assert report.organization_id == world["org_a"]
        assert report.reporter.startswith("api_key:")
    # org B never sees them
    assert c.get(f"/conversations/{world['b']['conversation_id']}/reports",
                 headers=hdr(world["key_b"])).json() == []


def test_large_event_count_is_truncated(world):
    ids, org = world["a"], world["org_a"]
    total = debug.MAX_EVENTS + 120
    with SessionLocal() as db, bypass(db):
        db.add_all(
            [
                RuntimeEvent(
                    organization_id=org,
                    event_id=str(uuid4()),
                    event_type="SpeechStarted",
                    conversation_id=ids["conversation_id"],
                    session_id=ids["session_id"],
                    payload={"turn_id": i},
                    received_at=BASE + timedelta(milliseconds=i),
                )
                for i in range(total)
            ]
        )
        db.commit()
    body = debug_of(world)
    assert body["truncated"] is True
    assert body["event_count"] == debug.MAX_EVENTS
    assert body["summary"]["turn_count"] == debug.MAX_EVENTS
    # earliest events are kept, in order
    assert body["turns"][0]["turn_id"] == 0


def test_exact_cap_is_not_truncated(world, monkeypatch):
    monkeypatch.setattr(debug, "MAX_EVENTS", 30)
    monkeypatch.setattr(debug, "PAGE_SIZE", 7)
    add_events(world["a"], world["org_a"], [("SpeechStarted", {"turn_id": i}) for i in range(30)])
    body = debug_of(world)
    assert body["event_count"] == 30 and body["truncated"] is False
    add_events(world["a"], world["org_a"], [("SpeechStarted", {"turn_id": 99})])
    body = debug_of(world)
    assert body["event_count"] == 30 and body["truncated"] is True


def test_migration_0012_up_and_down(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    platform_dir = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PLATFORM_DATABASE_URL": url, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PLATFORM_API_KEYS", None)

    def alembic(*args: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *args],
            cwd=platform_dir, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    alembic("upgrade", "head")
    engine = create_engine(url)
    insp = inspect(engine)
    assert "problem_reports" in insp.get_table_names()
    columns = {c["name"]: c for c in insp.get_columns("problem_reports")}
    assert set(columns) == {
        "id", "organization_id", "conversation_id", "session_id", "note", "reporter", "created_at"
    }
    assert columns["organization_id"]["nullable"] is False and columns["session_id"]["nullable"] is True
    indexed = {tuple(i["column_names"]) for i in insp.get_indexes("problem_reports")}
    assert ("conversation_id",) in indexed and ("organization_id",) in indexed
    alembic("downgrade", "0011_voice_tombstone")
    assert "problem_reports" not in inspect(engine).get_table_names()
    alembic("upgrade", "head")
    assert "problem_reports" in inspect(engine).get_table_names()
    engine.dispose()
