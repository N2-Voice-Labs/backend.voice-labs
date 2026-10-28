"""Per-call debug view and problem reports (ST-02b).

``GET /conversations/{id}/debug`` folds the Runtime events of one conversation into a
per-turn timeline (what STT heard, what the LLM answered, what TTS spoke, latency,
errors) plus a summary that names the first failing stage. It is admin-style: member
role or above, never a session token. ``POST /conversations/{id}/reports`` is the one
route a guest's session token may also use (the "Report a problem" button).
"""

import math
import re
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import audit
from .context import AuthContext, get_ctx, require
from .models import Agent, Conversation, ProblemReport, RuntimeEvent, VoiceSession
from .tenancy import get_owned, get_scoped_db, scoped
from .validation import SECRET_KEY_PARTS

router = APIRouter(tags=["debug"])

MAX_EVENTS = 5000
PAGE_SIZE = 500
MAX_NOTE_LENGTH = 2000
MAX_REPORTS_PER_CONVERSATION = 100
REDACTED = "[redacted]"

# Event types folded into a turn's own sections rather than its generic ``events`` list.
_ERROR_EVENT_TYPES = frozenset({"ProviderError", "AnswerEmpty", "TranslationEmpty"})
_TERMINAL_EVENT_TYPES = frozenset({"SessionEnded", "SessionFailed", "RuntimeFailed"})
# ``input_tokens`` / ``max_tokens`` are counts, not credentials.
_COUNT_KEY = re.compile(r"(^|_)tokens$")
_STAGE_BY_PROVIDER_KIND = {
    "stt": "stt",
    "llm": "llm",
    "tts": "tts",
    "translation": "translation",
    "realtime": "translation",
}


def _is_secret_key(key: Any) -> bool:
    text = str(key).casefold()
    if _COUNT_KEY.search(text):
        return False
    return any(part in text for part in SECRET_KEY_PARTS)


def redact(value: Any) -> Any:
    """Copy of ``value`` with the values of secret-looking dict keys masked."""

    if isinstance(value, dict):
        return {
            key: REDACTED if _is_secret_key(key) else redact(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def load_events(
    db: Session, ctx: AuthContext, conversation_id: str
) -> tuple[list[RuntimeEvent], bool]:
    """Events of one conversation in time order, read page by page, capped."""

    events: list[RuntimeEvent] = []
    while len(events) <= MAX_EVENTS:
        page = db.scalars(
            scoped(db, ctx, RuntimeEvent)
            .where(RuntimeEvent.conversation_id == conversation_id)
            .order_by(RuntimeEvent.received_at, RuntimeEvent.id)
            .limit(min(PAGE_SIZE, MAX_EVENTS + 1 - len(events)))
            .offset(len(events))
        ).all()
        events.extend(page)
        if len(page) < PAGE_SIZE:
            break
    truncated = len(events) > MAX_EVENTS
    return events[:MAX_EVENTS], truncated


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _event_dict(event: RuntimeEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "session_id": event.session_id,
        "received_at": _iso(event.received_at),
        "payload": redact(event.payload or {}),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _add_latency(latency: dict[str, Any], payload: dict[str, Any]) -> None:
    metric = payload.get("metric")
    if metric == "utterance_timeline":
        latency["utterance_timeline"] = {
            "value_ms": payload.get("value_ms"),
            "engine": payload.get("engine"),
            "marks_ms": payload.get("marks_ms") or {},
        }
    elif isinstance(metric, str):
        latency[metric] = payload.get("value_ms")
    else:  # tolerate flat payloads: {"llm_ttfb": 120, "utterance_timeline": {...}}
        for key, value in payload.items():
            if key == "turn_id":
                continue
            latency[key] = value


def _failure_stage(event: RuntimeEvent) -> str | None:
    """Stage a failure event points at, or None when the event is not a failure."""

    payload = event.payload or {}
    if event.event_type == "ProviderError":
        if payload.get("retryable"):
            return None  # the Runtime is retrying; not (yet) a failure
        return _STAGE_BY_PROVIDER_KIND.get(str(payload.get("kind")), "transport")
    if event.event_type == "AnswerEmpty":
        return "llm"
    if event.event_type == "TranslationEmpty":
        return "translation"
    if event.event_type in {"SessionFailed", "RuntimeFailed"}:
        return "transport"
    return None


def build_debug(
    conversation: Conversation,
    session: VoiceSession | None,
    events: list[RuntimeEvent],
    truncated: bool,
    agent: Agent | None,
) -> dict[str, Any]:
    turns: dict[Any, dict[str, Any]] = {}
    unassigned: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    first_audio: list[float] = []
    failed_stage: str | None = None
    end_event: RuntimeEvent | None = None
    runtime_status: str | None = None

    for event in events:
        payload = event.payload or {}
        if event.event_type == "RuntimeStarted" and runtime_status is None:
            runtime_status = "running"
        if event.event_type in _TERMINAL_EVENT_TYPES:
            end_event = event
            runtime_status = "failed" if event.event_type != "SessionEnded" else "ended"
        stage = _failure_stage(event)
        if stage is not None and failed_stage is None:
            failed_stage = stage
        if event.event_type == "ProviderError":
            errors.append(_event_dict(event))
        if (
            event.event_type == "LatencyMeasured"
            and payload.get("metric") == "turn_to_first_audio"
            and (value := _number(payload.get("value_ms"))) is not None
        ):
            first_audio.append(value)

        turn_id = payload.get("turn_id")
        if turn_id is None or isinstance(turn_id, (dict, list)):
            unassigned.append(_event_dict(event))
            continue
        turn = turns.setdefault(
            turn_id,
            {
                "turn_id": turn_id,
                "started_at": _iso(event.received_at),
                "stt": None,
                "llm": None,
                "tts": None,
                "input": None,
                "latency": {},
                "transcripts": [],
                "errors": [],
                "events": [],
            },
        )
        if event.event_type == "TurnDebug":
            for section in ("stt", "llm", "tts", "input"):
                turn[section] = redact(payload.get(section))
        elif event.event_type == "LatencyMeasured":
            _add_latency(turn["latency"], redact(payload))
        elif event.event_type == "TranscriptFinal":
            turn["transcripts"].append(
                {
                    "role": payload.get("role"),
                    "content": payload.get("content"),
                    "primary_language": payload.get("primary_language"),
                    "detected_languages": payload.get("detected_languages") or [],
                    "confidence": payload.get("confidence"),
                    "received_at": _iso(event.received_at),
                }
            )
        elif event.event_type in _ERROR_EVENT_TYPES:
            turn["errors"].append(_event_dict(event))
        else:
            turn["events"].append(_event_dict(event))

    end_payload = (end_event.payload or {}) if end_event else {}
    if conversation.outcome is None and end_payload.get("outcome"):
        outcome = end_payload.get("outcome")
    else:
        outcome = conversation.outcome

    if session is not None and session.runtime_contract:
        config: dict[str, Any] = redact(session.runtime_contract)
        config_source = "runtime_contract"
    elif agent is not None:
        config = redact(
            {
                "agent_id": agent.id,
                "name": agent.name,
                "mode": agent.mode,
                "system_prompt": agent.system_prompt,
                "enabled_languages": agent.enabled_languages,
                "voice_id": agent.voice_id,
                "interpreter": agent.interpreter,
                "translation": agent.translation,
            }
        )
        config_source = "agent"
    else:
        config, config_source = {}, None

    return {
        "conversation": {
            "id": conversation.id,
            "agent_id": conversation.agent_id,
            "status": conversation.status,
            "outcome": outcome,
            "started_at": _iso(conversation.started_at),
            "ended_at": _iso(conversation.ended_at),
        },
        "session": (
            {
                "id": session.id,
                "room_name": session.room_name,
                "status": session.status,
                "runtime_status": runtime_status or session.status,
                "end_reason": end_payload.get("reason"),
            }
            if session is not None
            else None
        ),
        "config": config,
        "config_source": config_source,
        "turns": list(turns.values()),
        "unassigned_events": unassigned,
        "errors": errors,
        "summary": {
            "turn_count": len(turns),
            "error_count": len(errors),
            "failed_stage": failed_stage,
            "turn_to_first_audio_ms": {
                "count": len(first_audio),
                "p50": _percentile(first_audio, 50),
                "p95": _percentile(first_audio, 95),
            },
        },
        "event_count": len(events),
        "truncated": truncated,
    }


@router.get("/conversations/{conversation_id}/debug")
def get_conversation_debug(
    conversation_id: UUID,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    conversation = get_owned(
        db, ctx, Conversation, str(conversation_id), not_found_detail="conversation_not_found"
    )
    session = db.scalars(
        scoped(db, ctx, VoiceSession)
        .where(VoiceSession.conversation_id == conversation.id)
        .order_by(VoiceSession.started_at.desc())
        .limit(1)
    ).first()
    agent = db.scalars(
        scoped(db, ctx, Agent).where(Agent.id == conversation.agent_id)
    ).first()
    events, truncated = load_events(db, ctx, conversation.id)
    return build_debug(conversation, session, events, truncated, agent)


# ------------------------------------------------------------------ problem reports


class ReportIn(BaseModel):
    note: str = Field(min_length=1, max_length=MAX_NOTE_LENGTH)

    @field_validator("note")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("note must not be blank")
        return value


def report_author(request: Request) -> AuthContext:
    """Members and above, or the conversation's own session token (auth.py scopes it)."""

    ctx = get_ctx(request)
    if ctx.principal == "session":
        return ctx
    return require("member")(request)


def _reporter(ctx: AuthContext) -> str:
    identity = ctx.session_id if ctx.principal == "session" else ctx.key_id
    return f"{ctx.principal}:{identity}" if identity else ctx.principal


def _report_dict(report: ProblemReport) -> dict[str, Any]:
    return {
        "id": report.id,
        "conversation_id": report.conversation_id,
        "session_id": report.session_id,
        "note": report.note,
        "reporter": report.reporter,
        "created_at": _iso(report.created_at),
    }


@router.post("/conversations/{conversation_id}/reports", status_code=201)
def create_report(
    conversation_id: UUID,
    payload: ReportIn,
    ctx: AuthContext = Depends(report_author),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    conversation = get_owned(
        db, ctx, Conversation, str(conversation_id), not_found_detail="conversation_not_found"
    )
    existing = db.scalar(
        select(func.count())
        .select_from(ProblemReport)
        .where(
            ProblemReport.conversation_id == conversation.id,
            ProblemReport.organization_id == conversation.organization_id,
        )
    )
    if (existing or 0) >= MAX_REPORTS_PER_CONVERSATION:
        raise HTTPException(status_code=429, detail="report_limit_reached")
    session_id = ctx.session_id
    if session_id is None:
        session_id = db.scalars(
            scoped(db, ctx, VoiceSession)
            .where(VoiceSession.conversation_id == conversation.id)
            .order_by(VoiceSession.started_at.desc())
            .limit(1)
        ).first()
        session_id = session_id.id if session_id is not None else None
    report = ProblemReport(
        id=str(uuid4()),
        organization_id=conversation.organization_id,
        conversation_id=conversation.id,
        session_id=session_id,
        note=payload.note.strip(),
        reporter=_reporter(ctx),
    )
    db.add(report)
    db.flush()
    audit.record(
        db,
        ctx,
        "problem.reported",
        resource_type="problem_report",
        resource_id=report.id,
        metadata={"conversation_id": conversation.id, "session_id": session_id},
    )
    db.commit()
    db.refresh(report)
    return {"id": report.id, "created_at": _iso(report.created_at)}


@router.get("/conversations/{conversation_id}/reports")
def list_reports(
    conversation_id: UUID,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> list[dict[str, Any]]:
    conversation = get_owned(
        db, ctx, Conversation, str(conversation_id), not_found_detail="conversation_not_found"
    )
    rows = db.scalars(
        scoped(db, ctx, ProblemReport)
        .where(ProblemReport.conversation_id == conversation.id)
        .order_by(ProblemReport.created_at.desc(), ProblemReport.id)
    ).all()
    return [_report_dict(row) for row in rows]
