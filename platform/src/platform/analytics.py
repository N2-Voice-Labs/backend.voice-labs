"""Read-only analytics over conversations, messages, and runtime events."""

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .context import AuthContext, require
from .models import Conversation, Message, RuntimeEvent
from .tenancy import get_scoped_db

router = APIRouter(tags=["analytics"])

# Runtime emits {"metric": "stt_ttfb"|"llm_ttfb"|"tts_ttfb"|"turn_to_first_audio", "value_ms": n};
# the alternative keys below are tolerated for other emitters.
METRIC_KEYS = ("metric", "stage", "name")
VALUE_KEYS = ("value_ms", "latency_ms", "duration_ms", "value")
# RM-003: one event per translated utterance with the whole timeline; aggregated by engine
# in the response's ``timeline`` section.
TIMELINE_METRIC = "utterance_timeline"


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty list."""
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def latency_sample(payload: Any) -> tuple[str, float] | None:
    if not isinstance(payload, dict):
        return None
    metric = next(
        (payload[k] for k in METRIC_KEYS if isinstance(payload.get(k), str)), None
    )
    value = next(
        (
            payload[k]
            for k in VALUE_KEYS
            if isinstance(payload.get(k), (int, float))
            and not isinstance(payload.get(k), bool)
            and math.isfinite(payload[k])
        ),
        None,
    )
    if not metric or value is None:
        return None
    # Pipecat reports a zero TTFB when each STT/TTS service connects, before any speech.
    if metric.endswith("_ttfb") and value <= 0:
        return None
    return metric, float(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def timeline_sample(payload: Any) -> tuple[str, float, dict[str, float]] | None:
    """``(engine, value_ms, marks_ms)`` of an ``utterance_timeline`` event, else None."""

    if not isinstance(payload, dict) or payload.get("metric") != TIMELINE_METRIC:
        return None
    value = _number(payload.get("value_ms"))
    if value is None:
        return None
    engine = payload.get("engine")
    marks = payload.get("marks_ms")
    clean = {
        name: number
        for name, raw in (marks.items() if isinstance(marks, dict) else ())
        if isinstance(name, str) and (number := _number(raw)) is not None
    }
    return (engine if isinstance(engine, str) and engine else "unknown", value, clean)


def distribution(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    return {
        "count": len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def _org(model: type, org: list[str]) -> list[Any]:
    return [model.organization_id == org[0]] if org else []  # type: ignore[attr-defined]


@router.get("/analytics/summary")
async def analytics_summary(
    agent_id: UUID | None = None,
    since: datetime | None = None,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    # Every query below carries an explicit organisation filter: the aggregate/column
    # selects must not depend on the session listener alone.
    org = [ctx.org_id] if ctx.org_id else []
    conversations = select(Conversation.id)
    if org:
        conversations = conversations.where(Conversation.organization_id == ctx.org_id)
    if agent_id is not None:
        conversations = conversations.where(Conversation.agent_id == str(agent_id))
    if since is not None:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        conversations = conversations.where(
            Conversation.started_at >= since.astimezone(timezone.utc)
        )
    rows = (
        db.query(Conversation.status, Conversation.outcome, Conversation.channel)
        .filter(Conversation.id.in_(conversations), *_org(Conversation, org))
        .all()
    )
    languages = Counter(
        language or "unknown"
        for (language,) in db.query(Message.primary_language).filter(
            Message.conversation_id.in_(conversations), *_org(Message, org)
        )
    )
    samples: dict[str, list[float]] = {}
    timeline_values: dict[str, list[float]] = {}
    timeline_marks: dict[str, dict[str, list[float]]] = {}
    for (payload,) in db.query(RuntimeEvent.payload).filter(
        RuntimeEvent.event_type == "LatencyMeasured",
        RuntimeEvent.conversation_id.in_(conversations),
        *_org(RuntimeEvent, org),
    ):
        sample = latency_sample(payload)
        if sample:
            samples.setdefault(sample[0], []).append(sample[1])
        timeline = timeline_sample(payload)
        if timeline:
            engine, value, marks = timeline
            timeline_values.setdefault(engine, []).append(value)
            for name, mark in marks.items():
                timeline_marks.setdefault(engine, {}).setdefault(name, []).append(mark)
    latency = {}
    for metric, values in sorted(samples.items()):
        values.sort()
        latency[metric] = {
            "count": len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }
    return {
        "conversations": {
            "total": len(rows),
            "by_status": dict(Counter(row[0] or "unknown" for row in rows)),
            "by_outcome": dict(Counter(row[1] or "unknown" for row in rows)),
            "by_channel": dict(Counter(row[2] or "unknown" for row in rows)),
        },
        "languages": dict(languages),
        "latency": latency,
        # Per engine: p50/p95 of time to first audio (T4 - T0, "value_ms") and of every
        # mark of the utterance timeline (ms after the speaker started).
        "timeline": {
            engine: {
                **distribution(values),
                "marks": {
                    name: distribution(marks)
                    for name, marks in sorted(timeline_marks.get(engine, {}).items())
                },
            }
            for engine, values in sorted(timeline_values.items())
        },
    }
