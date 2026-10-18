"""Read-only analytics over conversations, messages, and runtime events."""

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db
from .models import Conversation, Message, RuntimeEvent

router = APIRouter(tags=["analytics"])

# Runtime emits {"metric": "stt_ttfb"|"llm_ttfb"|"tts_ttfb"|"turn_to_first_audio", "value_ms": n};
# the alternative keys below are tolerated for other emitters.
METRIC_KEYS = ("metric", "stage", "name")
VALUE_KEYS = ("value_ms", "latency_ms", "duration_ms", "value")


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


@router.get("/analytics/summary")
async def analytics_summary(
    agent_id: UUID | None = None,
    since: datetime | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    conversations = select(Conversation.id)
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
        .filter(Conversation.id.in_(conversations))
        .all()
    )
    languages = Counter(
        language or "unknown"
        for (language,) in db.query(Message.primary_language).filter(
            Message.conversation_id.in_(conversations)
        )
    )
    samples: dict[str, list[float]] = {}
    for (payload,) in db.query(RuntimeEvent.payload).filter(
        RuntimeEvent.event_type == "LatencyMeasured",
        RuntimeEvent.conversation_id.in_(conversations),
    ):
        sample = latency_sample(payload)
        if sample:
            samples.setdefault(sample[0], []).append(sample[1])
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
    }
