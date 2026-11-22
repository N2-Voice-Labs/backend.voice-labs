"""Transport-independent versioned runtime events."""

from datetime import datetime, timezone
from typing import Any, Literal, get_args
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


RuntimeEventType = Literal[
    "SpeechStarted",
    "SpeechStopped",
    "TurnCompleted",
    "AgentSpeechStarted",
    "UserInterrupted",
    "AgentSpeechCancelled",
    "TranscriptFinal",
    "LatencyMeasured",
    "SessionEnded",
    "SessionFailed",
    "RuntimeFailed",
    "RuntimeStarted",
    "ToolCalled",
    "HandoffRequested",
    "OutcomeRecorded",
    "ProviderError",
    "EngineFallback",
    "TurnDebug",
    "AnswerEmpty",
    "AnswerTruncated",
    "TranslationEmpty",
]
EVENT_TYPES: tuple[str, ...] = get_args(RuntimeEventType)

# Delivery classes. Critical events carry state the Platform cannot reconstruct (call start,
# transcript, tool/outcome/handoff results, terminal state): they are retried for a long
# budget and, if delivery still fails, the call is ended (a call whose record is lost is
# worse than a dropped call). Every other type is best-effort telemetry (latency, turn
# debug, speech/turn timeline, provider diagnostics): it is dropped first under queue
# pressure and a failed delivery never affects the call. New event types default to
# best-effort.
TERMINAL_EVENT_TYPES: frozenset[str] = frozenset(
    {"SessionEnded", "SessionFailed", "RuntimeFailed"}
)
CRITICAL_EVENT_TYPES: frozenset[str] = TERMINAL_EVENT_TYPES | frozenset(
    {
        "RuntimeStarted",
        "TranscriptFinal",
        "ToolCalled",
        "HandoffRequested",
        "OutcomeRecorded",
    }
)


def is_critical_event(event_type: str) -> bool:
    return event_type in CRITICAL_EVENT_TYPES


class RuntimeEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: RuntimeEventType
    schema_version: int = 1
    session_id: UUID
    conversation_id: UUID
    agent_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
