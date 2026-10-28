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
