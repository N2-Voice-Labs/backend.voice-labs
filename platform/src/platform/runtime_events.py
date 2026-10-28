"""Runtime event type names accepted and surfaced by the platform.

Mirrors ``runtime.events.EVENT_TYPES``; a drift test keeps them in sync.
"""

ACCEPTED_EVENT_TYPES: frozenset[str] = frozenset(
    {
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
    }
)

RUNTIME_STATE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "SpeechStarted",
        "SpeechStopped",
        "TurnCompleted",
        "AgentSpeechStarted",
        "UserInterrupted",
        "AgentSpeechCancelled",
        "RuntimeStarted",
        "SessionEnded",
        "SessionFailed",
        "RuntimeFailed",
        "ToolCalled",
        "HandoffRequested",
        "OutcomeRecorded",
        "ProviderError",
        "EngineFallback",
        "AnswerEmpty",
        "TranslationEmpty",
    }
)
