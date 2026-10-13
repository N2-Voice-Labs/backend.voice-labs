"""Translate native Pipecat frames into the platform's versioned events."""

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .events import RuntimeEventType
from .session import RuntimeSession, SessionState


class RuntimeEventProcessor(FrameProcessor):
    """Forward Pipecat frames and emit transcript and lifecycle events."""

    def __init__(
        self,
        session: RuntimeSession,
        *,
        assistant_output: bool = False,
        control_events: bool = True,
    ) -> None:
        super().__init__()
        self._session = session
        self._assistant_output = assistant_output
        self._control_events = control_events
        self._assistant_text: list[str] = []
        self._assistant_speaking = False

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.finalized:
            language = frame.language.value if frame.language is not None else None
            await self._emit(
                "TranscriptFinal",
                {
                    "content": frame.text,
                    "role": "user",
                    "user_id": frame.user_id,
                    "timestamp": frame.timestamp,
                    "primary_language": language,
                    "detected_languages": [language] if language else [],
                },
            )
        elif self._assistant_output and isinstance(frame, TextFrame):
            self._assistant_text.append(frame.text)
        elif self._assistant_output and isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._assistant_text)
            self._assistant_text.clear()
            if text.strip():
                await self._emit("TranscriptFinal", {"content": text, "role": "assistant"})
        elif self._control_events and isinstance(frame, UserStartedSpeakingFrame):
            self._session.begin_listening()
            await self._emit("SpeechStarted")
        elif self._control_events and isinstance(frame, UserStoppedSpeakingFrame):
            await self._emit("SpeechStopped")
        elif self._control_events and isinstance(frame, BotStartedSpeakingFrame):
            self._assistant_speaking = True
            self._session.state = SessionState.SPEAKING
            await self._emit("AgentSpeechStarted")
        elif self._control_events and isinstance(frame, BotStoppedSpeakingFrame):
            self._assistant_speaking = False
            if self._session.state != SessionState.ENDED:
                self._session.begin_listening()
            await self._emit("TurnCompleted")
        elif self._control_events and isinstance(frame, InterruptionFrame):
            assistant_was_speaking = self._assistant_speaking
            self._assistant_speaking = False
            self._session.interrupt()
            await self._emit("UserInterrupted")
            if assistant_was_speaking:
                await self._emit("AgentSpeechCancelled")

        await self.push_frame(frame, direction)

    async def _emit(
        self,
        event_type: RuntimeEventType,
        payload: dict[str, object] | None = None,
    ) -> None:
        if self._session.event_client:
            self._session.enqueue_event(self._session.event(event_type, payload))
