"""Translate native Pipecat frames into the platform's versioned events."""

import math
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .events import RuntimeEventType
from .session import RuntimeSession, SessionState
from .timeline import TimelineTracker
from .turn_debug import stt_confidence
from .turn_ids import turn_state


def latency_stage(processor_name: str) -> str:
    """Classify a Pipecat processor name such as ``DeepgramSTTService#0``."""

    for marker, stage in (("STT", "stt"), ("LLM", "llm"), ("TTS", "tts")):
        if marker in processor_name:
            return stage
    return "other"


class RuntimeEventProcessor(FrameProcessor):
    """Forward Pipecat frames and emit transcript, lifecycle, and latency events.

    ``LatencyMeasured`` events are emitted only by the instance with
    ``control_events=True`` and ``assistant_output=False`` (the final instance in
    the pipeline, after TTS). Pipecat metrics frames travel downstream through
    every instance, so restricting emission to one avoids duplicates.
    """

    def __init__(
        self,
        session: RuntimeSession,
        *,
        assistant_output: bool = False,
        control_events: bool = True,
        transcript_events: bool = True,
        timeline: TimelineTracker | None = None,
    ) -> None:
        super().__init__()
        self._session = session
        # RM-003: translated audio leaving the pipeline (only the latency instance feeds it).
        self._timeline = timeline
        # Off when another processor owns the transcripts (agent + translator).
        self._transcript_events = transcript_events
        self._assistant_output = assistant_output
        self._control_events = control_events
        self._assistant_text: list[str] = []
        self._assistant_speaking = False
        self._latency_events = control_events and not assistant_output
        self._user_stopped_at: float | None = None
        # ST-02a: the session-wide turn counter; ``_answering_turn`` is the turn the user
        # just finished, which the latency events of the response belong to.
        self._turns = turn_state(session)
        self._answering_turn: int | None = None

    def _turn_id(self) -> int | None:
        if self._turns is None:
            return None
        return self._turns.current or None

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self._timeline is not None and self._latency_events:
            self._feed_timeline(frame, direction)
        if self._transcript_events and isinstance(frame, TranscriptionFrame) and frame.finalized:
            language = frame.language.value if frame.language is not None else None
            payload: dict[str, object] = {
                "content": frame.text,
                "role": "user",
                "user_id": frame.user_id,
                "timestamp": frame.timestamp,
                "primary_language": language,
                "detected_languages": [language] if language else [],
            }
            if self._turns is not None and not self._turns.current:
                self._turns.begin()  # a transcript before any turn start opens the first turn
            turn_id = self._turn_id()
            if turn_id is not None:
                payload["turn_id"] = turn_id
            confidence = stt_confidence(getattr(frame, "result", None))
            if confidence is not None:
                payload["confidence"] = confidence
            await self._emit("TranscriptFinal", payload)
        elif self._assistant_output and isinstance(frame, TextFrame):
            self._assistant_text.append(frame.text)
        elif self._assistant_output and isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._assistant_text)
            self._assistant_text.clear()
            if text.strip():
                payload = {"content": text, "role": "assistant"}
                turn_id = self._turn_id()
                if turn_id is not None:
                    payload["turn_id"] = turn_id
                await self._emit("TranscriptFinal", payload)
        elif self._latency_events and isinstance(frame, MetricsFrame):
            await self._emit_ttfb_latency(frame)
        elif self._control_events and isinstance(frame, UserStartedSpeakingFrame):
            self._user_stopped_at = None
            if self._turns is not None and direction == FrameDirection.DOWNSTREAM:
                self._turns.begin()
            self._session.begin_listening()
            await self._emit("SpeechStarted")
        elif self._control_events and isinstance(frame, UserStoppedSpeakingFrame):
            if self._latency_events:
                self._user_stopped_at = time.monotonic()
                self._answering_turn = self._turn_id()
            if self._turns is not None and direction == FrameDirection.DOWNSTREAM:
                self._turns.end()
            await self._emit("SpeechStopped")
        elif self._control_events and isinstance(frame, BotStartedSpeakingFrame):
            if self._latency_events and self._user_stopped_at is not None:
                elapsed = time.monotonic() - self._user_stopped_at
                self._user_stopped_at = None
                await self._emit_latency("turn_to_first_audio", elapsed, "runtime", None)
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

    def _feed_timeline(self, frame, direction: FrameDirection) -> None:
        timeline = self._timeline
        assert timeline is not None
        if isinstance(frame, BotStoppedSpeakingFrame):
            timeline.on_idle()
        elif direction != FrameDirection.DOWNSTREAM:
            return
        elif isinstance(frame, TTSAudioRawFrame):
            timeline.on_audio(
                frame.audio, frame.sample_rate, frame.num_channels, getattr(frame, "transport_destination", None)
            )
        elif isinstance(frame, TTSStoppedFrame):
            timeline.on_tts_stopped(getattr(frame, "transport_destination", None))

    async def _emit_ttfb_latency(self, frame: MetricsFrame) -> None:
        for data in frame.data:
            if isinstance(data, TTFBMetricsData):
                await self._emit_latency(
                    f"{latency_stage(data.processor)}_ttfb",
                    data.value,
                    data.processor,
                    data.model,
                )

    async def _emit_latency(
        self,
        metric: str,
        seconds: object,
        processor: str,
        model: str | None,
    ) -> None:
        """Emit LatencyMeasured; invalid values are skipped and never raise."""

        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            return
        payload: dict[str, object] = {
            "metric": metric,
            "value_ms": round(seconds * 1000, 1),
            "processor": processor,
            "model": model,
        }
        turn_id = self._answering_turn or self._turn_id()
        if turn_id is not None:
            payload["turn_id"] = turn_id
        try:
            await self._emit("LatencyMeasured", payload)
        except RuntimeError:
            # Telemetry must not break the media pipeline (e.g. event queue closed).
            pass

    async def _emit(
        self,
        event_type: RuntimeEventType,
        payload: dict[str, object] | None = None,
    ) -> None:
        if self._session.event_client:
            self._session.enqueue_event(self._session.event(event_type, payload))
