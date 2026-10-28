"""Per-turn debug data (ST-02a): what STT heard, what the LLM answered, what TTS spoke.

Two parts:

* :class:`~runtime.turn_ids.TurnState` (in ``turn_ids``, importable without Pipecat) - a per-session, monotonically increasing ``turn_id`` shared by the
  event processor, the timeline tracker and the observer (:func:`turn_state`). A turn
  starts with ``UserStartedSpeakingFrame`` (agent mode) or the VAD start of an utterance
  (timeline tracker) and ends with ``UserStoppedSpeakingFrame``; ``begin`` is idempotent
  while a turn is open, so several components can announce the same start.
* :class:`TurnDebugObserver` - a Pipecat observer (no pipeline processor, like
  :class:`~runtime.stall_watchdog.StallWatchdog`) that collects the facts of one turn and
  emits ONE ``TurnDebug`` event when the assistant's response finishes (or the turn is
  superseded / the session ends without a response).

``TurnDebug`` payload::

    {"turn_id": 3,
     "stt":   {"text", "language", "confidence"},            # confidence: mean of the finals
     "llm":   {"model", "input_tokens", "output_tokens", "first_token_ms"} | null,
     "tts":   {"provider", "voice_id", "text", "first_audio_ms"} | null,
     "input": {"rms_dbfs", "peak_dbfs", "voiced_ms"}}

Every leaf may be ``null`` when the provider did not report it. ``llm`` is null when the
turn got no LLM activity, ``tts`` when nothing was synthesized. ``input`` covers the
user's ``InputAudioRawFrame`` audio while the VAD said "speaking" (dBFS relative to int16
full scale; ``voiced_ms`` counts the frames whose RMS is above :data:`VOICED_RMS_DBFS`).

Frames used (pipecat 1.12): ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` (turn),
``VADUserStarted/StoppedSpeakingFrame`` (audio window), ``TranscriptionFrame`` (``result``
is Deepgram's ``ListenV1Results``: ``channel.alternatives[0].confidence``),
``LLMContextFrame`` into the LLM and the first ``LLMTextFrame`` (first token),
``MetricsFrame`` with ``LLMUsageMetricsData`` (tokens, model) and ``TTFBMetricsData``,
``TTSStartedFrame`` -> first ``TTSAudioRawFrame`` (first audio), ``TTSTextFrame`` (spoken
text), ``BotStoppedSpeakingFrame`` (end of the response).

A response that is still being produced (LLM request or function call in flight) is not
ended by a ``BotStoppedSpeakingFrame`` (a tool-call preamble), so one turn yields one event.

Limits: agent mode only. The interpreter engines translate per lane utterance without
a user-facing LLM/TTS answer; they get ``turn_id`` on ``utterance_timeline`` but no
``TurnDebug``. Telemetry never raises into the pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

from .provider_failures import provider_kind
from .turn_ids import TurnState, turn_state

_logger = logging.getLogger(__name__)

EVENT_TYPE = "TurnDebug"
INT16_FULL_SCALE = 32768.0
SILENCE_DBFS = -100.0
VOICED_RMS_DBFS = -50.0  # a frame quieter than this is not counted as voiced audio


# -- audio level ----------------------------------------------------------------------


def _dbfs(value: float) -> float:
    if value <= 0:
        return SILENCE_DBFS
    return round(max(SILENCE_DBFS, 20 * math.log10(value / INT16_FULL_SCALE)), 1)


def frame_levels(audio: bytes) -> tuple[float, int, int]:
    """``(sum_of_squares, peak, samples)`` of little-endian int16 PCM."""

    import numpy as np

    usable = audio[: len(audio) - len(audio) % 2]
    if not usable:
        return 0.0, 0, 0
    samples = np.frombuffer(usable, dtype=np.int16).astype(np.float64)
    return float(np.dot(samples, samples)), int(np.max(np.abs(samples))), int(samples.size)


@dataclass
class AudioLevel:
    """Running RMS/peak over the user's speech audio."""

    sum_squares: float = 0.0
    samples: int = 0
    peak: int = 0
    voiced_ms: float = 0.0

    def add(self, audio: bytes, sample_rate: int, channels: int = 1) -> None:
        squares, peak, count = frame_levels(audio)
        if not count:
            return
        self.sum_squares += squares
        self.samples += count
        self.peak = max(self.peak, peak)
        if _dbfs(math.sqrt(squares / count)) >= VOICED_RMS_DBFS and sample_rate > 0:
            self.voiced_ms += count / max(1, channels) / sample_rate * 1000

    def payload(self) -> dict[str, float | None]:
        if not self.samples:
            return {"rms_dbfs": None, "peak_dbfs": None, "voiced_ms": 0.0}
        return {
            "rms_dbfs": _dbfs(math.sqrt(self.sum_squares / self.samples)),
            "peak_dbfs": _dbfs(self.peak),
            "voiced_ms": round(self.voiced_ms, 1),
        }


# -- STT confidence -------------------------------------------------------------------


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def stt_confidence(result: Any) -> float | None:
    """Confidence (0..1) of a transcription result, None when the STT gave none.

    Deepgram: ``ListenV1Results.channel.alternatives[0].confidence`` (object or dict).
    """

    try:
        alternatives = _field(_field(result, "channel"), "alternatives")
        if not alternatives:
            return None
        value = _field(alternatives[0], "confidence")
    except Exception:  # pragma: no cover - defensive: never break the pipeline
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value), 4) if 0 <= value <= 1 else None


# -- observer -------------------------------------------------------------------------


@dataclass
class _Turn:
    id: int
    audio: AudioLevel
    user_done: bool = False
    stt_parts: list[str] = field(default_factory=list)
    stt_language: str | None = None
    stt_confidences: list[float] = field(default_factory=list)
    llm_seen: bool = False
    llm_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    llm_requested_at: float | None = None
    first_token_ms: float | None = None
    llm_ttfb_ms: float | None = None
    llm_text: list[str] = field(default_factory=list)
    tts_seen: bool = False
    tts_voice: str | None = None
    tts_started_at: float | None = None
    first_audio_ms: float | None = None
    tts_ttfb_ms: float | None = None
    tts_text: list[tuple[str, bool]] = field(default_factory=list)

    @property
    def responded(self) -> bool:
        return self.llm_seen or self.tts_seen


def _join(parts: list[tuple[str, bool]]) -> str:
    text = ""
    for part, spaced in parts:
        text += part if spaced or not text else f" {part}"
    return text.strip()


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class TurnDebugObserver(BaseObserver):
    """Collect per-turn STT/LLM/TTS facts and emit one ``TurnDebug`` event per turn."""

    def __init__(
        self,
        session: Any,
        providers: Mapping[str, str] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._session = session
        self._providers = dict(providers or {})
        self._clock = clock
        self._state = turn_state(session) or TurnState()
        self._turn: _Turn | None = None
        self._audio = AudioLevel()
        self._vad_speaking = False
        self._flushed_id = 0
        self._llm_inflight = 0
        self._functions_inflight = 0

    # -- observer ------------------------------------------------------------------

    async def on_push_frame(self, data: FramePushed) -> None:
        try:
            self._handle(data)
        except Exception:  # pragma: no cover - telemetry must not break the pipeline
            _logger.debug("TurnDebug observer failed", exc_info=True)

    async def cleanup(self) -> None:
        self.flush()
        await super().cleanup()

    def _handle(self, data: FramePushed) -> None:
        frame = data.frame
        if isinstance(frame, LLMContextFrame):
            # Every hop sees this frame; only the push into the LLM service counts.
            if provider_kind(data.destination) == "llm":
                turn = self._current()
                if turn is not None:
                    turn.llm_seen = True
                    turn.llm_requested_at = turn.llm_requested_at or self._clock()
                    turn.llm_model = turn.llm_model or self._service_model(data.destination)
                self._llm_inflight += 1
            return
        if not data.first_push:
            return
        if isinstance(frame, InputAudioRawFrame):
            if self._vad_speaking:
                self._audio.add(frame.audio, frame.sample_rate, frame.num_channels)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._vad_speaking:
                self._vad_speaking = True
                if self._turn is None or self._turn.user_done:
                    self._audio = AudioLevel()  # speech of a new turn, not the previous one
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_speaking = False
        elif isinstance(frame, UserStartedSpeakingFrame):
            if data.direction == FrameDirection.DOWNSTREAM:
                self._start_turn()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if data.direction == FrameDirection.DOWNSTREAM:
                self._state.end()
                if self._turn is not None:
                    self._turn.user_done = True
        elif isinstance(frame, TranscriptionFrame):
            self._on_transcript(frame)
        elif isinstance(frame, LLMTextFrame):
            self._on_llm_text(frame, data.source)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if provider_kind(data.source) == "llm":
                self._llm_inflight = max(0, self._llm_inflight - 1)
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._functions_inflight += 1
        elif isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame)):
            self._functions_inflight = max(0, self._functions_inflight - 1)
        elif isinstance(frame, MetricsFrame):
            self._on_metrics(frame)
        elif isinstance(frame, TTSStartedFrame):
            if provider_kind(data.source) == "tts":
                self._on_tts_started(data.source)
        elif isinstance(frame, TTSAudioRawFrame):
            if provider_kind(data.source) == "tts":
                self._on_tts_audio()
        elif isinstance(frame, TTSTextFrame):
            if provider_kind(data.source) == "tts" and frame.text:
                turn = self._current()
                if turn is not None:
                    turn.tts_seen = True
                    turn.tts_text.append((frame.text, bool(frame.includes_inter_frame_spaces)))
        elif isinstance(frame, BotStoppedSpeakingFrame):
            turn = self._turn
            if (
                turn is not None
                and turn.responded
                and self._llm_inflight == 0
                and self._functions_inflight == 0
            ):
                self.flush()
        elif isinstance(frame, (CancelFrame, EndFrame)):
            self.flush()

    # -- turn bookkeeping ----------------------------------------------------------

    def _start_turn(self) -> None:
        self.flush()  # the previous turn ends here, with or without a response
        turn_id = self._state.begin()
        if turn_id == self._flushed_id:  # already reported (defensive)
            return
        self._turn = _Turn(id=turn_id, audio=self._audio)
        self._llm_inflight = 0
        self._functions_inflight = 0

    def _current(self) -> _Turn | None:
        """The record of the state's current turn, created if a frame arrives before the start."""

        turn_id = self._state.current or self._state.begin()
        if self._turn is not None and self._turn.id == turn_id:
            return self._turn
        if turn_id == self._flushed_id:
            return None  # late data of a turn already reported
        self.flush()
        self._turn = _Turn(id=turn_id, audio=self._audio)
        return self._turn

    # -- collectors ----------------------------------------------------------------

    def _on_transcript(self, frame: TranscriptionFrame) -> None:
        turn = self._current()
        if turn is None:
            return
        if frame.text and frame.text.strip():
            turn.stt_parts.append(frame.text.strip())
        if frame.language is not None:
            turn.stt_language = getattr(frame.language, "value", None) or str(frame.language)
        confidence = stt_confidence(getattr(frame, "result", None))
        if confidence is not None:
            turn.stt_confidences.append(confidence)

    def _on_llm_text(self, frame: LLMTextFrame, source: Any) -> None:
        if provider_kind(source) != "llm" or not frame.text:
            return
        turn = self._current()
        if turn is None:
            return
        turn.llm_seen = True
        turn.llm_text.append(frame.text)
        turn.llm_model = turn.llm_model or self._service_model(source)
        if turn.first_token_ms is None and turn.llm_requested_at is not None:
            turn.first_token_ms = round((self._clock() - turn.llm_requested_at) * 1000, 1)

    def _on_metrics(self, frame: MetricsFrame) -> None:
        for item in frame.data:
            if isinstance(item, LLMUsageMetricsData):
                turn = self._current()
                if turn is None:
                    continue
                turn.llm_seen = True
                turn.llm_model = _text(item.model) or turn.llm_model
                usage = item.value
                turn.input_tokens = (turn.input_tokens or 0) + int(usage.prompt_tokens)
                turn.output_tokens = (turn.output_tokens or 0) + int(usage.completion_tokens)
            elif isinstance(item, TTFBMetricsData):
                kind = provider_kind_of_name(item.processor)
                turn = self._turn if kind in ("llm", "tts") else None
                if turn is None or not _valid_seconds(item.value):
                    continue
                if kind == "llm" and turn.llm_ttfb_ms is None:
                    turn.llm_ttfb_ms = round(item.value * 1000, 1)
                elif kind == "tts" and turn.tts_ttfb_ms is None:
                    turn.tts_ttfb_ms = round(item.value * 1000, 1)

    def _on_tts_started(self, source: Any) -> None:
        turn = self._current()
        if turn is None:
            return
        turn.tts_seen = True
        if turn.tts_started_at is None:
            turn.tts_started_at = self._clock()
        turn.tts_voice = turn.tts_voice or _text(
            getattr(getattr(source, "_settings", None), "voice", None)
        )

    def _on_tts_audio(self) -> None:
        turn = self._current()
        if turn is None:
            return
        turn.tts_seen = True
        if turn.first_audio_ms is None and turn.tts_started_at is not None:
            turn.first_audio_ms = round((self._clock() - turn.tts_started_at) * 1000, 1)

    @staticmethod
    def _service_model(service: Any) -> str | None:
        return _text(getattr(getattr(service, "_settings", None), "model", None))

    # -- emission ------------------------------------------------------------------

    def flush(self) -> None:
        """Emit ``TurnDebug`` for the open turn record (if any) and forget it."""

        turn, self._turn = self._turn, None
        if turn is None:
            return
        self._flushed_id = turn.id
        payload = self.build_payload(turn)
        session = self._session
        if session is None or not getattr(session, "event_client", None):
            return
        try:
            session.enqueue_event(session.event(EVENT_TYPE, payload))
        except RuntimeError:
            _logger.debug("Dropped TurnDebug for turn %s (event queue closed)", turn.id)

    def build_payload(self, turn: _Turn) -> dict[str, Any]:
        confidence = (
            round(sum(turn.stt_confidences) / len(turn.stt_confidences), 4)
            if turn.stt_confidences
            else None
        )
        llm = None
        if turn.llm_seen:
            llm = {
                "model": turn.llm_model,
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "first_token_ms": (
                    turn.first_token_ms if turn.first_token_ms is not None else turn.llm_ttfb_ms
                ),
            }
        tts = None
        if turn.tts_seen:
            tts = {
                "provider": self._providers.get("tts"),
                "voice_id": turn.tts_voice,
                "text": _join(turn.tts_text) or "".join(turn.llm_text).strip() or None,
                "first_audio_ms": (
                    turn.first_audio_ms if turn.first_audio_ms is not None else turn.tts_ttfb_ms
                ),
            }
        return {
            "turn_id": turn.id,
            "stt": {
                "text": " ".join(turn.stt_parts),
                "language": turn.stt_language,
                "confidence": confidence,
            },
            "llm": llm,
            "tts": tts,
            "input": turn.audio.payload(),
        }


def provider_kind_of_name(processor_name: str) -> str | None:
    """Classify a metrics ``processor`` name such as ``OpenAILLMService#0``."""

    for marker, kind in (("STT", "stt"), ("LLM", "llm"), ("TTS", "tts")):
        if marker in processor_name:
            return kind
    return None


def _valid_seconds(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def watch_turn_debug(worker: Any, session: Any | None, providers: Mapping[str, str]) -> None:
    """Attach a :class:`TurnDebugObserver` to ``worker`` (agent mode; no-op without a session)."""

    if session is not None and hasattr(worker, "add_observer"):
        worker.add_observer(TurnDebugObserver(session, providers))
