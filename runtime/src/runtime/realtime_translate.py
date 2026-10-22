"""Streaming interpreter engine: OpenAI ``gpt-realtime-translate`` over a WebSocket.

Selected with ``interpreter.engine == "realtime"`` (the cascaded STT -> LLM -> TTS engine
in :mod:`runtime.interpreter` stays the default). One session translates a continuous
audio stream into one fixed output language, so this engine is one-way only
(``languages[0]`` -> ``languages[1]``); see "Two-way" below.

Verified API (2026-10-18)
-------------------------
Sources: https://developers.openai.com/api/docs/guides/realtime-translation,
https://developers.openai.com/api/reference/resources/realtime/translation-client-events,
https://developers.openai.com/api/reference/resources/realtime/translation-server-events,
https://developers.openai.com/cookbook/examples/voice_solutions/realtime_translation_guide,
https://developers.openai.com/api/docs/models/gpt-realtime-translate

* ``wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate`` with
  ``Authorization: Bearer <key>`` (and ``OpenAI-Safety-Identifier: <hashed user id>``).
* Client events: ``session.update`` (``session.audio.output.language``,
  ``session.audio.input.transcription.model``, ``session.audio.input.noise_reduction``),
  ``session.input_audio_buffer.append`` (``audio``: base64 PCM16, 24 kHz, mono,
  little-endian; ~200 ms chunks; stream continuously, silence included), ``session.close``.
* Server events: ``session.created`` / ``session.updated`` / ``session.closed``,
  ``session.output_audio.delta`` (``delta``: base64 audio, plus ``sample_rate``,
  ``channels``, ``format: "pcm16"``, ``elapsed_ms``), ``session.output_transcript.delta``
  (translated text, append-only), ``session.input_transcript.delta`` (source text,
  append-only, only when ``audio.input.transcription`` is configured) and ``error``.
* There is no response/turn concept, no ``response.create`` and no "transcript done" event:
  transcripts are continuous append-only deltas. The API does not report the detected
  source language and has no voice parameter (the voice adapts to the speaker), so the
  user's own cloned voice and the agent's voice profile are not used by this engine.
* ``session.update`` accepts no instructions, prompt or glossary field, so the NS-08
  glossary (``interpreter.glossary``) does not apply to this engine: it is honoured by the
  cascaded engine only. Use the cascaded engine when terminology must be enforced.
* Output languages (13): es pt fr ja ru zh de ko hi id vi it en. Input: 70+ languages,
  detected by the model. Pricing: $0.034 per minute of audio.

Utterances
----------
Because the API has no turn boundaries, the processor builds them: translated audio is
played as it arrives, and the accumulated source/translated transcripts are emitted as one
``TranscriptFinal`` pair (user = source text, assistant = translated text) once the
speaker has stopped (Silero VAD, when present in the pipeline) and no transcript delta has
arrived for :data:`TRANSCRIPT_IDLE_SECONDS`. Without VAD frames only the idle rule applies.
The source language is the configured ``languages[0]`` (the API does not report it).

Latency metrics (``LatencyMeasured``, ``processor="realtime-translate"``)
------------------------------------------------------------------------
* ``realtime_first_audio_ms``: from the start of the user's speech (VAD start, backdated by
  the VAD confirmation window) to the arrival of the first translated audio chunk of that
  utterance in this process. Emitted once per utterance; needs VAD frames. It excludes
  output-transport buffering. Because the model streams, it is usually shorter than the
  speaker's utterance and is *not* comparable to the cascaded engine's end-of-speech metric.
* ``turn_to_first_audio`` is emitted by :class:`RuntimeEventProcessor`, exactly as for the
  cascaded engine: from ``UserStoppedSpeakingFrame`` (re-emitted here from the VAD stop) to
  the output transport's ``BotStartedSpeakingFrame``. It appears only when translated
  audio *starts* after the speaker stopped; when playback already began during speech there
  is no value (the streaming engine has no end-of-speech wait to measure).

Failure handling
----------------
A socket error (connect failure, close, send/receive error, an ``error`` event before the
session is ready) triggers one reconnect attempt after :data:`RECONNECT_DELAY_SECONDS`;
audio captured meanwhile is queued (bounded, oldest dropped). The retry allowance is
restored after a connection stayed up for :data:`STABLE_CONNECTION_SECONDS`. When the
retry fails too, the session is marked failed (``SessionFailed`` with reason
``realtime_translate_failed``) and the pipeline is ended.

Two-way
-------
OpenAI's guidance for two-way calls is one session per direction, each session's output
language being the *listener's* language, fed with that speaker's audio. A single input
stream with unknown language cannot be routed to the right session because the API reports
no source language, so ``direction: "auto"`` is rejected by Platform; a future two-leg
(SIP) bridge can run two processors, one per leg.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .interpreter import InterpreterSettings
from .timeline import VOICED_PEAK_THRESHOLD, TimelineTracker, UtteranceTimeline, is_voiced  # noqa: F401

_logger = logging.getLogger(__name__)

REALTIME_TRANSLATE_MODEL = "gpt-realtime-translate"
REALTIME_TRANSLATE_URL = "wss://api.openai.com/v1/realtime/translations"
SOURCE_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
REALTIME_TRANSLATE_OUTPUT_LANGUAGES = frozenset(
    {"es", "pt", "fr", "ja", "ru", "zh", "de", "ko", "hi", "id", "vi", "it", "en"}
)

INPUT_SAMPLE_RATE = 24000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
INPUT_CHUNK_MS = 200  # the API's recommended append size
MAX_PENDING_CHUNKS = 150  # 30 s of audio kept while reconnecting; oldest dropped
TRANSCRIPT_IDLE_SECONDS = 1.5
TRANSCRIPT_SOFT_LIMIT_CHARS = 800
RECONNECT_DELAY_SECONDS = 0.5
STABLE_CONNECTION_SECONDS = 30.0
CLOSE_TIMEOUT_SECONDS = 5.0
WS_CLOSE_TIMEOUT_SECONDS = 0.5
# The API streams audio continuously, silence included. Chunks whose peak stays below the
# voiced threshold (``timeline.VOICED_PEAK_THRESHOLD``, shared with the TTS audio check)
# are played as plain audio, so they neither count as the agent speaking nor stop the
# first-audio latency clock.
FAILURE_REASON = "realtime_translate_failed"
# Input chunks whose queue time is remembered to date "translation started" (T3).
RECENT_CHUNK_TIMES = 16


_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")

# connect(url, headers) -> websocket-like: ``send(str)``, ``close()`` and ``async for``
# over received text frames. ``websockets.asyncio.client.connect`` fits.
ConnectFn = Callable[[str, dict[str, str]], Awaitable[Any]]


class RealtimeTranslateError(RuntimeError):
    """The translation socket failed in a way that warrants a reconnect or a failure."""


def realtime_translate_url(model: str = REALTIME_TRANSLATE_MODEL) -> str:
    return f"{REALTIME_TRANSLATE_URL}?model={model}"


async def default_connect(url: str, headers: dict[str, str]) -> Any:
    from websockets.asyncio.client import connect

    return await connect(url, additional_headers=headers, open_timeout=10, max_size=None)


def validate_realtime_settings(settings: InterpreterSettings) -> None:
    """Raise RuntimeError unless the realtime engine can serve ``settings``."""

    if settings.direction != "one_way":
        raise RuntimeError("realtime interpreter engine supports direction one_way only")
    if settings.languages[1] not in REALTIME_TRANSLATE_OUTPUT_LANGUAGES:
        raise RuntimeError(
            f"realtime interpreter engine cannot output language: {settings.languages[1]}"
        )


def build_session_update(
    target: str,
    *,
    transcription_model: str | None = SOURCE_TRANSCRIPTION_MODEL,
    noise_reduction: str | None = "near_field",
) -> dict[str, Any]:
    audio_input: dict[str, Any] = {}
    if transcription_model:
        audio_input["transcription"] = {"model": transcription_model}
    if noise_reduction:
        audio_input["noise_reduction"] = {"type": noise_reduction}
    audio: dict[str, Any] = {"output": {"language": target}}
    if audio_input:
        audio["input"] = audio_input
    return {"type": "session.update", "session": {"audio": audio}}


class RealtimeTranslateProcessor(FrameProcessor):
    """Stream user audio to ``gpt-realtime-translate`` and play the translation back.

    Sits between the LiveKit input (plus optional VAD) and the output transport. Input
    audio is consumed (resampled to 24 kHz and appended to the socket); translated audio
    leaves as :class:`TTSAudioRawFrame` at the sample rate the API reports. VAD frames are
    forwarded and mirrored as ``UserStarted/StoppedSpeakingFrame`` so the downstream
    :class:`RuntimeEventProcessor` emits speech and ``turn_to_first_audio`` events.
    """

    def __init__(
        self,
        *,
        settings: InterpreterSettings,
        api_key: str,
        session: Any | None = None,
        model: str = REALTIME_TRANSLATE_MODEL,
        url: str | None = None,
        connect: ConnectFn | None = None,
        transcription_model: str | None = SOURCE_TRANSCRIPTION_MODEL,
        noise_reduction: str | None = "near_field",
        chunk_ms: int = INPUT_CHUNK_MS,
        transcript_idle_seconds: float = TRANSCRIPT_IDLE_SECONDS,
        reconnect_delay: float = RECONNECT_DELAY_SECONDS,
        stable_connection_seconds: float = STABLE_CONNECTION_SECONDS,
        close_timeout: float = CLOSE_TIMEOUT_SECONDS,
        timeline: TimelineTracker | None = None,
    ) -> None:
        super().__init__()
        validate_realtime_settings(settings)
        self._timeline = timeline
        self._utterance: UtteranceTimeline | None = None
        self._chunk_times: deque[float] = deque(maxlen=RECENT_CHUNK_TIMES)
        self._source, self._target = settings.languages
        self._api_key = api_key
        self._session = session
        self._model = model
        self._url = url or realtime_translate_url(model)
        self._connect = connect or default_connect
        self._transcription_model = transcription_model
        self._noise_reduction = noise_reduction
        self._chunk_bytes = INPUT_SAMPLE_RATE * chunk_ms // 1000 * 2
        self._idle_seconds = transcript_idle_seconds
        self._reconnect_delay = reconnect_delay
        self._stable_seconds = stable_connection_seconds
        self._close_timeout = close_timeout

        self._resampler = create_stream_resampler()
        self._pending = bytearray()
        self._chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._run_task: asyncio.Task | None = None
        self._ws: Any | None = None
        self._stopping = False
        self._failed = False

        self._source_text: list[str] = []
        self._target_text: list[str] = []
        self._flush_handle: asyncio.TimerHandle | None = None
        self._user_speaking = False
        self._speech_started_at: float | None = None

    # -- frame handling --------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._ensure_run_task()
        elif isinstance(frame, EndFrame):
            await self._finish()
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self._abort()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputAudioRawFrame):
            await self._on_input_audio(frame)  # consumed: not forwarded
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._on_speech_started(frame)
            await self.push_frame(frame, direction)
            await self.push_frame(UserStartedSpeakingFrame(), direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            if self._timeline is not None:
                self._timeline.turn_stopped("")
            self._arm_flush()
            await self.push_frame(frame, direction)
            await self.push_frame(UserStoppedSpeakingFrame(), direction)
        else:
            await self.push_frame(frame, direction)

    def _on_speech_started(self, frame: VADUserStartedSpeakingFrame) -> None:
        self._user_speaking = True
        self._cancel_flush()
        self._speech_started_at = time.monotonic() - max(0.0, frame.start_secs or 0.0)
        self._start_timeline(frame)

    def _start_timeline(self, frame: VADUserStartedSpeakingFrame) -> None:
        """T0/T1 of a new utterance (RM-003); the previous one ends here."""

        tracker = self._timeline
        if tracker is None:
            return
        previous = self._utterance
        utterance = tracker.speech_started("", frame.start_secs)
        if utterance is not None and utterance is not previous:
            # The audio is streamed to the model continuously; translation of this
            # utterance starts with the first chunk queued at or after its T0.
            at = next((t for t in self._chunk_times if t >= utterance.t0), None)
            if at is not None:
                utterance.marks["translation_start"] = at
        self._utterance = utterance

    # -- input audio -----------------------------------------------------------

    async def _on_input_audio(self, frame: InputAudioRawFrame) -> None:
        if self._stopping or self._failed or not frame.audio:
            return
        audio = frame.audio
        if frame.num_channels > 1:
            audio = _downmix(audio, frame.num_channels)
        if frame.sample_rate != INPUT_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, frame.sample_rate, INPUT_SAMPLE_RATE)
        self._pending += audio
        while len(self._pending) >= self._chunk_bytes:
            chunk = bytes(self._pending[: self._chunk_bytes])
            del self._pending[: self._chunk_bytes]
            self._queue_chunk(chunk)

    def _queue_chunk(self, chunk: bytes | None) -> None:
        if self._timeline is not None and chunk is not None:
            now = self._timeline.now()
            self._chunk_times.append(now)
            utterance = self._utterance
            if utterance is not None and "translation_start" not in utterance.marks:
                # Speech already running: this chunk is the first sent for it.
                self._timeline.mark(utterance, "translation_start")
        if self._chunks.qsize() >= MAX_PENDING_CHUNKS:
            try:
                self._chunks.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - single consumer race
                pass
        self._chunks.put_nowait(chunk)

    # -- connection lifecycle --------------------------------------------------

    def _ensure_run_task(self) -> None:
        if self._run_task is None or self._run_task.done():
            self._run_task = self.create_task(self._run(), "realtime-translate")

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        agent_id = getattr(self._session, "agent_id", None)
        if agent_id is not None:
            headers["OpenAI-Safety-Identifier"] = hashlib.sha256(
                str(agent_id).encode()
            ).hexdigest()
        return headers

    async def _run(self) -> None:
        retry_available = True
        while True:
            connected_at: float | None = None
            try:
                ws = await self._connect(self._url, self._headers())
                self._ws = ws
                connected_at = time.monotonic()
                await self._pump(ws)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stopping:
                    _logger.info("Realtime translate socket ended during shutdown: %s", exc)
                    return
                if (
                    connected_at is not None
                    and time.monotonic() - connected_at >= self._stable_seconds
                ):
                    retry_available = True
                if not retry_available:
                    await self._fail(exc)
                    return
                retry_available = False
                _logger.warning(
                    "Realtime translate socket error (%s); reconnecting once",
                    type(exc).__name__,
                )
                await asyncio.sleep(self._reconnect_delay)
            finally:
                await self._close_ws()

    async def _pump(self, ws: Any) -> None:
        """Run one connection until it fails (raise) or shuts down cleanly (return)."""

        await ws.send(
            json.dumps(
                build_session_update(
                    self._target,
                    transcription_model=self._transcription_model,
                    noise_reduction=self._noise_reduction,
                )
            )
        )
        sender = asyncio.ensure_future(self._send_loop(ws))
        receiver = asyncio.ensure_future(self._receive_loop(ws))
        try:
            await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if sender.done() and sender.exception() is None:
                # Graceful shutdown: session.close was sent; let the tail audio arrive.
                try:
                    await asyncio.wait_for(asyncio.shield(receiver), self._close_timeout)
                except (asyncio.TimeoutError, Exception):
                    pass
                return
            for task in (sender, receiver):
                if task.done() and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
            if not self._stopping:
                raise RealtimeTranslateError("realtime translate socket closed")
        finally:
            for task in (sender, receiver):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    async def _send_loop(self, ws: Any) -> None:
        while True:
            chunk = await self._chunks.get()
            if chunk is None:
                await ws.send(json.dumps({"type": "session.close"}))
                return
            await ws.send(
                json.dumps(
                    {
                        "type": "session.input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )

    async def _receive_loop(self, ws: Any) -> None:
        ready = False
        async for message in ws:
            try:
                event = json.loads(message)
            except (TypeError, ValueError):
                _logger.warning("Ignoring non-JSON realtime translate message")
                continue
            if not isinstance(event, Mapping):
                continue
            kind = event.get("type")
            if kind in ("session.created", "session.updated"):
                ready = True
            elif kind == "session.output_audio.delta":
                await self._on_output_audio(event)
            elif kind == "session.output_transcript.delta":
                self._on_transcript_delta(self._target_text, event)
            elif kind == "session.input_transcript.delta":
                self._on_transcript_delta(self._source_text, event)
            elif kind == "session.closed":
                return
            elif kind == "error":
                error = event.get("error") if isinstance(event.get("error"), Mapping) else {}
                detail = f"{error.get('type', 'error')}: {error.get('message', '')}"
                if not ready:
                    raise RealtimeTranslateError(f"session rejected ({detail})")
                _logger.warning("Realtime translate error event: %s", detail)

    async def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            # The server may not answer the close handshake promptly (the real API keeps
            # the socket open after session.closed); never block shutdown on it.
            await asyncio.wait_for(ws.close(), WS_CLOSE_TIMEOUT_SECONDS)
        except Exception:
            _logger.debug("Closing realtime translate socket failed", exc_info=True)
            transport = getattr(ws, "transport", None)
            if transport is not None and hasattr(transport, "abort"):
                transport.abort()

    # -- output ----------------------------------------------------------------

    async def _on_output_audio(self, event: Mapping[str, Any]) -> None:
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        try:
            audio = base64.b64decode(delta)
        except ValueError:
            _logger.warning("Dropped undecodable translated audio chunk")
            return
        if not audio:
            return
        sample_rate = int(event.get("sample_rate") or DEFAULT_OUTPUT_SAMPLE_RATE)
        num_channels = int(event.get("channels") or 1)
        voiced = is_voiced(audio)
        tracker = self._timeline
        if tracker is not None:
            if voiced:
                tracker.enqueue_audio(self._utterance)
            tracker.on_audio(audio, sample_rate, num_channels)
        if not voiced:
            await self.push_frame(OutputAudioRawFrame(audio, sample_rate, num_channels))
            return
        if self._speech_started_at is not None:
            self._emit_latency("realtime_first_audio_ms", time.monotonic() - self._speech_started_at)
            self._speech_started_at = None
        await self.push_frame(TTSAudioRawFrame(audio, sample_rate=sample_rate, num_channels=num_channels))

    def _on_transcript_delta(self, buffer: list[str], event: Mapping[str, Any]) -> None:
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        if self._timeline is not None and buffer is self._source_text:
            self._timeline.mark(self._utterance, "first_transcript")
        buffer.append(delta)
        text = "".join(self._target_text)
        if len(text) >= TRANSCRIPT_SOFT_LIMIT_CHARS and text.rstrip().endswith(_SENTENCE_END):
            self._flush_transcripts()
        else:
            self._arm_flush()

    # -- transcripts -----------------------------------------------------------

    def _arm_flush(self) -> None:
        self._cancel_flush()
        if self._user_speaking or not (self._source_text or self._target_text):
            return
        self._flush_handle = asyncio.get_running_loop().call_later(
            self._idle_seconds, self._flush_transcripts
        )

    def _cancel_flush(self) -> None:
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None

    def _flush_transcripts(self) -> None:
        """Emit the accumulated source/translated text as one user + assistant pair."""

        self._cancel_flush()
        if self._timeline is not None and self._utterance is not None:
            self._timeline.finish(self._utterance)
            self._utterance = None
        source = "".join(self._source_text).strip()
        translated = "".join(self._target_text).strip()
        self._source_text.clear()
        self._target_text.clear()
        if source:
            self._emit(
                "TranscriptFinal",
                {
                    "content": source,
                    "role": "user",
                    "primary_language": self._source,
                    "detected_languages": [self._source],
                },
            )
        if translated:
            self._emit(
                "TranscriptFinal",
                {
                    "content": translated,
                    "role": "assistant",
                    "primary_language": self._target,
                    "detected_languages": [self._target],
                },
            )

    # -- shutdown --------------------------------------------------------------

    async def _finish(self) -> None:
        """EndFrame: send the buffered tail, close the session, wait for the last audio."""

        if self._stopping:
            return
        self._stopping = True
        if self._pending and not self._failed:
            self._queue_chunk(bytes(self._pending))
        self._pending.clear()
        task = self._run_task
        if task is not None and not task.done():
            self._queue_chunk(None)
            try:
                await asyncio.wait_for(asyncio.shield(task), self._close_timeout + 1)
            except (asyncio.TimeoutError, Exception):
                await self.cancel_task(task)
        self._flush_transcripts()

    async def _abort(self) -> None:
        self._stopping = True
        self._flush_transcripts()
        task, self._run_task = self._run_task, None
        if task is not None and not task.done():
            await self.cancel_task(task)
        await self._close_ws()

    async def _fail(self, exc: BaseException) -> None:
        """Permanent failure: record it on the session and end the pipeline.

        Ending goes through a separate task because cancelling the pipeline cancels this
        processor's run task, which is the caller.
        """

        self._failed = True
        _logger.error("Realtime translate failed permanently: %s", type(exc).__name__)
        self._flush_transcripts()
        session = self._session
        if session is not None and hasattr(session, "fail"):
            session.fail(FAILURE_REASON)
        self.create_task(self._end_session(exc), "realtime-translate-fail")

    async def _end_session(self, exc: BaseException) -> None:
        session = self._session
        if session is not None and hasattr(session, "request_end"):
            try:
                if await session.request_end(FAILURE_REASON):
                    return
            except Exception:
                _logger.exception("Could not end the session after a realtime translate failure")
        await self.push_error(
            f"realtime translate failed: {type(exc).__name__}",
            exception=exc if isinstance(exc, Exception) else None,
            force_treat_as_permanent=True,
        )

    # -- events ----------------------------------------------------------------

    def _emit_latency(self, metric: str, seconds: float) -> None:
        self._emit(
            "LatencyMeasured",
            {
                "metric": metric,
                "value_ms": round(seconds * 1000, 1),
                "processor": "realtime-translate",
                "model": self._model,
            },
        )

    def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        """Queue a Platform event; telemetry failures never break interpretation."""

        session = self._session
        if session is None or not getattr(session, "event_client", None):
            return
        try:
            session.enqueue_event(session.event(event_type, payload))
        except RuntimeError:
            _logger.warning("Dropped %s event for session %s", event_type, session.session_id)


def _downmix(audio: bytes, channels: int) -> bytes:
    import numpy as np

    samples = np.frombuffer(audio, dtype=np.int16)
    usable = len(samples) - len(samples) % channels
    mixed = samples[:usable].reshape(-1, channels).mean(axis=1)
    return mixed.astype(np.int16).tobytes()
