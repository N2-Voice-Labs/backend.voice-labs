"""Stall watchdog (ST-10a): find a provider that hangs without raising any error.

A dead STT socket, an LLM request that never answers or a TTS that never returns audio
raise nothing: the call just goes quiet. :class:`StallWatchdog` is a Pipecat pipeline
*observer* (it watches frames, it does not sit in the pipeline, so it works for every
processor order, lanes included) that times three hand-offs and reports an overrun through
:func:`runtime.provider_failures.report_provider_failure` with category ``timeout``, which
emits ``ProviderError`` and applies the retry-once-then-fail policy.

======  =====================================  ==========================================  =========================
Kind    Armed by                               Disarmed by                                 Default / ``stage``
======  =====================================  ==========================================  =========================
stt     ``VADUserStoppedSpeakingFrame`` after  ``TranscriptionFrame`` (final text);        5 s, ``no_final_transcript``
        >= 1 s of speech and no transcript     new speech; interruption
        (interim or final) since the speech
        started
llm     ``LLMContextFrame`` pushed to an LLM   ``LLMTextFrame`` (first token), thought     8 s, ``no_first_token``
        service                                text, a function call, end of response
tts     ``TTSStartedFrame`` from a TTS         ``TTSAudioRawFrame`` (first audio),         4 s, ``no_first_audio``
        service (text was sent to the          ``TTSStoppedFrame``, interruption
        provider)
======  =====================================  ==========================================  =========================

Why these frames (and not others):

* The STT timer starts at the raw VAD stop, not at ``UserStoppedSpeakingFrame``: with the
  speech-timeout and smart-turn strategies the turn only ends *after* the transcript
  arrived, so a hung STT would never produce that frame. Silence must never fire: the timer
  is armed only when the VAD heard at least :data:`STT_MIN_SPEECH_SECS` of speech (a click
  or breath is shorter) and the STT produced no text at all, not even an interim, since the
  speech started (a final that arrived while the user was still talking means the STT is
  alive and the turn is covered). A long cough or background noise that the STT rightly
  ignores can still look like a stall; that is why the policy tolerates the first
  ``timeout`` of a window (``retryable`` True) and only fails the call on the second.
* ``LLMFullResponseStartFrame`` is not a disarm: Pipecat pushes it when the request is
  *sent*, before the first token. An empty response or one that is only a function call is
  not a stall, so ``LLMFullResponseEndFrame`` / function-call frames disarm too.
* ``TTSStartedFrame`` marks "text handed to the provider" (a sentence is only sent once it
  is complete, so merely receiving text would false-fire on a slow LLM).
* Interruptions, cancel and end frames disarm everything: the work was dropped on purpose.

Limits: one timer per kind, so with several lanes/TTS services an answer from any lane
satisfies the timer (a hang on one lane can be masked while another is active). The
realtime engine has no STT/LLM/TTS service and is not watched (its socket has its own
failure path). Streaming translators (no Pipecat LLM service) are covered by
:func:`guard_first_token` instead.

Constants are overridable per instance; ``clock`` and ``sleep`` are injectable for tests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FunctionCallInProgressFrame,
    FunctionCallsStartedFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    LLMThoughtTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from .provider_failures import provider_kind, report_provider_failure

_logger = logging.getLogger(__name__)

STT_STALL_SECS = 5.0  # user stopped speaking -> no final transcription
LLM_STALL_SECS = 8.0  # context pushed to the LLM -> no first text token
TTS_STALL_SECS = 4.0  # text sent to the TTS -> no first audio
STT_MIN_SPEECH_SECS = 1.0  # shorter VAD "speech" (clicks, breaths) never arms the STT timer
TRANSLATION_STALL_SECS = LLM_STALL_SECS  # translator request -> first token (guard_first_token)

STAGES = {"stt": "no_final_transcript", "llm": "no_first_token", "tts": "no_first_audio"}


class StallWatchdog(BaseObserver):
    """Report ``timeout`` when an STT/LLM/TTS hand-off takes longer than its threshold."""

    def __init__(
        self,
        session: Any,
        providers: Mapping[str, str] | None = None,
        *,
        stt_secs: float = STT_STALL_SECS,
        llm_secs: float = LLM_STALL_SECS,
        tts_secs: float = TTS_STALL_SECS,
        min_speech_secs: float = STT_MIN_SPEECH_SECS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        report: Callable[..., str] = report_provider_failure,
    ) -> None:
        super().__init__()
        self._session = session
        self._providers = dict(providers or {})
        self._thresholds = {"stt": stt_secs, "llm": llm_secs, "tts": tts_secs}
        self._min_speech = min_speech_secs
        self._clock = clock
        self._sleep = sleep
        self._report = report
        self._timers: dict[str, asyncio.Task] = {}
        self._speech_started_at: float | None = None
        self._transcript_seen = False

    @property
    def armed(self) -> frozenset[str]:
        """The kinds with a running timer."""

        return frozenset(kind for kind, task in self._timers.items() if not task.done())

    # -- observer ------------------------------------------------------------------

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if isinstance(frame, LLMContextFrame):
            # Every hop sees this frame; only the push *into* the LLM service starts the clock.
            if provider_kind(data.destination) == "llm":
                self._arm("llm")
            return
        if isinstance(frame, TTSStartedFrame):
            if provider_kind(data.source) == "tts":
                self._arm("tts")
            return
        if not data.first_push:
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._disarm("stt")
            if self._speech_started_at is None:  # broadcast: the same event arrives twice
                self._speech_started_at = self._clock() - frame.start_secs
                self._transcript_seen = False
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            started, self._speech_started_at = self._speech_started_at, None
            if started is None:
                return
            speech = (self._clock() - frame.stop_secs) - started
            if speech >= self._min_speech and not self._transcript_seen:
                self._arm("stt")
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            self._transcript_seen = True
            if isinstance(frame, TranscriptionFrame):
                self._disarm("stt")
        elif isinstance(
            frame,
            (
                LLMTextFrame,
                LLMThoughtTextFrame,
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                LLMFullResponseEndFrame,
            ),
        ):
            self._disarm("llm")
        elif isinstance(frame, (TTSAudioRawFrame, TTSStoppedFrame)):
            self._disarm("tts")
        elif isinstance(frame, InterruptionFrame):
            self._speech_started_at = None
            for kind in ("stt", "llm", "tts"):
                self._disarm(kind)
        elif isinstance(frame, (CancelFrame, EndFrame)):
            self.cancel_all()
        elif isinstance(frame, ErrorFrame):
            # The error already travels the provider-failure path; do not double count it.
            for kind in ("llm", "tts"):
                if provider_kind(data.source) == kind:
                    self._disarm(kind)

    async def cleanup(self) -> None:
        self.cancel_all()
        await super().cleanup()

    # -- timers ----------------------------------------------------------------------

    def cancel_all(self) -> None:
        for kind in list(self._timers):
            self._disarm(kind)

    def _arm(self, kind: str) -> None:
        task = self._timers.get(kind)
        if task is not None and not task.done():
            return  # keep the earliest deadline
        self._timers[kind] = asyncio.get_running_loop().create_task(self._expire(kind))

    def _disarm(self, kind: str) -> None:
        task = self._timers.pop(kind, None)
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _expire(self, kind: str) -> None:
        await self._sleep(self._thresholds[kind])
        self._timers.pop(kind, None)
        stage = STAGES[kind]
        _logger.warning("%s stalled (%s, > %.1fs)", kind, stage, self._thresholds[kind])
        outcome = self._report(
            self._session,
            kind,
            self._providers.get(kind, kind),
            "timeout",
            stage=stage,
            clock=self._clock,
        )
        if outcome in ("failed", "fallback"):
            self.cancel_all()


async def guard_first_token(
    stream: Any, secs: float = TRANSLATION_STALL_SECS
) -> AsyncIterator[str]:
    """``stream`` unchanged, except that no first item within ``secs`` raises ``TimeoutError``.

    The translators call OpenAI directly (no Pipecat LLM service for the observer to watch).
    ``TimeoutError`` classifies as ``timeout`` in :func:`runtime.provider_failures.classify_error`
    and so takes the normal translator failure path.
    """

    iterator = stream.__aiter__()
    first = True
    try:
        while True:
            try:
                if first:
                    delta = await asyncio.wait_for(iterator.__anext__(), secs)
                else:
                    delta = await iterator.__anext__()
            except StopAsyncIteration:
                return
            first = False
            yield delta
    finally:
        closer = getattr(iterator, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # pragma: no cover - cleanup only
                _logger.debug("Closing guarded stream failed", exc_info=True)
