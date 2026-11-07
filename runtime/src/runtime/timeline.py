"""First-audio latency instrumentation (RM-003): one timeline per utterance.

Every interpreter engine (cascaded single or two-person routed, realtime) and the
agent + translator pipeline record the same six moments for each utterance and emit them
as ONE ``LatencyMeasured`` event with ``metric == "utterance_timeline"``::

    T0 speaker speech start   VAD start, backdated by the VAD confirmation window
    T1 VAD detection          ``vad``
    T2 first transcript       ``first_transcript``   interim or final, whichever is first
    T3 translation started    ``translation_start``  first translation request of the turn
    T4 first audible audio    ``first_audio``        first *voiced* chunk leaving Runtime
    T5 playback stable        ``stable``             see below

PR-20 splits T3 -> T4 (cascaded and agent translation; omitted when they did not happen)
with three extra marks, recorded between T3 and T4::

    translation_first_token   first non-empty LLM delta       T3 -> this: LLM time to first token
    translation_first_chunk   first chunk cut for TTS         -> this: chunking wait
    tts_request               first text sent to the TTS      this -> T4: TTS time to first byte
                              (just before the speak frame leaves the processor)

The first chunk and the TTS request are normally microseconds apart; ``tts_request`` is
kept separate because a language/voice change reaches the TTS between them.

Payload: ``{"metric": "utterance_timeline", "value_ms": T4 - T0, "engine": ...,
"speaker"?, "listener"?, "marks_ms": {"vad": T1 - T0, ...}}``. A mark that did not happen
is omitted. ``engine`` is ``cascaded``, ``realtime`` or ``agent_translation``. An
utterance that never produced audible audio (nobody to hear it, provider failure) emits
nothing.

Audible audio
-------------
T4 is when the first voiced chunk starts playing on the playout clock described below: the
chunk's arrival, or later when audio sent before it (silence padding, the tail of the
previous utterance) is still ahead of it. Silence is ignored the way ``realtime_translate``
always did: a chunk counts only when its PCM16 peak reaches :data:`VOICED_PEAK_THRESHOLD`.
The same test now applies to TTS audio.

Stable playback (T5)
--------------------
Chunks are laid on a simulated playout clock (a chunk starts when it arrives or when the
previous chunk ends, whichever is later), so a burst of TTS audio is not mistaken for
500 ms of played sound. Playback is stable once audio has continued for
:data:`STABLE_SECONDS` without a gap longer than :data:`MAX_GAP_SECONDS` between voiced
chunks (a longer gap restarts the count). T5 is then the playout time that 500 ms run
completes. An utterance whose audio ends before that gets T5 = end of its last voiced
audio ("end of utterance").

Attribution
-----------
Frames after the TTS do not say which utterance they belong to, but the translator speaks
utterances one after another, so audio goes to the oldest utterance that has asked for
audio and is not finished (:meth:`TimelineTracker.enqueue_audio`). An utterance finishes
when all the chunks it handed to TTS reported ``TTSStoppedFrame`` (or, when that count is
unknown, when the output went idle), or when the owning engine calls
:meth:`TimelineTracker.finish`.

Telemetry must never break the media pipeline: public tracker methods swallow errors.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
from array import array
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .turn_ids import turn_state

_logger = logging.getLogger(__name__)

METRIC = "utterance_timeline"
DIAG_ENV = "RUNTIME_TIMELINE_DIAG"


def diag_enabled() -> bool:
    return os.getenv(DIAG_ENV) == "1"


def diag(message: str) -> None:
    """Diagnostic line on stderr, only when ``RUNTIME_TIMELINE_DIAG=1`` (benchmarks).

    Off by default: a diagnostic line can carry what was said (``partial_probe``), which is
    user speech and must not reach production logs.
    """

    if diag_enabled():
        sys.stderr.write(message + "\n")
        sys.stderr.flush()


def probe_line(
    key: str, t_partial: float | None, t_vad_stop: float, t_final: float, partial: str | None, final: str
) -> str:
    """One ``partial_probe`` line: the last partial before the VAD stop against the first final."""

    return "partial_probe " + json.dumps(
        {"key": key, "t_partial": t_partial, "t_vad_stop": t_vad_stop, "t_final": t_final, "partial": partial, "final": final},
        ensure_ascii=False,
    )
# The API streams audio continuously, silence included. Chunks whose peak stays below this
# PCM16 amplitude are not audible speech.
VOICED_PEAK_THRESHOLD = 500
STABLE_SECONDS = 0.5
MAX_GAP_SECONDS = 0.2
# A finished-speaking utterance that never got audio is dropped after this long, so a TTS
# failure cannot make later audio attach to it.
STALE_SECONDS = 15.0
MAX_QUEUED = 8

MARK_ORDER = (
    "vad",
    "first_transcript",
    "audio_in_last_voiced",  # diagnostics only: last voiced input frame seen before the VAD stop
    # Fine-grained marks for the time between the end of speech and the translation. They are
    # observations on the server's monotonic clock, in the order they happen:
    "speech_end",  # estimated: vad_stop minus the VAD's stop_secs (the last voiced audio)
    "vad_stop",  # the VAD reported the silence
    "final_transcript",  # the first finalized transcript of the turn reached the translator
    "turn_end",  # the turn-stop strategy ended the turn (pause timer and transcript both done)
    "translation_start",
    "translation_first_token",
    "translation_first_chunk",
    "tts_request",
    "tts_first_frame",  # first audio frame from the TTS (voiced or not) reached the output side
    "first_audio",
    "livekit_submit",  # first voiced output chunk handed to the LiveKit audio source
    "livekit_submitted",  # ... and capture_frame returned (queue back-pressure shows here)
    "stable",
)


def is_voiced(audio: bytes) -> bool:
    """Return whether a PCM16 chunk contains sound above the silence threshold."""

    samples = array("h")
    samples.frombytes(audio[: len(audio) - len(audio) % 2])
    return any(abs(sample) >= VOICED_PEAK_THRESHOLD for sample in samples)


@dataclass
class UtteranceTimeline:
    """The moments of one utterance; times are seconds on the tracker's clock."""

    engine: str
    t0: float
    speaker: str | None = None
    listener: str | None = None
    turn_id: int | None = None
    marks: dict[str, float] = field(default_factory=dict)
    turn_stopped: bool = False
    # Audio bookkeeping. ``chunks_pushed`` is None when the number of TTS contexts is
    # unknown (the output going idle then ends the utterance).
    chunks_pushed: int | None = 0
    chunks_done: int = 0
    speech_complete: bool = False
    completed_at: float | None = None
    queued: bool = False
    finished: bool = False
    emitted: bool = False
    run_start: float | None = None
    last_voiced_end: float | None = None


def _safe(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self: "TimelineTracker", *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except Exception:  # pragma: no cover - telemetry must not break the pipeline
            _logger.debug("Timeline %s failed", method.__name__, exc_info=True)
            return None

    return wrapper


class TimelineTracker:
    """Collects the marks of every utterance of one pipeline and emits the timelines.

    ``key`` identifies a speaker (the LiveKit identity in two-person sessions, ``""``
    otherwise); each key has at most one *pending* utterance: the one being spoken or
    waiting for its translation.
    """

    def __init__(
        self,
        session: Any | None,
        engine: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        stable_seconds: float = STABLE_SECONDS,
        max_gap_seconds: float = MAX_GAP_SECONDS,
    ) -> None:
        self._session = session
        self.engine = engine
        self._clock = clock
        self._stable = stable_seconds
        self._max_gap = max_gap_seconds
        self._pending: dict[str, UtteranceTimeline] = {}
        self._early_transcript: dict[str, deque[float]] = {}
        self._queue: deque[UtteranceTimeline] = deque()
        self._playhead: float | None = None

    def now(self) -> float:
        return self._clock()

    # -- speech and transcripts ------------------------------------------------

    @_safe
    def speech_started(self, key: str = "", start_secs: float = 0.0) -> UtteranceTimeline:
        """VAD detected speech: T1 is now, T0 is ``start_secs`` earlier.

        A pause inside one turn restarts the VAD without starting a new utterance, so the
        pending utterance is kept; one whose turn already ended is replaced.
        """

        pending = self._pending.get(key)
        if pending is not None:
            if not pending.turn_stopped:
                return pending
            self.finish(pending)
        now = self._clock()
        t0 = now - max(0.0, start_secs or 0.0)
        timeline = UtteranceTimeline(
            engine=self.engine,
            t0=t0,
            speaker=key or None,
            marks={"vad": now},
            turn_id=self._start_turn(key),
        )
        early = [at for at in self._early_transcript.pop(key, ()) if at >= t0]
        if early:
            timeline.marks["first_transcript"] = early[0]
        self._pending[key] = timeline
        return timeline

    @_safe
    def transcript(self, key: str = "") -> None:
        """An interim or final transcript arrived: T2 of the pending utterance."""

        timeline = self._pending.get(key)
        if timeline is None:
            # The STT can beat the VAD's confirmation window; remember it for T2.
            self._early_transcript.setdefault(key, deque(maxlen=4)).append(self._clock())
        else:
            self.mark(timeline, "first_transcript")

    @_safe
    def voiced_input(self, key: str = "") -> None:
        """A voiced input frame reached the translator (diagnostics): remember the latest."""

        timeline = self._pending.get(key)
        if timeline is not None and not timeline.emitted and "vad_stop" not in timeline.marks:
            timeline.marks["audio_in_last_voiced"] = self._clock()

    @_safe
    def output_submitting(self, audio: bytes) -> UtteranceTimeline | None:
        """Just before a chunk goes to LiveKit: mark the first voiced one of the head utterance."""

        if not diag_enabled():
            return None
        timeline = self._head()
        if timeline is None or "livekit_submit" in timeline.marks or not is_voiced(audio):
            return None
        timeline.marks["livekit_submit"] = self._clock()
        return timeline

    @_safe
    def output_submitted(self, timeline: UtteranceTimeline | None) -> None:
        if timeline is not None:
            timeline.marks.setdefault("livekit_submitted", self._clock())

    @_safe
    def vad_stopped(self, key: str = "", stop_secs: float = 0.0) -> None:
        """The VAD reported the end of speech: ``vad_stop`` now, ``speech_end`` ``stop_secs`` earlier.

        Pipecat stamps its frames with the wall clock, which can jump; the
        marks are taken here, on the tracker's monotonic clock, and the frame's stamp is not used.
        """

        timeline = self._pending.get(key)
        if timeline is not None and not timeline.emitted:
            now = self._clock()
            timeline.marks["vad_stop"] = now
            timeline.marks["speech_end"] = now - max(0.0, stop_secs or 0.0)

    @_safe
    def final_transcript(self, key: str = "") -> None:
        """A finalized transcript arrived; only the first of the turn is recorded."""

        self.mark(self._pending.get(key), "final_transcript")

    @_safe
    def turn_stopped(self, key: str = "") -> None:
        timeline = self._pending.get(key)
        if timeline is not None:
            timeline.turn_stopped = True
            self.mark(timeline, "turn_end")
        if not key:
            state = turn_state(self._session)
            if state is not None:
                state.end()
        self._early_transcript.pop(key, None)

    @_safe
    def take(self, key: str = "") -> UtteranceTimeline | None:
        """Detach the pending utterance (its turn became a translation job)."""

        return self._pending.pop(key, None)

    def pending(self, key: str = "") -> UtteranceTimeline | None:
        return self._pending.get(key)

    @_safe
    def mark(self, timeline: UtteranceTimeline | None, name: str) -> None:
        """Record ``name`` now; the first occurrence wins."""

        if timeline is not None and not timeline.emitted and name not in timeline.marks:
            timeline.marks[name] = self._clock()

    # -- audio -----------------------------------------------------------------

    @_safe
    def enqueue_audio(
        self, timeline: UtteranceTimeline | None, listener: str | None = None
    ) -> None:
        """The utterance is about to receive translated audio (idempotent)."""

        if timeline is None or timeline.queued or timeline.finished:
            return
        if listener:
            timeline.listener = listener
        now = self._clock()
        self._queue = deque(
            t
            for t in self._queue
            if not (
                t.speech_complete
                and "first_audio" not in t.marks
                and now - (t.completed_at or now) > STALE_SECONDS
            )
        )
        while len(self._queue) >= MAX_QUEUED:
            self._queue.popleft()
        timeline.queued = True
        self._queue.append(timeline)

    @_safe
    def chunk_pushed(self, timeline: UtteranceTimeline | None) -> None:
        """One more piece of speech was handed to TTS (one ``TTSStoppedFrame`` expected)."""

        if timeline is not None and timeline.chunks_pushed is not None:
            timeline.chunks_pushed += 1

    @_safe
    def speech_done(
        self, timeline: UtteranceTimeline | None, *, count_known: bool = True
    ) -> None:
        """The engine handed all speech of the utterance to TTS."""

        if timeline is None or timeline.finished:
            return
        timeline.speech_complete = True
        timeline.completed_at = self._clock()
        if not count_known:
            timeline.chunks_pushed = None
        if not timeline.queued:
            timeline.finished = True  # nothing was spoken: no timeline
            return
        self._maybe_finish_by_count(timeline)

    @_safe
    def abandon(self, timeline: UtteranceTimeline | None) -> bool:
        """Take back an utterance whose speech was cut off before any audio was heard.

        It leaves the audio queue (so it cannot capture the next utterance's audio) and
        can be queued again for the next reply to the same speech. Returns False when it
        already has audio (or is finished): then it must be finished instead.
        """

        if timeline is None or timeline.finished or "first_audio" in timeline.marks:
            return False
        if timeline in self._queue:
            self._queue.remove(timeline)
        timeline.queued = False
        timeline.speech_complete = False
        timeline.chunks_pushed = 0
        timeline.chunks_done = 0
        return True

    @_safe
    def on_audio(self, audio: bytes, sample_rate: int, channels: int = 1) -> None:
        """A chunk of outgoing audio (voiced or not) reached the output."""

        if not audio or sample_rate <= 0:
            return
        now = self._clock()
        duration = len(audio) / 2 / max(1, channels) / sample_rate
        start = now if self._playhead is None else max(now, self._playhead)
        self._playhead = start + duration
        if diag_enabled():
            head = self._head()
            if head is not None and not head.emitted:
                head.marks.setdefault("tts_first_frame", now)
        if not is_voiced(audio):
            return
        timeline = self._head()
        if timeline is None:
            return
        end = start + duration
        if timeline.last_voiced_end is None:
            timeline.marks.setdefault("first_audio", start)
            timeline.run_start = start
        elif start - timeline.last_voiced_end > self._max_gap:
            timeline.run_start = start
        timeline.last_voiced_end = end
        if (
            "stable" not in timeline.marks
            and timeline.run_start is not None
            and end - timeline.run_start >= self._stable - 1e-6
        ):
            timeline.marks["stable"] = timeline.run_start + self._stable
            self._emit(timeline)

    @_safe
    def on_tts_stopped(self) -> None:
        timeline = self._head()
        if timeline is not None:
            timeline.chunks_done += 1
            self._maybe_finish_by_count(timeline)

    @_safe
    def on_idle(self) -> None:
        """The output transport went quiet: ends an utterance whose chunk count is unknown."""

        timeline = self._head()
        if (
            timeline is not None
            and timeline.speech_complete
            and timeline.chunks_pushed is None
            and "first_audio" in timeline.marks
        ):
            self.finish(timeline)

    @_safe
    def finish(self, timeline: UtteranceTimeline | None) -> None:
        """End of utterance: emit it (T5 = end of its audio if not yet stable) and forget it."""

        if timeline is None or timeline.finished:
            return
        timeline.finished = True
        if self._pending.get(timeline.speaker or "") is timeline:
            del self._pending[timeline.speaker or ""]
        if timeline in self._queue:
            self._queue.remove(timeline)
        if timeline.last_voiced_end is not None and "stable" not in timeline.marks:
            timeline.marks["stable"] = timeline.last_voiced_end
        self._emit(timeline)

    # -- internals -------------------------------------------------------------

    def _start_turn(self, key: str) -> int | None:
        """The ``turn_id`` of a new utterance (ST-02a), shared with the session's other events.

        One speaker (``key == ""``) joins the session's turn, which the agent pipeline's
        ``UserStartedSpeakingFrame`` may have opened already (``begin`` is idempotent); speakers in a two-person
        session talk over each other, so each utterance gets its own id.
        """

        state = turn_state(self._session)
        if state is None:
            return None
        if key:
            return state.new_turn()
        return state.begin()  # ``turn_stopped`` closed the previous turn

    def _head(self) -> UtteranceTimeline | None:
        while self._queue and self._queue[0].finished:
            self._queue.popleft()
        return self._queue[0] if self._queue else None

    def _maybe_finish_by_count(self, timeline: UtteranceTimeline) -> None:
        if (
            timeline.speech_complete
            and timeline.chunks_pushed is not None
            and timeline.chunks_done >= timeline.chunks_pushed
        ):
            self.finish(timeline)

    def _emit(self, timeline: UtteranceTimeline) -> None:
        if timeline.emitted:
            return
        first_audio = timeline.marks.get("first_audio")
        if first_audio is None:
            return
        timeline.emitted = True
        marks_ms = {
            name: round(max(0.0, timeline.marks[name] - timeline.t0) * 1000, 1)
            for name in MARK_ORDER
            if name in timeline.marks
        }
        payload: dict[str, Any] = {
            "metric": METRIC,
            "value_ms": marks_ms["first_audio"],
            "engine": timeline.engine,
        }
        if timeline.turn_id is not None:
            payload["turn_id"] = timeline.turn_id
        if timeline.speaker:
            payload["speaker"] = timeline.speaker
        if timeline.listener:
            payload["listener"] = timeline.listener
        payload["marks_ms"] = marks_ms
        # The anchor of the marks on this host's monotonic clock (CLOCK_MONOTONIC is shared by
        # all processes of one Linux host), so a client on the same host can line its own
        # monotonic timestamps up with them. Meaningless across machines.
        payload["t0_monotonic_s"] = round(timeline.t0, 6)
        diag(
            f"utterance_timeline turn={timeline.turn_id} engine={timeline.engine} "
            f"t0_monotonic_s={timeline.t0:.6f} marks_ms={json.dumps(marks_ms)}"
        )
        session = self._session
        if session is None or not getattr(session, "event_client", None):
            return
        try:
            session.enqueue_event(session.event("LatencyMeasured", payload))
        except RuntimeError:
            _logger.warning("Dropped %s event for session %s", METRIC, session.session_id)
