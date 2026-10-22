"""RM-003: utterance timelines (T0..T5) with fake clocks."""

import asyncio
import struct
from dataclasses import dataclass
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test

from src.runtime.frame_events import RuntimeEventProcessor
from src.runtime.interpreter import InterpreterSettings, TranslationProcessor
from src.runtime.participants import ParticipantDirectory
from src.runtime.timeline import (
    METRIC,
    MAX_GAP_SECONDS,
    STABLE_SECONDS,
    TimelineTracker,
    is_voiced,
)

RATE = 16000


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSession:
    def __init__(self) -> None:
        self.event_client = object()
        self.session_id = uuid4()
        self.events: list[tuple[str, dict]] = []
        self.state = None

    def begin_listening(self) -> None:
        pass

    def event(self, event_type, payload=None):
        return (event_type, payload or {})

    def enqueue_event(self, event) -> None:
        self.events.append(event)

    def timelines(self) -> list[dict]:
        return [
            payload
            for kind, payload in self.events
            if kind == "LatencyMeasured" and payload.get("metric") == METRIC
        ]


def pcm(ms: int, amplitude: int = 4000, rate: int = RATE) -> bytes:
    return struct.pack("<h", amplitude) * (rate * ms // 1000)


def make(engine: str = "cascaded"):
    clock, session = FakeClock(), FakeSession()
    return TimelineTracker(session, engine, clock=clock), clock, session


def speak_utterance(tracker, clock, *, listener=None, chunks=1):
    """Speech at t=100 (T0 99.8), transcript +0.1, translation +0.05, speech handed to TTS."""

    timeline = tracker.speech_started("", 0.2)
    clock.advance(0.1)
    tracker.transcript("")
    clock.advance(0.05)
    taken = tracker.take("")
    assert taken is timeline
    tracker.mark(timeline, "translation_start")
    tracker.enqueue_audio(timeline, listener)
    for _ in range(chunks):
        tracker.chunk_pushed(timeline)
    tracker.speech_done(timeline)
    return timeline


# -- voiced audio ------------------------------------------------------------------


def test_voiced_chunk_detection_is_shared_with_realtime() -> None:
    from src.runtime.realtime_translate import VOICED_PEAK_THRESHOLD, is_voiced as realtime

    assert realtime is is_voiced and VOICED_PEAK_THRESHOLD == 500
    assert not is_voiced(pcm(20, amplitude=0))
    assert not is_voiced(pcm(20, amplitude=499))
    assert is_voiced(pcm(20, amplitude=500))


# -- marks and payload ---------------------------------------------------------------


def test_one_event_with_every_mark_when_playback_becomes_stable() -> None:
    tracker, clock, session = make()
    speak_utterance(tracker, clock)  # now = 100.15
    clock.advance(0.3)  # first audio at 100.45
    tracker.on_audio(pcm(600), RATE)  # 0.6 s of voiced audio in one chunk

    [event] = session.timelines()
    assert event["metric"] == "utterance_timeline" and event["engine"] == "cascaded"
    marks = event["marks_ms"]
    assert marks == pytest.approx(
        {"vad": 200, "first_transcript": 300, "translation_start": 350, "first_audio": 650, "stable": 1150},
        abs=0.2,
    )
    assert event["value_ms"] == pytest.approx(650, abs=0.2) == pytest.approx(marks["first_audio"])
    assert list(marks) == ["vad", "first_transcript", "translation_start", "first_audio", "stable"]
    assert "speaker" not in event and "listener" not in event
    # Later audio, the end of the speech and the idle signal never emit a second event.
    tracker.on_tts_stopped()
    tracker.on_idle()
    tracker.on_audio(pcm(200), RATE)
    assert len(session.timelines()) == 1


def test_missing_marks_are_omitted() -> None:
    tracker, clock, session = make()
    timeline = tracker.speech_started("", 0.0)
    tracker.take("")
    tracker.enqueue_audio(timeline)
    tracker.speech_done(timeline, count_known=False)
    clock.advance(0.4)
    tracker.on_audio(pcm(100), RATE)
    tracker.on_idle()
    [event] = session.timelines()
    assert set(event["marks_ms"]) == {"vad", "first_audio", "stable"}
    assert event["marks_ms"]["vad"] == 0


def test_speaker_and_listener_identities_are_reported() -> None:
    tracker, clock, session = make()
    timeline = tracker.speech_started("alice", 0.2)
    tracker.take("alice")
    tracker.enqueue_audio(timeline, "bob")
    tracker.chunk_pushed(timeline)
    tracker.speech_done(timeline)
    tracker.on_audio(pcm(100), RATE)
    tracker.on_tts_stopped()
    [event] = session.timelines()
    assert (event["speaker"], event["listener"]) == ("alice", "bob")


def test_silent_audio_is_not_first_audio() -> None:
    tracker, clock, session = make()
    speak_utterance(tracker, clock)
    clock.advance(0.2)
    tracker.on_audio(pcm(100, amplitude=0), RATE)  # TTS padding
    clock.advance(0.3)
    tracker.on_audio(pcm(100), RATE)  # first audible sound at 100.65
    tracker.on_tts_stopped()
    [event] = session.timelines()
    assert event["marks_ms"]["first_audio"] == pytest.approx(850, abs=0.2)


def test_utterance_without_audible_audio_emits_nothing() -> None:
    tracker, clock, session = make()
    timeline = speak_utterance(tracker, clock)
    tracker.on_audio(pcm(100, amplitude=0), RATE)
    tracker.on_tts_stopped()
    assert timeline.finished and session.timelines() == []
    # Nothing was ever spoken (nobody to hear it): no timeline either.
    silent = tracker.speech_started("", 0.2)
    tracker.take("")
    tracker.speech_done(silent)
    tracker.on_audio(pcm(100), RATE)
    assert session.timelines() == []


# -- T5: playback stable ------------------------------------------------------------


def test_stable_after_500ms_of_continuous_audio_not_a_burst_of_arrivals() -> None:
    tracker, clock, session = make()
    speak_utterance(tracker, clock, chunks=2)
    clock.advance(0.3)
    for _ in range(4):  # 4 x 100 ms arrive in the same instant: 400 ms of playout
        tracker.on_audio(pcm(100), RATE)
    assert session.timelines() == []
    tracker.on_audio(pcm(100), RATE)  # the fifth completes 500 ms of playout
    [event] = session.timelines()
    assert event["marks_ms"]["stable"] - event["marks_ms"]["first_audio"] == pytest.approx(500, abs=0.2)
    assert STABLE_SECONDS == 0.5


def test_a_gap_over_200ms_restarts_the_stable_count() -> None:
    tracker, clock, session = make()
    speak_utterance(tracker, clock, chunks=2)
    clock.advance(0.3)  # first audio at 100.45
    tracker.on_audio(pcm(300), RATE)  # plays 100.45 - 100.75
    clock.advance(0.3 + 0.25)  # silence 100.75 - 101.0, next chunk arrives at 101.0
    tracker.on_audio(pcm(300), RATE)  # gap 0.25 s > 0.2: a new run begins at 101.0
    assert session.timelines() == []
    tracker.on_audio(pcm(300), RATE)  # run is now 0.6 s long
    [event] = session.timelines()
    assert MAX_GAP_SECONDS == 0.2
    # Stable counted from the restarted run: 101.0 + 0.5 s on the playout clock.
    assert event["marks_ms"]["stable"] == pytest.approx((101.5 - 99.8) * 1000, abs=0.2)
    assert event["marks_ms"]["first_audio"] == pytest.approx(650, abs=0.2)


def test_a_gap_up_to_200ms_does_not_restart_it() -> None:
    tracker, clock, session = make()
    speak_utterance(tracker, clock, chunks=2)
    clock.advance(0.3)
    tracker.on_audio(pcm(300), RATE)  # 100.45 - 100.75
    clock.advance(0.3 + 0.15)  # next chunk at 100.9: a 150 ms gap
    tracker.on_audio(pcm(300), RATE)
    [event] = session.timelines()
    assert event["marks_ms"]["stable"] == pytest.approx((100.45 + 0.5 - 99.8) * 1000, abs=0.2)


def test_short_utterance_is_stable_at_the_end_of_its_audio() -> None:
    tracker, clock, session = make()
    speak_utterance(tracker, clock)
    clock.advance(0.3)
    tracker.on_audio(pcm(200), RATE)
    assert session.timelines() == []  # still waiting for 500 ms or the end
    tracker.on_tts_stopped()  # all chunks played: end of utterance
    [event] = session.timelines()
    assert event["marks_ms"]["stable"] == pytest.approx((100.45 + 0.2 - 99.8) * 1000, abs=0.2)


# -- attribution ---------------------------------------------------------------------


def test_audio_goes_to_utterances_in_order() -> None:
    tracker, clock, session = make()
    first = tracker.speech_started("", 0.2)
    tracker.take("")
    tracker.enqueue_audio(first)
    tracker.chunk_pushed(first)
    tracker.speech_done(first)
    clock.advance(1.0)
    second = tracker.speech_started("", 0.2)  # second speaker turn while the first plays
    tracker.take("")
    tracker.enqueue_audio(second)
    tracker.chunk_pushed(second)
    tracker.speech_done(second)

    clock.advance(0.2)
    tracker.on_audio(pcm(100), RATE)  # first
    tracker.on_tts_stopped()  # first is done
    clock.advance(0.5)
    tracker.on_audio(pcm(100), RATE)  # second
    tracker.on_tts_stopped()
    one, two = session.timelines()
    assert one["marks_ms"]["first_audio"] == pytest.approx(1400, abs=0.2)  # 101.2 - 99.8
    assert two["marks_ms"]["first_audio"] == pytest.approx(900, abs=0.2)  # 101.7 - 100.8


def test_unknown_chunk_count_ends_when_the_output_goes_idle() -> None:
    tracker, clock, session = make()
    timeline = tracker.speech_started("", 0.2)
    tracker.take("")
    tracker.enqueue_audio(timeline)
    tracker.speech_done(timeline, count_known=False)
    tracker.on_audio(pcm(100), RATE)
    tracker.on_tts_stopped()  # a TTS context ended; the count is unknown
    assert session.timelines() == []
    tracker.on_idle()
    assert len(session.timelines()) == 1


def test_idle_before_audio_does_not_end_a_waiting_utterance() -> None:
    tracker, clock, session = make()
    timeline = tracker.speech_started("", 0.2)
    tracker.take("")
    tracker.enqueue_audio(timeline)
    tracker.speech_done(timeline, count_known=False)
    tracker.on_idle()  # the previous utterance's tail
    clock.advance(0.2)
    tracker.on_audio(pcm(100), RATE)
    tracker.on_idle()
    assert len(session.timelines()) == 1


def test_a_dead_utterance_does_not_steal_later_audio() -> None:
    tracker, clock, session = make()
    dead = tracker.speech_started("", 0.2)
    tracker.take("")
    tracker.enqueue_audio(dead)
    tracker.speech_done(dead, count_known=False)  # TTS failed: no audio ever
    clock.advance(60)
    alive = tracker.speech_started("", 0.2)
    tracker.take("")
    tracker.enqueue_audio(alive)  # purges the stale one
    tracker.chunk_pushed(alive)
    tracker.speech_done(alive)
    tracker.on_audio(pcm(100), RATE)
    tracker.on_tts_stopped()
    assert len(session.timelines()) == 1


# -- speech and transcript bookkeeping ----------------------------------------------------


def test_a_pause_inside_one_turn_keeps_the_first_speech_start() -> None:
    tracker, clock, _ = make()
    first = tracker.speech_started("", 0.2)
    clock.advance(0.4)
    assert tracker.speech_started("", 0.2) is first  # same turn, VAD restarted
    tracker.turn_stopped("")
    clock.advance(1.0)
    second = tracker.speech_started("", 0.2)  # the turn had ended: a new utterance
    assert second is not first and second.t0 == pytest.approx(101.2)


def test_first_transcript_is_the_earliest_interim_or_final() -> None:
    tracker, clock, _ = make()
    timeline = tracker.speech_started("", 0.2)
    clock.advance(0.1)
    tracker.transcript("")  # interim
    clock.advance(0.5)
    tracker.transcript("")  # final
    assert timeline.marks["first_transcript"] == pytest.approx(100.1)


def test_a_transcript_that_beats_the_vad_confirmation_window_is_kept() -> None:
    tracker, clock, _ = make()
    clock.advance(0.1)
    tracker.transcript("")  # at 100.1, before the VAD reports speech
    clock.advance(0.1)
    timeline = tracker.speech_started("", 0.2)  # at 100.2 => T0 100.0
    assert timeline.marks["first_transcript"] == pytest.approx(100.1)


def test_a_transcript_older_than_the_speech_start_belongs_to_a_previous_utterance() -> None:
    tracker, clock, _ = make()
    tracker.transcript("")  # a late segment of the previous turn, at 100.0
    clock.advance(5)
    timeline = tracker.speech_started("", 0.2)  # T0 = 104.8
    assert "first_transcript" not in timeline.marks


def test_speakers_are_tracked_separately() -> None:
    tracker, clock, _ = make()
    alice = tracker.speech_started("alice", 0.2)
    clock.advance(0.1)
    bob = tracker.speech_started("bob", 0.2)
    tracker.transcript("bob")
    assert "first_transcript" in bob.marks and "first_transcript" not in alice.marks
    assert tracker.take("alice") is alice and tracker.take("alice") is None
    assert tracker.pending("bob") is bob


def test_telemetry_without_a_session_or_event_client_is_silent() -> None:
    for session in (None, type("S", (), {"event_client": None})()):
        tracker = TimelineTracker(session, "realtime", clock=FakeClock())
        timeline = tracker.speech_started("", 0.1)
        tracker.take("")
        tracker.enqueue_audio(timeline)
        tracker.chunk_pushed(timeline)
        tracker.speech_done(timeline)
        tracker.on_audio(pcm(700), RATE)
        assert timeline.emitted


# -- TranslationProcessor wiring ---------------------------------------------------------------


@dataclass
class Tick(SystemFrame):
    seconds: float = 0.0


class ClockDriver(FrameProcessor):
    """Advances the fake clock when a Tick passes (system frames keep their order)."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self._clock = clock

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, Tick):
            self._clock.advance(frame.seconds)
        await self.push_frame(frame, direction)


def drive(clock, processor, frames):
    asyncio.run(
        run_test(Pipeline([ClockDriver(clock), processor]), frames_to_send=frames, expected_down_frames=None)
    )


def translator(tracker, session, *, directory=None, **kwargs):
    async def translate(messages):
        return "T:" + messages[-1]["content"]

    return TranslationProcessor(
        settings=InterpreterSettings(
            languages=directory.languages if directory else ("en", "ru"), direction="auto"
        ),
        translate=translate,
        model="test-model",
        session=session,
        initial_tts_language="ru",
        participants=directory,
        timeline=tracker,
        **kwargs,
    )


def settle(seconds=0.05):
    return SleepFrame(sleep=seconds)


def test_translation_processor_records_speech_transcript_and_translation_start() -> None:
    tracker, clock, session = make()
    processor = translator(tracker, session)
    drive(
        clock,
        processor,
        [
            VADUserStartedSpeakingFrame(start_secs=0.2),
            UserStartedSpeakingFrame(),
            settle(),
            Tick(seconds=0.1),
            InterimTranscriptionFrame("Hel", "speaker", "2026-01-01T00:00:00Z"),
            settle(),
            Tick(seconds=0.2),
            TranscriptionFrame("Hello there", "speaker", "2026-01-01T00:00:00Z", None, finalized=True),
            settle(),
            Tick(seconds=0.1),
            UserStoppedSpeakingFrame(),
            settle(0.1),
        ],
    )
    # The utterance waits for audio in the tracker's queue.
    [queued] = list(tracker._queue)
    assert queued.t0 == pytest.approx(99.8)
    assert queued.marks["vad"] == pytest.approx(100.0)
    assert queued.marks["first_transcript"] == pytest.approx(100.1)  # the interim
    assert queued.marks["translation_start"] == pytest.approx(100.4)
    assert queued.chunks_pushed == 1 and queued.speech_complete
    assert tracker.pending("") is None

    clock.advance(0.25)  # TTS audio reaches the observer after the speech
    tracker.on_audio(pcm(600), RATE)
    tracker.on_tts_stopped()
    [event] = session.timelines()
    assert event["marks_ms"]["first_audio"] == pytest.approx(850, abs=0.2)
    assert event["engine"] == "cascaded"


def test_two_participant_timelines_name_speaker_and_listener() -> None:
    tracker, clock, session = make()
    directory = ParticipantDirectory.from_mapping(
        ("en", "ko"),
        {"alice": {"language": "en"}, "bob": {"language": "ko"}},
    )
    processor = translator(tracker, session, directory=directory)
    vad = VADUserStartedSpeakingFrame(start_secs=0.2)
    vad.metadata["user_id"] = "alice"
    started = UserStartedSpeakingFrame()
    started.metadata["user_id"] = "alice"
    stopped = UserStoppedSpeakingFrame()
    stopped.metadata["user_id"] = "alice"
    untagged = VADUserStartedSpeakingFrame(start_secs=0.2)  # no lane identity: ignored
    drive(
        clock,
        processor,
        [
            untagged,
            vad,
            started,
            settle(),
            TranscriptionFrame("Hello", "alice", "2026-01-01T00:00:00Z", None, finalized=True),
            settle(),
            stopped,
            settle(0.1),
        ],
    )
    [queued] = list(tracker._queue)
    assert (queued.speaker, queued.listener) == ("alice", "bob")
    clock.advance(0.2)
    tracker.on_audio(pcm(100), RATE)
    tracker.on_tts_stopped()
    [event] = session.timelines()
    assert (event["speaker"], event["listener"]) == ("alice", "bob")


def test_utterance_without_a_vad_start_has_no_timeline() -> None:
    tracker, clock, session = make()
    drive(
        clock,
        translator(tracker, session),
        [
            UserStartedSpeakingFrame(),
            settle(),
            TranscriptionFrame("Hello", "speaker", "2026-01-01T00:00:00Z", None, finalized=True),
            settle(),
            UserStoppedSpeakingFrame(),
            settle(0.1),
        ],
    )
    assert list(tracker._queue) == []


# -- RuntimeEventProcessor feeding the tracker ------------------------------------------------------


def test_event_processor_feeds_tts_audio_to_the_timeline() -> None:
    tracker, clock, session = make()
    timeline = speak_utterance(tracker, clock)
    clock.advance(0.3)
    observer = RuntimeEventProcessor(session, timeline=tracker)
    audio = TTSAudioRawFrame(pcm(200), RATE, 1)
    drive(clock, observer, [audio, settle(0.05), TTSStoppedFrame(), settle(0.05)])
    assert timeline.finished
    [event] = session.timelines()
    assert event["marks_ms"]["first_audio"] == pytest.approx(650, abs=0.2)


def test_only_the_latency_instance_of_the_event_processor_feeds_the_timeline() -> None:
    tracker, clock, session = make()
    speak_utterance(tracker, clock)
    observer = RuntimeEventProcessor(session, assistant_output=True, timeline=tracker)
    drive(clock, observer, [TTSAudioRawFrame(pcm(200), RATE, 1), settle(0.05), TTSStoppedFrame(), settle(0.05)])
    assert session.timelines() == []


def test_bot_stopped_ends_an_utterance_of_unknown_length() -> None:
    tracker, clock, session = make("agent_translation")
    timeline = tracker.speech_started("", 0.2)
    tracker.take("")
    tracker.enqueue_audio(timeline)
    tracker.speech_done(timeline, count_known=False)
    observer = RuntimeEventProcessor(session, timeline=tracker)
    drive(
        clock,
        observer,
        [TTSAudioRawFrame(pcm(100), RATE, 1), settle(0.05), BotStoppedSpeakingFrame(), settle(0.05)],
    )
    [event] = session.timelines()
    assert event["engine"] == "agent_translation"


def test_abandon_returns_an_unheard_utterance_and_unblocks_the_queue() -> None:
    tracker, clock, session = make()
    cut = speak_utterance(tracker, clock)  # queued, nothing played yet
    assert tracker.abandon(cut) is True and not cut.queued and not cut.speech_complete
    # The queue no longer waits for it: the next utterance gets the audio.
    other = tracker.speech_started("", 0.2)
    tracker.take("")
    tracker.enqueue_audio(other)
    tracker.chunk_pushed(other)
    tracker.speech_done(other)
    tracker.on_audio(pcm(100), RATE)
    tracker.on_tts_stopped()
    assert len(session.timelines()) == 1 and other.emitted and not cut.emitted
    # ... and it can be queued again for the next reply to the same speech.
    tracker.enqueue_audio(cut)
    assert cut.queued
    # Once audio has been heard it cannot be taken back.
    assert tracker.abandon(other) is False
