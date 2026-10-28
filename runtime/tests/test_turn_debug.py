"""ST-02a: turn_id correlation and the TurnDebug observer, fed with synthetic frames."""

import asyncio
import math
import struct
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
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
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language

from src.runtime.events import EVENT_TYPES
from src.runtime.frame_events import RuntimeEventProcessor
from src.runtime.timeline import TimelineTracker
from src.runtime.turn_debug import (
    AudioLevel,
    TurnDebugObserver,
    frame_levels,
    stt_confidence,
    watch_turn_debug,
)
from src.runtime.turn_ids import TurnState, turn_state

RATE = 16000


class DeepgramSTTService:
    pass


class OpenAILLMService:
    def __init__(self) -> None:
        self._settings = SimpleNamespace(model="gpt-4.1-mini")


class CartesiaTTSService:
    def __init__(self) -> None:
        self._settings = SimpleNamespace(voice="voice-123")


class Sink:  # transports, aggregators: not a provider
    pass


class FakeSession:
    def __init__(self) -> None:
        self.event_client = object()
        self.session_id = uuid4()
        self.events: list[tuple[str, dict]] = []

    def event(self, event_type, payload=None):
        return (event_type, payload or {})

    def enqueue_event(self, event) -> None:
        self.events.append(event)

    def of(self, event_type: str) -> list[dict]:
        return [payload for kind, payload in self.events if kind == event_type]


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def pcm(ms: int, amplitude: int) -> bytes:
    return struct.pack("<h", amplitude) * (RATE * ms // 1000)


class Rig:
    """Feeds frames to the observer the way a pipeline would."""

    def __init__(self, providers=None):
        self.session = FakeSession()
        self.clock = Clock()
        self.observer = TurnDebugObserver(
            self.session, providers or {"tts": "cartesia"}, clock=self.clock
        )
        self.stt, self.llm, self.tts, self.other = (
            DeepgramSTTService(),
            OpenAILLMService(),
            CartesiaTTSService(),
            Sink(),
        )

    def push(self, frame, source=None, destination=None, *, first=True, direction=None):
        data = FramePushed(
            source=source or self.other,
            destination=destination or self.other,
            frame=frame,
            direction=direction or FrameDirection.DOWNSTREAM,
            timestamp=0,
            first_push=first,
        )
        asyncio.run(self.observer.on_push_frame(data))

    def advance(self, seconds: float) -> None:
        self.clock.now += seconds

    def user_turn(self, text="hello there", *, result=None, speech_ms=200, amplitude=3000):
        self.push(VADUserStartedSpeakingFrame())
        self.push(UserStartedSpeakingFrame())
        for _ in range(speech_ms // 20):
            self.push(InputAudioRawFrame(pcm(20, amplitude), RATE, 1))
        self.push(VADUserStoppedSpeakingFrame())
        self.push(
            TranscriptionFrame(text, "u1", "now", language=Language.EN, result=result),
            source=self.stt,
        )
        self.push(UserStoppedSpeakingFrame())

    def answer(self, text="Hi, how can I help?", *, tokens=(120, 18)):
        self.push(LLMContextFrame(LLMContext()), destination=self.llm)
        self.advance(0.4)
        self.push(LLMTextFrame(text), source=self.llm)
        self.push(
            MetricsFrame(
                data=[
                    LLMUsageMetricsData(
                        processor="OpenAILLMService#0",
                        model="gpt-4.1-mini",
                        value=LLMTokenUsage(
                            prompt_tokens=tokens[0],
                            completion_tokens=tokens[1],
                            total_tokens=sum(tokens),
                        ),
                    )
                ]
            ),
            source=self.llm,
        )
        self.push(LLMFullResponseEndFrame(), source=self.llm)
        self.push(TTSStartedFrame(), source=self.tts)
        self.advance(0.25)
        self.push(TTSAudioRawFrame(pcm(20, 1000), RATE, 1), source=self.tts)
        for word in text.split():
            self.push(TTSTextFrame(word, "sentence"), source=self.tts)
        self.push(BotStoppedSpeakingFrame())


def deepgram_result(confidence):
    return SimpleNamespace(
        channel=SimpleNamespace(alternatives=[SimpleNamespace(confidence=confidence)])
    )


# -- payload ---------------------------------------------------------------------------


def test_turn_debug_event_type_is_declared() -> None:
    assert "TurnDebug" in EVENT_TYPES


def test_full_turn_emits_one_turn_debug_with_every_section() -> None:
    rig = Rig()
    rig.user_turn(result=deepgram_result(0.93))
    rig.answer()

    [payload] = rig.session.of("TurnDebug")
    assert payload["turn_id"] == 1
    assert payload["stt"] == {"text": "hello there", "language": "en", "confidence": 0.93}
    assert payload["llm"] == {
        "model": "gpt-4.1-mini",
        "input_tokens": 120,
        "output_tokens": 18,
        "first_token_ms": 400.0,
    }
    assert payload["tts"] == {
        "provider": "cartesia",
        "voice_id": "voice-123",
        "text": "Hi, how can I help?",
        "first_audio_ms": 250.0,
    }
    assert payload["input"]["voiced_ms"] == 200.0
    assert payload["input"]["rms_dbfs"] == pytest.approx(20 * math.log10(3000 / 32768), abs=0.1)
    assert payload["input"]["peak_dbfs"] == payload["input"]["rms_dbfs"]


def test_turn_ids_increment_and_each_turn_is_reported_once() -> None:
    rig = Rig()
    rig.user_turn("first")
    rig.answer()
    rig.user_turn("second")
    rig.answer()
    rig.push(BotStoppedSpeakingFrame())  # a repeated stop must not re-emit

    payloads = rig.session.of("TurnDebug")
    assert [p["turn_id"] for p in payloads] == [1, 2]
    assert [p["stt"]["text"] for p in payloads] == ["first", "second"]
    # The second turn's audio level is its own, not accumulated over the first.
    assert payloads[1]["input"]["voiced_ms"] == 200.0


def test_turn_without_response_is_flushed_by_the_next_turn_with_null_llm_tts() -> None:
    rig = Rig()
    rig.user_turn("anyone there")
    assert rig.session.of("TurnDebug") == []
    rig.user_turn("hello?")

    [first] = rig.session.of("TurnDebug")
    assert first["turn_id"] == 1
    assert first["stt"]["text"] == "anyone there"
    assert first["llm"] is None and first["tts"] is None
    assert first["stt"]["confidence"] is None


def test_open_turn_is_flushed_on_cancel_and_cleanup() -> None:
    rig = Rig()
    rig.user_turn()
    rig.push(CancelFrame())
    assert len(rig.session.of("TurnDebug")) == 1

    rig = Rig()
    rig.user_turn()
    asyncio.run(rig.observer.cleanup())
    assert len(rig.session.of("TurnDebug")) == 1


def test_bot_stop_between_tool_call_preamble_and_answer_does_not_split_the_turn() -> None:
    rig = Rig()
    rig.user_turn("what is the weather")
    rig.push(LLMContextFrame(LLMContext()), destination=rig.llm)
    rig.push(LLMTextFrame("Let me check."), source=rig.llm)
    rig.push(LLMFullResponseEndFrame(), source=rig.llm)
    rig.push(FunctionCallInProgressFrame(function_name="w", tool_call_id="1", arguments={}), source=rig.llm)
    rig.push(TTSStartedFrame(), source=rig.tts)
    rig.push(BotStoppedSpeakingFrame())  # preamble done, tool still running
    assert rig.session.of("TurnDebug") == []

    rig.push(
        FunctionCallResultFrame(function_name="w", tool_call_id="1", arguments={}, result="sunny")
    )
    rig.push(LLMContextFrame(LLMContext()), destination=rig.llm)
    rig.push(LLMTextFrame("Sunny."), source=rig.llm)
    rig.push(LLMFullResponseEndFrame(), source=rig.llm)
    rig.push(BotStoppedSpeakingFrame())

    [payload] = rig.session.of("TurnDebug")
    assert payload["turn_id"] == 1
    assert payload["tts"]["text"] == "Let me check.Sunny."


def test_token_usage_is_summed_over_llm_calls_and_missing_usage_stays_null() -> None:
    rig = Rig()
    rig.user_turn()
    rig.push(LLMContextFrame(LLMContext()), destination=rig.llm)
    rig.push(LLMTextFrame("ok"), source=rig.llm)
    rig.push(LLMFullResponseEndFrame(), source=rig.llm)
    rig.push(BotStoppedSpeakingFrame())
    [payload] = rig.session.of("TurnDebug")
    assert payload["llm"]["input_tokens"] is None and payload["llm"]["output_tokens"] is None
    assert payload["llm"]["model"] == "gpt-4.1-mini"  # from the service settings

    rig = Rig()
    rig.user_turn()
    rig.answer(tokens=(100, 10))
    assert rig.session.of("TurnDebug")[0]["llm"]["input_tokens"] == 100
    rig.user_turn("again")
    for _ in range(2):
        rig.push(LLMContextFrame(LLMContext()), destination=rig.llm)
        rig.push(
            MetricsFrame(
                data=[
                    LLMUsageMetricsData(
                        processor="OpenAILLMService#0",
                        value=LLMTokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
                    )
                ]
            ),
            source=rig.llm,
        )
        rig.push(LLMFullResponseEndFrame(), source=rig.llm)
    rig.push(BotStoppedSpeakingFrame())
    last = rig.session.of("TurnDebug")[-1]
    assert (last["llm"]["input_tokens"], last["llm"]["output_tokens"]) == (10, 4)


def test_ttfb_metrics_back_up_missing_first_token_and_first_audio_timings() -> None:
    rig = Rig()
    rig.user_turn()
    rig.push(
        MetricsFrame(
            data=[
                TTFBMetricsData(processor="OpenAILLMService#0", value=0.31),
                TTFBMetricsData(processor="CartesiaTTSService#0", value=0.12),
            ]
        ),
        source=rig.llm,
    )
    rig.push(LLMTextFrame("hi"), source=rig.llm)  # no context frame seen: no measured timing
    rig.push(TTSTextFrame("hi", "sentence"), source=rig.tts)
    rig.push(BotStoppedSpeakingFrame())
    [payload] = rig.session.of("TurnDebug")
    assert payload["llm"]["first_token_ms"] == 310.0
    assert payload["tts"]["first_audio_ms"] == 120.0


def test_non_provider_processors_and_replayed_hops_are_ignored() -> None:
    rig = Rig()
    rig.user_turn()
    rig.push(LLMTextFrame("not from an llm"), source=rig.other)
    rig.push(TTSStartedFrame(), source=rig.other)
    rig.push(LLMTextFrame("repeat"), source=rig.llm, first=False)
    rig.push(BotStoppedSpeakingFrame())
    assert rig.session.of("TurnDebug") == []  # nothing responded


def test_watch_turn_debug_attaches_the_observer() -> None:
    attached = []
    worker = SimpleNamespace(add_observer=attached.append)
    watch_turn_debug(worker, FakeSession(), {"tts": "cartesia"})
    assert [type(o) for o in attached] == [TurnDebugObserver]
    watch_turn_debug(worker, None, {})
    assert len(attached) == 1


# -- audio level -----------------------------------------------------------------------


def test_rms_and_peak_of_known_signals() -> None:
    squares, peak, samples = frame_levels(pcm(20, 1000))
    assert (peak, samples) == (1000, 320)
    assert squares == pytest.approx(320 * 1000**2)
    assert frame_levels(b"") == (0.0, 0, 0)
    assert frame_levels(b"\x01") == (0.0, 0, 0)  # a dangling byte is not a sample

    level = AudioLevel()
    level.add(pcm(20, 16384), RATE)  # half of full scale
    assert level.payload() == {"rms_dbfs": -6.0, "peak_dbfs": -6.0, "voiced_ms": 20.0}


def test_mixed_levels_use_true_rms_and_loudest_peak_and_silence_is_not_voiced() -> None:
    level = AudioLevel()
    level.add(pcm(20, 0), RATE)
    level.add(pcm(20, 8192), RATE)
    payload = level.payload()
    assert payload["voiced_ms"] == 20.0
    assert payload["peak_dbfs"] == pytest.approx(20 * math.log10(8192 / 32768), abs=0.1)
    # sqrt(mean(0, 8192^2)) = 8192 / sqrt(2): 3 dB under the peak
    assert payload["rms_dbfs"] == pytest.approx(payload["peak_dbfs"] - 3.0, abs=0.1)


def test_no_audio_gives_null_levels() -> None:
    assert AudioLevel().payload() == {"rms_dbfs": None, "peak_dbfs": None, "voiced_ms": 0.0}
    assert AudioLevel().payload() == {"rms_dbfs": None, "peak_dbfs": None, "voiced_ms": 0.0}
    silent = AudioLevel()
    silent.add(pcm(20, 0), RATE)
    assert silent.payload()["rms_dbfs"] == -100.0


def test_audio_outside_vad_speech_is_not_counted() -> None:
    rig = Rig()
    rig.push(InputAudioRawFrame(pcm(20, 20000), RATE, 1))  # before any speech
    rig.user_turn(speech_ms=40, amplitude=1000)
    rig.answer()
    [payload] = rig.session.of("TurnDebug")
    assert payload["input"]["voiced_ms"] == 40.0
    assert payload["input"]["peak_dbfs"] == pytest.approx(20 * math.log10(1000 / 32768), abs=0.1)


# -- confidence ------------------------------------------------------------------------


def test_stt_confidence_extraction() -> None:
    assert stt_confidence(deepgram_result(0.9876543)) == 0.9877
    assert stt_confidence({"channel": {"alternatives": [{"confidence": 0.5}]}}) == 0.5
    assert stt_confidence(None) is None
    assert stt_confidence(object()) is None
    assert stt_confidence({"channel": {"alternatives": []}}) is None
    assert stt_confidence(deepgram_result(None)) is None
    assert stt_confidence(deepgram_result("0.9")) is None
    assert stt_confidence(deepgram_result(float("nan"))) is None
    assert stt_confidence(deepgram_result(1.5)) is None


def test_confidence_is_the_mean_of_the_turn_segments() -> None:
    rig = Rig()
    rig.push(UserStartedSpeakingFrame())
    for text, confidence in (("one", 0.8), ("two", 0.6), ("three", None)):
        rig.push(
            TranscriptionFrame(
                text, "u1", "now", result=deepgram_result(confidence) if confidence else None
            ),
            source=rig.stt,
        )
    rig.push(UserStoppedSpeakingFrame())
    rig.answer()
    [payload] = rig.session.of("TurnDebug")
    assert payload["stt"] == {"text": "one two three", "language": None, "confidence": 0.7}


# -- turn ids in other events ----------------------------------------------------------


def test_turn_state_is_idempotent_while_open_and_shared_per_session() -> None:
    session = FakeSession()
    state = turn_state(session)
    assert state is turn_state(session) and state.current == 0
    assert state.begin() == 1 and state.begin() == 1
    state.end()
    assert state.begin() == 2
    assert state.new_turn() == 3
    assert turn_state(None) is None
    assert turn_state(FakeSession()).current == 0  # sessions do not share


class _Recorder:
    pass


async def _drive(processor, frame, direction=FrameDirection.DOWNSTREAM) -> None:
    async def forward(forwarded, forwarded_direction) -> None:
        return None

    processor.push_frame = forward
    await processor.process_frame(frame, direction)


def test_transcript_and_latency_events_carry_the_turn_id() -> None:
    session = FakeSession()
    user = RuntimeEventProcessor(session, control_events=False)
    assistant = RuntimeEventProcessor(session, assistant_output=True, control_events=False)
    control = RuntimeEventProcessor(session)
    session.begin_listening = lambda: None
    session.interrupt = lambda: None

    async def run() -> None:
        for turn in (1, 2):
            await _drive(control, UserStartedSpeakingFrame())
            await _drive(
                user,
                TranscriptionFrame(
                    f"q{turn}",
                    "u",
                    "now",
                    language=Language.EN,
                    finalized=True,
                    result=deepgram_result(0.9) if turn == 1 else None,
                ),
            )
            await _drive(control, UserStoppedSpeakingFrame())
            await _drive(assistant, LLMTextFrame(f"a{turn}"))
            await _drive(assistant, LLMFullResponseEndFrame())
            await _drive(
                control,
                MetricsFrame(data=[TTFBMetricsData(processor="CartesiaTTSService#0", value=0.2)]),
            )

    asyncio.run(run())

    transcripts = session.of("TranscriptFinal")
    assert [(t["role"], t["turn_id"]) for t in transcripts] == [
        ("user", 1),
        ("assistant", 1),
        ("user", 2),
        ("assistant", 2),
    ]
    assert transcripts[0]["confidence"] == 0.9
    assert "confidence" not in transcripts[2]
    assert [p["turn_id"] for p in session.of("LatencyMeasured")] == [1, 2]


def test_timeline_events_carry_the_session_turn_id() -> None:
    session = FakeSession()
    clock = Clock()
    tracker = TimelineTracker(session, "agent_translation", clock=clock)

    def utterance() -> None:
        timeline = tracker.speech_started("", 0.2)
        clock.now += 0.1
        tracker.turn_stopped("")
        tracker.take("")
        tracker.enqueue_audio(timeline)
        tracker.chunk_pushed(timeline)
        tracker.speech_done(timeline)
        tracker.on_audio(pcm(600, 4000), RATE)
        tracker.on_tts_stopped()

    state = turn_state(session)
    state.begin()  # the agent pipeline already opened turn 1 at UserStartedSpeakingFrame
    utterance()
    utterance()

    ids = [p["turn_id"] for p in session.of("LatencyMeasured")]
    assert ids == [1, 2]


def test_two_person_utterances_get_distinct_turn_ids() -> None:
    session = FakeSession()
    tracker = TimelineTracker(session, "cascaded", clock=Clock())
    a = tracker.speech_started("alice", 0.0)
    b = tracker.speech_started("bob", 0.0)
    assert (a.turn_id, b.turn_id) == (1, 2)


def test_timeline_without_a_session_has_no_turn_id() -> None:
    timeline = TimelineTracker(None, "cascaded", clock=Clock()).speech_started("", 0.0)
    assert timeline.turn_id is None


def test_turn_state_class_is_importable_without_pipecat_side_effects() -> None:
    assert TurnState().current == 0
