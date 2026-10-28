"""ST-10a: the stall watchdog reports hung STT/LLM/TTS hand-offs, and only those."""

import asyncio
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker

from src.runtime.provider_failures import watch_provider_failures
from src.runtime.stall_watchdog import (
    LLM_STALL_SECS,
    STT_STALL_SECS,
    TTS_STALL_SECS,
    StallWatchdog,
    guard_first_token,
)


class FakeDeepgramSTTService(FrameProcessor):
    pass


class FakeOpenAILLMService(FrameProcessor):
    pass


class FakeCartesiaTTSService(FrameProcessor):
    pass


class Aggregator(FrameProcessor):
    pass


class FakeTime:
    """Injectable clock + sleep: ``advance`` moves time and wakes the due sleepers."""

    def __init__(self) -> None:
        self.now = 0.0
        self._sleepers: list[tuple[float, asyncio.Future]] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, secs: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self._sleepers.append((self.now + secs, future))
        await future

    async def advance(self, secs: float) -> None:
        for _ in range(5):  # let timers armed just before register their sleep
            await asyncio.sleep(0)
        self.now += secs
        for due, future in list(self._sleepers):
            if due <= self.now + 1e-9 and not future.done():
                future.set_result(None)
        for _ in range(5):  # let woken timers run
            await asyncio.sleep(0)


class FakeSession:
    def __init__(self) -> None:
        self.event_client = object()
        self.session_id = uuid4()
        self.events: list[tuple[str, dict]] = []
        self.failures: list[str] = []
        self.end_requests: list[str] = []

    def event(self, event_type, payload=None):
        return (event_type, payload or {})

    def enqueue_event(self, event) -> None:
        self.events.append(event)

    def fail(self, reason) -> None:
        self.failures.append(reason)

    async def request_end(self, reason="") -> bool:
        self.end_requests.append(reason)
        return True

    def of(self, event_type: str) -> list[dict]:
        return [payload for kind, payload in self.events if kind == event_type]


STT, LLM, TTS, AGG = (
    FakeDeepgramSTTService(),
    FakeOpenAILLMService(),
    FakeCartesiaTTSService(),
    Aggregator(),
)


def push(source, destination, frame, *, first=True) -> FramePushed:
    return FramePushed(source, destination, frame, FrameDirection.DOWNSTREAM, 0, first)


def make(**kwargs):
    session = FakeSession()
    fake = FakeTime()
    watchdog = StallWatchdog(
        session,
        {"stt": "deepgram", "llm": "openai", "tts": "cartesia"},
        clock=fake.clock,
        sleep=fake.sleep,
        **kwargs,
    )
    return watchdog, session, fake


def transcript(text="hello", final=True):
    cls = TranscriptionFrame if final else InterimTranscriptionFrame
    return cls(text, "u", "2026-01-01T00:00:00Z", None)


async def user_speech(watchdog, fake, secs=2.0, *, stop_secs=0.2):
    await watchdog.on_push_frame(push(AGG, STT, VADUserStartedSpeakingFrame(start_secs=0.2)))
    await fake.advance(secs)
    await watchdog.on_push_frame(push(AGG, STT, VADUserStoppedSpeakingFrame(stop_secs=stop_secs)))


def run(coro):
    return asyncio.run(coro)


def test_defaults_match_the_documented_thresholds() -> None:
    assert (STT_STALL_SECS, LLM_STALL_SECS, TTS_STALL_SECS) == (5.0, 8.0, 4.0)


# -- STT ---------------------------------------------------------------------------


def test_stt_fires_at_the_threshold_and_not_before() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await user_speech(watchdog, fake)
        assert watchdog.armed == {"stt"}
        await fake.advance(4.9)
        assert session.of("ProviderError") == []
        await fake.advance(0.1)
        return session

    [event] = run(scenario()).of("ProviderError")
    assert event == {
        "provider": "deepgram",
        "kind": "stt",
        "category": "timeout",
        "stage": "no_final_transcript",
        "retryable": True,
        "attempt": 1,
        "message": "deepgram stt timed out",
    }


def test_stt_no_false_fire_when_the_final_transcript_arrives_in_time() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await user_speech(watchdog, fake)
        await fake.advance(3.0)
        await watchdog.on_push_frame(push(STT, AGG, transcript()))
        assert watchdog.armed == frozenset()
        await fake.advance(60.0)
        return session

    assert run(scenario()).of("ProviderError") == []


def test_stt_ignores_clicks_and_breaths_shorter_than_the_minimum_speech() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await user_speech(watchdog, fake, secs=0.6)  # 0.6 s - 0.2 stop + 0.2 start = 0.6 s
        assert watchdog.armed == frozenset()
        await fake.advance(60.0)
        return session

    assert run(scenario()).of("ProviderError") == []


def test_stt_final_that_arrived_during_the_speech_covers_the_turn() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(AGG, STT, VADUserStartedSpeakingFrame(start_secs=0.2)))
        await fake.advance(1.5)
        await watchdog.on_push_frame(push(STT, AGG, transcript()))  # endpointed mid-speech
        await fake.advance(1.0)
        await watchdog.on_push_frame(push(AGG, STT, VADUserStoppedSpeakingFrame(stop_secs=0.2)))
        assert watchdog.armed == frozenset()
        await fake.advance(60.0)
        return session

    assert run(scenario()).of("ProviderError") == []


def test_stt_interim_text_means_the_stt_is_alive() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(AGG, STT, VADUserStartedSpeakingFrame(start_secs=0.2)))
        await watchdog.on_push_frame(push(STT, AGG, transcript(final=False)))
        await fake.advance(2.0)
        await watchdog.on_push_frame(push(AGG, STT, VADUserStoppedSpeakingFrame(stop_secs=0.2)))
        await fake.advance(60.0)
        return session

    assert run(scenario()).of("ProviderError") == []


def test_stt_new_speech_before_the_deadline_cancels_the_timer() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await user_speech(watchdog, fake)
        await fake.advance(3.0)
        await watchdog.on_push_frame(push(AGG, STT, VADUserStartedSpeakingFrame(start_secs=0.2)))
        assert watchdog.armed == frozenset()
        await fake.advance(60.0)
        return session

    assert run(scenario()).of("ProviderError") == []


def test_broadcast_vad_frames_arriving_twice_are_harmless() -> None:
    async def scenario():
        watchdog, session, fake = make()
        started = VADUserStartedSpeakingFrame(start_secs=0.2)
        await watchdog.on_push_frame(push(AGG, STT, started))
        await watchdog.on_push_frame(push(AGG, AGG, VADUserStartedSpeakingFrame(start_secs=0.2)))
        await fake.advance(2.0)
        await watchdog.on_push_frame(push(AGG, STT, VADUserStoppedSpeakingFrame(stop_secs=0.2)))
        await watchdog.on_push_frame(push(AGG, AGG, VADUserStoppedSpeakingFrame(stop_secs=0.2)))
        await fake.advance(5.0)
        return session

    assert len(run(scenario()).of("ProviderError")) == 1


# -- LLM ---------------------------------------------------------------------------


def context_frame() -> LLMContextFrame:
    return LLMContextFrame(LLMContext())


def test_llm_fires_at_the_threshold_without_a_first_token() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(AGG, LLM, context_frame()))
        assert watchdog.armed == {"llm"}
        await fake.advance(7.9)
        assert session.of("ProviderError") == []
        await fake.advance(0.1)
        return session

    [event] = run(scenario()).of("ProviderError")
    assert (event["kind"], event["provider"], event["category"], event["stage"]) == (
        "llm",
        "openai",
        "timeout",
        "no_first_token",
    )


def test_llm_context_pushed_to_a_non_llm_does_not_arm() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(AGG, AGG, context_frame()))
        return watchdog.armed

    assert run(scenario()) == frozenset()


def test_llm_no_false_fire_when_tokens_arrive_or_the_response_is_empty() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(AGG, LLM, context_frame()))
        await fake.advance(6.0)
        await watchdog.on_push_frame(push(LLM, TTS, LLMTextFrame("Hi")))
        await fake.advance(60.0)
        await watchdog.on_push_frame(push(AGG, LLM, context_frame()))
        await watchdog.on_push_frame(push(LLM, TTS, LLMFullResponseEndFrame()))  # empty answer
        await fake.advance(60.0)
        return session

    assert run(scenario()).of("ProviderError") == []


# -- TTS ---------------------------------------------------------------------------


def test_tts_fires_at_the_threshold_without_first_audio() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(TTS, AGG, TTSStartedFrame()))
        await fake.advance(3.9)
        assert session.of("ProviderError") == []
        await fake.advance(0.1)
        return session

    [event] = run(scenario()).of("ProviderError")
    assert (event["kind"], event["provider"], event["category"], event["stage"]) == (
        "tts",
        "cartesia",
        "timeout",
        "no_first_audio",
    )


def test_tts_no_false_fire_when_audio_arrives_or_the_user_interrupts() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(TTS, AGG, TTSStartedFrame()))
        await fake.advance(2.0)
        audio = TTSAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1)
        await watchdog.on_push_frame(push(TTS, AGG, audio))
        await fake.advance(60.0)
        await watchdog.on_push_frame(push(TTS, AGG, TTSStartedFrame()))
        await watchdog.on_push_frame(push(AGG, TTS, InterruptionFrame()))
        await fake.advance(60.0)
        return session

    assert run(scenario()).of("ProviderError") == []


def test_tts_started_frames_not_from_a_tts_service_do_not_arm() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(AGG, AGG, TTSStartedFrame()))  # a later hop
        return watchdog.armed

    assert run(scenario()) == frozenset()


# -- policy ----------------------------------------------------------------------------


def test_two_stalls_in_the_window_end_the_call_one_does_not() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(TTS, AGG, TTSStartedFrame()))
        await fake.advance(4.0)
        assert session.failures == [] and session.end_requests == []  # retry, call goes on
        await watchdog.on_push_frame(push(TTS, AGG, TTSStartedFrame()))
        await fake.advance(4.0)
        await asyncio.sleep(0)
        return watchdog, session

    watchdog, session = run(scenario())
    assert [(e["attempt"], e["retryable"]) for e in session.of("ProviderError")] == [
        (1, True),
        (2, False),
    ]
    assert session.failures == ["provider_failed"]
    assert session.end_requests == ["provider_failed"]
    assert watchdog.armed == frozenset()


def test_stall_then_recovery_emits_the_error_but_no_session_failure() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(AGG, LLM, context_frame()))
        await fake.advance(8.0)
        await watchdog.on_push_frame(push(LLM, TTS, LLMTextFrame("late but fine")))
        await fake.advance(60.0)
        return session

    session = run(scenario())
    [event] = session.of("ProviderError")
    assert event["retryable"] is True and event["attempt"] == 1
    assert session.failures == [] and session.end_requests == []


def test_end_frame_and_cleanup_cancel_pending_timers() -> None:
    async def scenario():
        watchdog, session, fake = make()
        await watchdog.on_push_frame(push(TTS, AGG, TTSStartedFrame()))
        await watchdog.on_push_frame(push(AGG, LLM, context_frame()))
        await watchdog.on_push_frame(push(AGG, TTS, EndFrame()))
        assert watchdog.armed == frozenset()
        await watchdog.on_push_frame(push(TTS, AGG, TTSStartedFrame()))
        await watchdog.cleanup()
        await fake.advance(60.0)
        return session

    assert run(scenario()).of("ProviderError") == []


# -- wiring --------------------------------------------------------------------------------


def test_watch_provider_failures_attaches_the_watchdog_to_a_real_worker() -> None:
    async def scenario():
        worker = PipelineWorker(Pipeline([FrameProcessor()]), params=PipelineParams())
        added = []
        real = worker.add_observer
        worker.add_observer = lambda observer: (added.append(observer), real(observer))[1]
        watch_provider_failures(worker, FakeSession(), {"stt": "deepgram"})
        skipped = []
        worker.add_observer = lambda observer: skipped.append(observer)
        watch_provider_failures(worker, FakeSession(), {}, stall_watchdog=False)
        return added, skipped

    added, skipped = run(scenario())
    assert len(added) == 1 and isinstance(added[0], StallWatchdog)
    assert skipped == []


# -- translator first-token guard -------------------------------------------------------


def test_guard_first_token_passes_a_prompt_stream_through_and_times_out_a_hung_one() -> None:
    async def prompt():
        yield "a"
        await asyncio.sleep(0.05)  # later gaps are not a stall
        yield "b"

    async def hung():
        await asyncio.sleep(3600)
        yield "never"

    async def scenario():
        got = [delta async for delta in guard_first_token(prompt(), 0.2)]
        with pytest.raises(TimeoutError):
            async for _ in guard_first_token(hung(), 0.05):
                pass
        return got

    assert run(scenario()) == ["a", "b"]


# -- end to end through a real worker (frame names, source/destination, first_push) ------------


class HungCartesiaTTSService(FrameProcessor):
    """A TTS that accepts text ("started") and then never returns audio."""

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        from pipecat.frames.frames import TTSSpeakFrame

        if isinstance(frame, TTSSpeakFrame):
            await self.push_frame(TTSStartedFrame())
        else:
            await self.push_frame(frame, direction)


def test_real_worker_reports_a_tts_that_never_returns_audio() -> None:
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.tests.utils import SleepFrame, run_test

    async def scenario():
        session = FakeSession()
        watchdog = StallWatchdog(session, {"tts": "cartesia"}, tts_secs=0.2)
        await run_test(
            HungCartesiaTTSService(),
            frames_to_send=[TTSSpeakFrame("hello"), SleepFrame(sleep=0.6)],
            expected_down_frames=None,
            observers=[watchdog],
        )
        return session

    [event] = asyncio.run(scenario()).of("ProviderError")
    assert (event["kind"], event["category"], event["stage"]) == ("tts", "timeout", "no_first_audio")
