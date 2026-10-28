"""ST-10b: a terminal provider failure plays a pre-rendered apology before the call ends."""

import array
import asyncio
import wave
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

import src.runtime.pipeline
import src.runtime.session
from pipecat.frames.frames import OutputAudioRawFrame, TTSStoppedFrame

from src.runtime import apology
from src.runtime.apology import (
    APOLOGY_TEXTS,
    ASSET_DIR,
    MAX_PLAYBACK_SECONDS,
    OUTPUT_SAMPLE_RATE,
    apology_plans,
    available_languages,
    load_apology,
    resample_linear,
)
from src.runtime.participants import ParticipantDirectory
from src.runtime.pipeline import PipelineConfig
from src.runtime.provider_failures import report_provider_failure
from src.runtime.session import RuntimeSession

TIMEOUT = 10


def config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        "ws://livekit", "token", "room-42", "deepgram", "openai", "cartesia", "voice", "prompt", **kwargs
    )


def write_wav(path: Path, seconds: float = 0.1, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(array.array("h", [1000] * int(seconds * rate)).tobytes())


@pytest.fixture
def short_clips(tmp_path) -> Path:
    for language in ("en", "ru", "ko"):
        write_wav(tmp_path / f"{language}.wav")
    return tmp_path


# -- assets and language selection -----------------------------------------------------


def test_shipped_clips_are_small_mono_pcm_and_cover_the_required_languages() -> None:
    languages = available_languages()
    assert {"en", "ru", "ko"} <= set(languages)
    assert set(languages) <= set(APOLOGY_TEXTS)
    for language in languages:
        path = ASSET_DIR / f"{language}.wav"
        assert path.stat().st_size < 100_000
        with wave.open(str(path), "rb") as handle:
            assert (handle.getnchannels(), handle.getsampwidth()) == (1, 2)
        clip = load_apology(language)
        assert clip.language == language
        assert clip.sample_rate == OUTPUT_SAMPLE_RATE  # converted to the transport output rate
        assert 1.0 < clip.duration < MAX_PLAYBACK_SECONDS - apology.PLAYBACK_MARGIN_SECONDS


def test_language_codes_are_normalised_and_unknown_languages_fall_back_to_english() -> None:
    assert load_apology("ru").language == "ru"
    assert load_apology("RU-ru").language == "ru"
    assert load_apology("ko_KR").language == "ko"
    for unknown in ("xx", "", None):
        assert load_apology(unknown).language == "en"


def test_fallback_to_english_when_a_language_clip_is_missing(short_clips) -> None:
    (short_clips / "ru.wav").unlink()
    assert load_apology("ru", directory=short_clips).language == "en"
    assert load_apology("ru", directory=short_clips.parent / "empty") is None  # nothing to play


def test_resample_linear_changes_length_and_keeps_the_shape() -> None:
    samples = array.array("h", [0, 100, 200, 300])
    up = resample_linear(samples, 8000, 16000)
    assert len(up) == 8 and up[0] == 0 and up[2] == 100 and up[-1] == 300
    assert list(resample_linear(samples, 8000, 8000)) == [0, 100, 200, 300]
    assert len(resample_linear(array.array("h"), 8000, 24000)) == 0


def test_a_clip_at_another_rate_is_converted_and_a_long_one_truncated(tmp_path) -> None:
    write_wav(tmp_path / "en.wav", seconds=10, rate=8000)
    clip = load_apology("en", directory=tmp_path)
    assert clip.sample_rate == OUTPUT_SAMPLE_RATE
    assert clip.duration == pytest.approx(apology.MAX_CLIP_SECONDS, abs=0.01)


def plans_of(cfg, **kwargs):
    return [(p.destination, p.clip.language) for p in apology_plans(cfg, **kwargs)]


def test_agent_without_translation_speaks_english() -> None:
    assert plans_of(config()) == [(None, "en")]


def test_agent_translation_uses_the_callers_current_language() -> None:
    cfg = config(translation={"working_language": "en", "caller_languages": ["ko", "ru"]})
    assert plans_of(cfg) == [(None, "ko")]  # before any caller turn: the first caller language
    assert plans_of(cfg, caller_language=lambda: "ru") == [(None, "ru")]
    assert plans_of(cfg, caller_language=lambda: None) == [(None, "ko")]

    def broken():
        raise RuntimeError("state gone")

    assert plans_of(cfg, caller_language=broken) == [(None, "ko")]
    assert plans_of(config(), caller_language=lambda: "xx") == [(None, "en")]


def test_one_way_interpreter_speaks_the_listeners_language() -> None:
    cfg = config(mode="interpreter", interpreter={"languages": ["ru", "ko"], "direction": "one_way"})
    assert plans_of(cfg) == [(None, "ko")]


def test_two_person_interpreter_gives_each_listener_their_own_language_and_track() -> None:
    cfg = config(
        mode="interpreter",
        interpreter={"languages": ["ru", "ko"]},
        participants={"alice": {"language": "ru"}, "bob": {"language": "ko"}},
    )
    expected = [("translation-lane-0", "ru"), ("translation-lane-1", "ko")]
    assert plans_of(cfg) == expected
    directory = ParticipantDirectory.from_mapping(("ru", "ko"), cfg.participants)
    assert plans_of(cfg, participants=directory) == expected

    # A language without a clip falls back to English on that listener's own track.
    odd = ParticipantDirectory.from_mapping(("ru", "sw"), {"a": {"language": "ru"}, "b": {"language": "sw"}})
    assert plans_of(cfg, participants=odd) == [("translation-lane-0", "ru"), ("translation-lane-1", "en")]


# -- RuntimeSession.play_apology -----------------------------------------------------------


class Output:
    """transport.output(): records what is queued and when."""

    def __init__(self, log: list, *, hang: bool = False) -> None:
        self.log = log
        self.hang = hang
        self.frames: list[OutputAudioRawFrame] = []  # audio only
        self.stops: list[TTSStoppedFrame] = []

    async def queue_frame(self, frame, *_):
        if self.hang:
            await asyncio.Event().wait()
        if isinstance(frame, TTSStoppedFrame):  # flushes the transport's last partial chunk
            self.stops.append(frame)
            return
        self.frames.append(frame)
        self.log.append(("frame", frame.transport_destination))


class Transport:
    def __init__(self, log: list, *, hang: bool = False) -> None:
        self.handlers = {}
        self.cleaned = False
        self.out = Output(log, hang=hang)

    def output(self) -> Output:
        return self.out

    def add_event_handler(self, name, handler) -> None:
        self.handlers[name] = handler

    async def cleanup(self) -> None:
        self.cleaned = True


def live_session(transport, cfg=None, **kwargs) -> RuntimeSession:
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), None)
    session._transport = transport
    session._pipeline_config = cfg or config()
    session._pipeline_worker = object()
    return session


@pytest.fixture
def short_session_clips(monkeypatch, short_clips):
    real = apology.apology_plans
    monkeypatch.setattr(
        src.runtime.session,
        "apology_plans",
        lambda cfg, participants, caller: real(cfg, participants, caller, directory=short_clips),
    )
    monkeypatch.setattr(src.runtime.session, "PLAYBACK_MARGIN_SECONDS", 0.05)


def test_play_apology_queues_output_audio_on_the_transport_output(short_session_clips) -> None:
    log: list = []
    transport = Transport(log)
    session = live_session(transport)
    started = asyncio.new_event_loop()
    try:
        began = started.time()
        assert started.run_until_complete(session.play_apology()) is True
        elapsed = started.time() - began
    finally:
        started.close()
    (frame,) = transport.out.frames
    assert isinstance(frame, OutputAudioRawFrame)
    assert (frame.sample_rate, frame.num_channels, frame.transport_destination) == (OUTPUT_SAMPLE_RATE, 1, None)
    assert frame.interruptible is False
    assert [stop.transport_destination for stop in transport.out.stops] == [None]
    assert len(frame.audio) > 0 and len(frame.audio) % 2 == 0
    assert 0.1 <= elapsed < 1.5  # waited for the clip (0.1 s) and the margin, no longer


def test_two_person_apology_goes_to_each_listeners_destination(short_session_clips) -> None:
    transport = Transport([])
    cfg = config(
        mode="interpreter",
        interpreter={"languages": ["ru", "ko"]},
        participants={"alice": {"language": "ru"}, "bob": {"language": "ko"}},
        private_listening=True,
    )
    session = live_session(transport, cfg)
    assert asyncio.run(session.play_apology()) is True
    assert [f.transport_destination for f in transport.out.frames] == [
        "translation-lane-0",
        "translation-lane-1",
    ]


def test_play_apology_does_not_hang_when_the_output_never_accepts_the_frame(short_session_clips) -> None:
    session = live_session(Transport([], hang=True))

    async def run() -> bool:
        loop = asyncio.get_running_loop()
        began = loop.time()
        played = await asyncio.wait_for(session.play_apology(timeout=0.2), TIMEOUT)
        assert loop.time() - began < 1.5
        return played

    assert asyncio.run(run()) is False


@pytest.mark.parametrize("missing", ["transport", "config", "worker", "ending", "output", "broken"])
def test_play_apology_is_a_noop_when_there_is_nothing_to_play_to(short_session_clips, missing) -> None:
    transport = Transport([])
    session = live_session(transport)
    if missing == "transport":
        session._transport = None
    elif missing == "config":
        session._pipeline_config = None
    elif missing == "worker":
        session._pipeline_worker = None
    elif missing == "ending":
        session._pipeline_cancel_requested = True
    elif missing == "output":
        session._transport = object()  # e.g. a transport stand-in without an output
    else:
        transport.output = lambda: (_ for _ in ()).throw(RuntimeError("no output"))
    assert asyncio.run(session.play_apology()) is False
    assert transport.out.frames == []


def test_play_apology_without_any_clip_is_a_noop(monkeypatch, tmp_path) -> None:
    real = apology.apology_plans
    monkeypatch.setattr(
        src.runtime.session, "apology_plans", lambda *a: real(*a, directory=tmp_path)
    )
    transport = Transport([])
    assert asyncio.run(live_session(transport).play_apology()) is False
    assert transport.out.frames == []


# -- the failure policy plays it (and only on a terminal failure) ----------------------------


class Client:
    def __init__(self, log: list) -> None:
        self.log = log
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)
        self.log.append(("event", event.event_type))

    def of(self, event_type: str) -> list[dict]:
        return [e.payload for e in self.events if e.event_type == event_type]


class Worker:
    def __init__(self, log: list) -> None:
        self.log = log

    def cancel(self, *, reason=None) -> None:
        self.log.append(("cancel", reason))


def run_failing_session(monkeypatch, report, *, cfg=None, hang: bool = False, fallback: bool = False):
    """Run a RuntimeSession whose pipeline reports ``report(session)``; return (log, client, transport, outcome)."""

    import pipecat.workers.runner

    log: list = []
    client = Client(log)
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    transport = Transport(log, hang=hang)
    worker = Worker(log)
    outcome: list = []
    builds: list = []

    def build(*_):
        builds.append(1)
        return transport, worker

    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", build)

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, _worker) -> None:
            pass

        async def run(self) -> None:
            if len(builds) > 1:
                return  # the restarted (cascaded) pipeline just ends
            outcome.append(report(session))
            for _ in range(300):  # until the worker was told to stop (or the test gives up)
                if any(entry[0] == "cancel" for entry in log):
                    return
                await asyncio.sleep(0.01)

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(asyncio.wait_for(session.run_pipeline(cfg or config()), TIMEOUT))
    return log, client, transport, outcome[0]


def test_terminal_failure_plays_the_apology_before_failing_and_ending(monkeypatch, short_session_clips) -> None:
    log, client, transport, outcome = run_failing_session(
        monkeypatch, lambda s: report_provider_failure(s, "tts", "cartesia", "quota")
    )
    assert outcome == "failed"
    assert len(transport.out.frames) == 1
    # frame first, then the end request, then the terminal event.
    kinds = [entry[0] for entry in log if entry != ("event", "ProviderError")]
    assert kinds == ["frame", "cancel", "event"]
    assert log[-1] == ("event", "SessionFailed")
    assert client.of("SessionFailed") == [{"outcome": "failed", "reason": "provider_failed"}]


def test_apology_uses_the_callers_language_on_a_translated_agent_call(monkeypatch, short_session_clips) -> None:
    cfg = config(translation={"working_language": "en", "caller_languages": ["ko"]})

    def report(session):
        session.caller_language = lambda: "ru"
        return report_provider_failure(session, "llm", "openai", "auth")

    seen: list = []
    real = src.runtime.session.apology_plans

    def spy(*args):
        plans = real(*args)
        seen.extend(p.clip.language for p in plans)
        return plans

    monkeypatch.setattr(src.runtime.session, "apology_plans", spy)
    _, _, transport, outcome = run_failing_session(monkeypatch, report, cfg=cfg)
    assert outcome == "failed" and seen == ["ru"]
    assert len(transport.out.frames) == 1


def test_a_stuck_transport_does_not_block_ending_the_call(monkeypatch, short_session_clips) -> None:
    # play_apology's default timeout is bound at definition time; shrink it for the test.
    original = RuntimeSession.play_apology

    async def bounded(self, timeout: float = 0.2):
        return await original(self, timeout)

    monkeypatch.setattr(RuntimeSession, "play_apology", bounded)
    log, client, transport, outcome = run_failing_session(
        monkeypatch, lambda s: report_provider_failure(s, "tts", "cartesia", "auth"), hang=True
    )
    assert outcome == "failed"
    assert transport.out.frames == []
    assert [e for e in log if e[0] == "cancel"] == [("cancel", "provider_failed")]
    assert client.of("SessionFailed") == [{"outcome": "failed", "reason": "provider_failed"}]


def test_a_transient_error_and_a_noted_error_play_no_apology(monkeypatch, short_session_clips) -> None:
    def report(session):
        first = report_provider_failure(session, "stt", "deepgram", "timeout")  # tolerated once
        other = report_provider_failure(session, "llm", "openai", "other")  # below the limit
        return first, other

    log, client, transport, outcome = run_failing_session(monkeypatch, report)
    assert outcome == ("retry", "noted")
    assert transport.out.frames == []
    assert not any(e[0] == "frame" for e in log)


def test_engine_fallback_plays_no_apology(monkeypatch, short_session_clips) -> None:
    # One-way realtime with the cascaded providers configured: the session can fall back.
    cfg = config(
        mode="interpreter",
        interpreter={"languages": ["ru", "en"], "direction": "one_way", "engine": "realtime"},
    )
    log, client, transport, outcome = run_failing_session(
        monkeypatch,
        lambda s: report_provider_failure(
            s, "realtime", "openai", "unavailable", fatal=True, allow_fallback=True
        ),
        cfg=cfg,
    )
    assert outcome == "fallback"
    assert transport.out.frames == []
    assert [c.event_type for c in client.events] == ["ProviderError", "EngineFallback", "SessionEnded"]


def test_realtime_failure_without_fallback_plays_the_apology(monkeypatch, short_session_clips) -> None:
    cfg = config(mode="interpreter", interpreter={"languages": ["ru", "ko"], "direction": "one_way", "engine": "realtime"})
    log, client, transport, outcome = run_failing_session(
        monkeypatch,
        lambda s: report_provider_failure(s, "realtime", "openai", "auth", allow_fallback=True),
        cfg=cfg,
    )
    assert outcome == "failed"
    assert len(transport.out.frames) == 1
    assert [e[0] for e in log if e[0] != "event"] == ["frame", "cancel"]


def test_a_failing_transport_provider_skips_the_apology(monkeypatch, short_session_clips) -> None:
    log, client, transport, outcome = run_failing_session(
        monkeypatch, lambda s: report_provider_failure(s, "transport", "livekit", "unavailable", fatal=True)
    )
    assert outcome == "failed"
    assert transport.out.frames == []
    assert [e for e in log if e[0] == "cancel"] == [("cancel", "provider_failed")]


def test_failure_with_a_session_that_cannot_play_still_ends_the_call() -> None:
    class Plain:
        event_client = None
        session_id = uuid4()
        ending = False

        def __init__(self) -> None:
            self.failures, self.ends = [], []

        def fail(self, reason) -> None:
            self.failures.append(reason)

        async def request_end(self, reason="") -> bool:
            self.ends.append(reason)
            return True

    class Raising(Plain):
        async def play_apology(self) -> bool:
            raise RuntimeError("boom")

    class Hanging(Plain):
        async def play_apology(self) -> bool:
            await asyncio.Event().wait()

    async def go(session) -> None:
        report_provider_failure(session, "tts", "cartesia", "auth")
        for _ in range(200):
            if session.ends:
                return
            await asyncio.sleep(0.01)

    for cls, backstop in ((Plain, 8.0), (Raising, 8.0), (Hanging, 0.1)):
        import src.runtime.provider_failures as failures

        old = failures.APOLOGY_BACKSTOP_SECONDS
        failures.APOLOGY_BACKSTOP_SECONDS = backstop
        try:
            session = cls()
            asyncio.run(asyncio.wait_for(go(session), TIMEOUT))
        finally:
            failures.APOLOGY_BACKSTOP_SECONDS = old
        assert session.failures == ["provider_failed"] and session.ends == ["provider_failed"]


# -- real Pipecat output transport ---------------------------------------------------------


def test_frame_queued_on_a_real_output_transport_reaches_its_audio_write(monkeypatch, short_clips) -> None:
    """The injection path used in production, against Pipecat's own BaseOutputTransport."""

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams
    from pipecat.workers.runner import WorkerRunner

    written: list[tuple[str | None, int, int]] = []

    class Sink(BaseOutputTransport):
        async def start(self, frame) -> None:
            await super().start(frame)
            await self.set_transport_ready(frame)  # as LiveKitOutputTransport.start does

        async def write_audio_frame(self, frame) -> bool:
            written.append((frame.transport_destination, len(frame.audio), frame.sample_rate))
            return True

    sink = Sink(
        TransportParams(audio_out_enabled=True, audio_out_destinations=["translation-lane-1"])
    )

    class FakeTransport:
        def output(self):
            return sink

    cfg = config(
        mode="interpreter",
        interpreter={"languages": ["ru", "ko"]},
        participants={"alice": {"language": "ru"}, "bob": {"language": "ko"}},
        private_listening=True,
    )
    real = apology.apology_plans
    monkeypatch.setattr(
        src.runtime.session,
        "apology_plans",
        lambda c, p, l: [x for x in real(c, p, l, directory=short_clips) if x.destination == "translation-lane-1"],
    )

    async def run() -> bool:
        worker = PipelineWorker(Pipeline([sink]), params=PipelineParams(enable_metrics=False))
        started = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def _started(*_):
            started.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        run_task = asyncio.create_task(runner.run())
        await asyncio.wait_for(started.wait(), TIMEOUT)
        session = live_session(FakeTransport(), cfg)
        played = await session.play_apology()
        await worker.cancel()
        await asyncio.wait_for(run_task, TIMEOUT)
        return played

    assert asyncio.run(run()) is True
    assert written and all(d == "translation-lane-1" and r == OUTPUT_SAMPLE_RATE for d, _, r in written)
    # 0.1 s of 24 kHz 16-bit mono, chunked by the transport.
    # The TTSStoppedFrame flushes the partial last chunk (padded with silence): nothing is lost.
    assert sum(size for _, size, _ in written) >= int(0.1 * OUTPUT_SAMPLE_RATE) * 2
