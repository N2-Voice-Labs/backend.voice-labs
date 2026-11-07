"""Cinematic dubbing: profiles, the mixer's audio, the channel, and the routing rules."""

import asyncio
import math

import numpy as np
import pytest

pytest.importorskip("pipecat")
livekit_rtc = pytest.importorskip("livekit.rtc")

from src.runtime import audio_routing
from src.runtime.dubbing import (
    BLOCK,
    BLOCK_MS,
    PROFILES,
    RATE,
    DubbingChannel,
    DubbingProfile,
    DubMixer,
    parse_dubbing,
)
from src.runtime.interpreter import InterpreterSettings
from src.runtime.participants import translation_destination

from test_audio_routing import (  # noqa: F401  (fixtures and helpers)
    ALICE,
    BOB,
    FakeRemote,
    FakeRoom,
    directory_with,
    fake_rtc,
    make_router,
    run_async,
    settle,
    spec,
)

BLOCK_S = BLOCK / RATE


def tone(seconds: float, amplitude: float = 8000, hz: float = 220) -> bytes:
    n = int(seconds * RATE)
    return (amplitude * np.sin(2 * np.pi * hz * np.arange(n) / RATE)).astype(np.int16).tobytes()


def samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16)


def level_db(x: np.ndarray) -> float:
    return 20 * math.log10(max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) / 32768, 1e-9))


# -- settings ----------------------------------------------------------------------------


def test_cinematic_profile_is_balanced_with_the_original_24_db_down() -> None:
    cinematic = parse_dubbing("cinematic")
    assert cinematic == PROFILES["cinematic"]
    assert cinematic.duck_gain_db == -24
    balanced = PROFILES["balanced"]
    assert (cinematic.delay_ms, cinematic.base_gain_db, cinematic.attack_ms, cinematic.release_ms) == (
        balanced.delay_ms, balanced.base_gain_db, balanced.attack_ms, balanced.release_ms)


def test_dubbing_is_off_unless_asked_for() -> None:
    assert parse_dubbing(None) is None and parse_dubbing(False) is None
    assert parse_dubbing({"enabled": False}) is None
    base = {"languages": ["en", "ko"]}
    assert InterpreterSettings.from_mapping(base).dubbing is None


def test_dubbing_profiles_and_overrides() -> None:
    assert parse_dubbing(True) == PROFILES["balanced"]
    assert parse_dubbing("subtle") == PROFILES["subtle"]
    tuned = parse_dubbing({"profile": "present", "duck_gain_db": -26, "delay_ms": 100})
    assert (tuned.name, tuned.duck_gain_db, tuned.delay_ms, tuned.base_gain_db) == ("present", -26.0, 100.0, -6.0)
    settings = InterpreterSettings.from_mapping({"languages": ["en", "ko"], "dubbing": "balanced"})
    assert settings.dubbing == PROFILES["balanced"]


@pytest.mark.parametrize(
    "bad",
    [
        "loud",
        3,
        {"profile": "nope"},
        {"bogus": 1},
        {"delay_ms": "slow"},
        {"delay_ms": True},
        {"delay_ms": float("nan")},
        {"delay_ms": 5000},
        {"base_gain_db": 3},
        {"base_gain_db": -30, "duck_gain_db": -10},
        {"attack_ms": 0},
    ],
)
def test_bad_dubbing_settings_are_rejected(bad) -> None:
    with pytest.raises(RuntimeError):
        parse_dubbing(bad)


def test_dubbing_cannot_be_combined_with_hear_original() -> None:
    with pytest.raises(RuntimeError, match="replaces hear_original"):
        InterpreterSettings.from_mapping({"languages": ["en", "ko"], "dubbing": True, "hear_original": True})


# -- the mixer ---------------------------------------------------------------------------


def test_the_translation_alone_passes_through_unchanged_and_undelayed() -> None:
    mixer = DubMixer(PROFILES["balanced"])
    audio = samples(tone(0.1))
    mixer.push_translation(audio.tobytes())
    out = np.concatenate([samples(mixer.render(0.0)) for _ in range(len(audio) // BLOCK)])
    assert np.array_equal(out, audio[: len(out)])  # first block already holds the first sample


def test_the_original_waits_for_its_delay_and_never_the_translation() -> None:
    profile = DubbingProfile(delay_ms=200)
    mixer = DubMixer(profile)
    mixer.push_original(tone(0.1, 8000), now=0.0)
    early = samples(mixer.render(0.1))  # not due yet
    assert not early.any()
    for _ in range(4):
        mixer.render(0.2)  # let the gate open
    mixer.push_original(tone(0.1, 8000), now=1.0)
    late = np.concatenate([samples(mixer.render(1.2 + i * BLOCK_S)) for i in range(5)])
    assert late.any()


def test_the_original_is_ducked_under_the_translation_and_comes_back() -> None:
    profile = PROFILES["balanced"]
    mixer = DubMixer(profile)
    now = 0.0
    gains = []

    def step():
        nonlocal now
        mixer.push_original(tone(BLOCK_S, 6000), now)
        out = mixer.render(now)
        now += BLOCK_S
        gains.append(mixer.gain)
        return samples(out)

    for _ in range(40):  # 0.8 s of the speaker alone
        step()
    alone_db = level_db(step())
    assert alone_db == pytest.approx(level_db(samples(tone(BLOCK_S, 6000))) + profile.base_gain_db, abs=1.5)
    mixer.push_translation(tone(1.0, 6000))
    for _ in range(40):
        step()
    ducked = gains[-1]
    assert 20 * math.log10(ducked) == pytest.approx(profile.duck_gain_db, abs=1.0)
    for _ in range(100):  # the translation is over
        step()
    assert 20 * math.log10(gains[-1]) == pytest.approx(profile.base_gain_db, abs=1.0)
    # no abrupt change: the original's volume moves by well under 6 dB per 20 ms block
    steps = np.abs(np.diff(20 * np.log10(np.maximum(gains, 1e-6))))
    assert steps.max() < 6


def test_microphone_noise_between_words_stays_out_of_the_mix() -> None:
    mixer = DubMixer(DubbingProfile(delay_ms=0, gate_hold_ms=0))
    now = 0.0
    for _ in range(30):  # speech opens the gate, then it closes on hiss at -75 dBFS
        mixer.push_original(tone(BLOCK_S, 6000), now)
        mixer.render(now)
        now += BLOCK_S
    outs = []
    for _ in range(60):
        mixer.push_original(tone(BLOCK_S, 6), now)
        outs.append(samples(mixer.render(now)))
        now += BLOCK_S
    assert level_db(outs[-1]) < -90 or not outs[-1].any()


def test_a_loud_overlap_is_limited_not_clipped_or_wrapped() -> None:
    mixer = DubMixer(DubbingProfile(delay_ms=0, base_gain_db=0, duck_gain_db=0))
    now = 0.0
    peak = 0
    for _ in range(30):
        mixer.push_original(tone(BLOCK_S, 30000), now)
        mixer.push_translation(tone(BLOCK_S, 30000))
        out = samples(mixer.render(now)).astype(np.int32)
        now += BLOCK_S
        peak = max(peak, int(np.abs(out).max()))
        assert np.abs(out).max() < 32768
    assert peak > 26000  # loud, but compressed smoothly rather than wrapped around


def test_buffers_stay_bounded_and_are_released() -> None:
    mixer = DubMixer(PROFILES["balanced"])
    for i in range(600):  # 12 s of microphone with nobody rendering
        mixer.push_original(tone(0.02), now=i * 0.02)
    assert mixer.buffered_bytes <= 2 * int((0.15 + 2.0 + 0.1) * RATE)
    assert mixer.dropped_samples > 0
    mixer.push_translation(tone(1.0))
    mixer.clear()
    assert mixer.buffered_bytes == 0 and not mixer.pending


def test_an_interruption_drops_only_the_translation_that_has_not_played() -> None:
    mixer = DubMixer(PROFILES["balanced"])
    mixer.push_translation(tone(1.0))
    mixer.render(0.0)
    mixer.clear_translation()
    assert not samples(mixer.render(0.02)).any()
    mixer.push_translation(tone(0.1))
    assert len(mixer.take_translation()) == 2 * int(0.1 * RATE)


def test_mixing_is_cheap() -> None:
    import time

    mixer = DubMixer(PROFILES["balanced"])
    mixer.push_translation(tone(5.0))
    started = time.perf_counter()
    blocks = 200
    for i in range(blocks):
        mixer.push_original(tone(BLOCK_S), i * BLOCK_S)
        mixer.render(i * BLOCK_S)
    per_block_ms = (time.perf_counter() - started) / blocks * 1000
    assert per_block_ms < 2.0  # a 20 ms block; typically ~0.02 ms, the bound is for slow CI


# -- the channel (the clocked writer) --------------------------------------------------------


class RecordingSource:
    def __init__(self) -> None:
        self.frames: list[bytes] = []

    async def capture_frame(self, frame) -> None:
        self.frames.append(frame)


def make_channel(profile=None):
    source = RecordingSource()
    return source, DubbingChannel(source, profile or PROFILES["balanced"], make_frame=lambda pcm: pcm)


def test_the_channel_plays_the_translation_at_once_then_goes_quiet() -> None:
    async def scenario():
        source, channel = make_channel()
        channel.push_translation(tone(0.1, 8000))
        await asyncio.sleep(0.02)
        first = len(source.frames)  # the first blocks are written without waiting for a tick
        await asyncio.sleep(0.4)
        done = len(source.frames)
        await asyncio.sleep(0.2)
        idle = len(source.frames)
        channel.push_translation(tone(0.04, 8000))  # and wakes up again
        await asyncio.sleep(0.15)
        resumed = len(source.frames)
        await channel.aclose()
        return source, first, done, idle, resumed

    source, first, done, idle, resumed = run_async(scenario())
    assert first >= 2  # a block now and one ahead
    blocks = 100 // BLOCK_MS
    assert blocks <= done <= blocks + 4 and idle == done  # 100 ms of audio (+ a fade tail); nothing when idle
    assert resumed > idle
    played = np.concatenate([samples(b) for b in source.frames])
    assert played[: 480].any()


def test_closing_a_channel_releases_its_buffers_and_returns_unplayed_translation() -> None:
    async def scenario():
        source, channel = make_channel()
        channel.push_translation(tone(2.0))
        await asyncio.sleep(0.05)
        rest = await channel.stop()
        channel.push_translation(tone(1.0))  # ignored once closed
        return channel, rest, len(source.frames)

    channel, rest, written = run_async(scenario())
    assert len(rest) > 0 and channel.mixer.buffered_bytes == 0
    assert written < 10  # it was stopped, not left running


# -- routing -----------------------------------------------------------------------------------


def frame(seconds=0.04):
    pcm = tone(seconds)
    return livekit_rtc.AudioFrame(pcm, RATE, 1, len(pcm) // 2)


async def attached_router(fake_rtc_unused=None):
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
    router, _ = make_router(directory)
    room = FakeRoom()
    room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
    room.remote_participants[BOB] = FakeRemote(BOB, mic_sid="MIC_B")
    await router.attach(room, sample_rate=24000)
    await settle(router)
    return directory, router


def test_without_dubbing_frames_take_the_direct_path(fake_rtc) -> None:
    async def scenario():
        directory, router = await attached_router()
        item = frame()
        await router.write(translation_destination(directory.lane_of(BOB)), item)
        await router.feed_original(ALICE, tone(0.02), RATE)  # ignored when off
        return router, item

    router, item = run_async(scenario())
    assert router._tracks[BOB].source.frames == [item]  # untouched, same object
    assert router._channels == {} and router.dubbing is None


def test_each_listener_gets_only_the_other_persons_voice(fake_rtc) -> None:
    async def scenario():
        directory, router = await attached_router()
        await router.set_dubbing(PROFILES["balanced"])
        to_bob = translation_destination(directory.lane_of(BOB))
        await router.write(to_bob, frame())  # creates Bob's mix
        await router.write(translation_destination(directory.lane_of(ALICE)), frame())
        await router.feed_original(ALICE, tone(0.04), RATE)  # Alice speaks
        mixers = {who: router._channels[who].mixer for who in (ALICE, BOB)}
        sizes = {who: m._original_samples for who, m in mixers.items()}
        await router.close()
        return sizes, router

    sizes, router = run_async(scenario())
    assert sizes[BOB] > 0  # Bob hears Alice's original
    assert sizes[ALICE] == 0  # Alice never hears her own voice back
    assert router._channels == {}


def test_the_original_is_resampled_to_the_track_rate(fake_rtc) -> None:
    async def scenario():
        directory, router = await attached_router()
        await router.set_dubbing(PROFILES["balanced"])
        await router.write(translation_destination(directory.lane_of(BOB)), frame())
        pcm16k = (6000 * np.sin(2 * np.pi * 200 * np.arange(1600) / 16000)).astype(np.int16).tobytes()  # 100 ms
        for i in range(0, len(pcm16k), 320):
            await router.feed_original(ALICE, pcm16k[i : i + 320], 16000)
        queued = router._channels[BOB].mixer._original_samples
        await router.close()
        return queued

    queued = run_async(scenario())
    assert 0.5 * 2400 <= queued <= 1.1 * 2400  # about 100 ms at 24 kHz (the resampler holds back its start-up filter delay)


def test_switching_dubbing_off_flushes_the_sentence_in_progress(fake_rtc) -> None:
    async def scenario():
        directory, router = await attached_router()
        await router.set_dubbing(PROFILES["balanced"])
        to_bob = translation_destination(directory.lane_of(BOB))
        await router.write(to_bob, frame(1.0))
        await router.set_dubbing(None)
        source = router._tracks[BOB].source
        total = sum(len(f.data) for f in source.frames)
        direct = frame()
        await router.write(to_bob, direct)
        return router, source, total, direct

    router, source, total, direct = run_async(scenario())
    assert router._channels == {} and router.dubbing is None
    assert total > 0 and source.frames[-1] is direct  # back on the direct path


def test_an_interruption_clears_the_dubbed_translation(fake_rtc) -> None:
    async def scenario():
        directory, router = await attached_router()
        await router.set_dubbing(PROFILES["balanced"])
        await router.write(translation_destination(directory.lane_of(BOB)), frame(3.0))
        router.clear()
        pending = router._channels[BOB].mixer._translation_samples
        await router.close()
        return pending

    assert run_async(scenario()) == 0


def test_feeding_the_original_never_raises(fake_rtc, monkeypatch) -> None:
    async def scenario():
        directory, router = await attached_router()
        await router.set_dubbing(PROFILES["balanced"])
        await router.write(translation_destination(directory.lane_of(BOB)), frame())
        await router.feed_original("stranger", tone(0.02), RATE)  # not in the call
        await router.feed_original(ALICE, b"", RATE)
        monkeypatch.setattr(audio_routing.DubbingChannel, "push_original", lambda *a: 1 / 0)
        await router.feed_original(ALICE, tone(0.02), RATE)  # a failure inside is contained
        await router.close()

    run_async(scenario())


# -- the pipeline ----------------------------------------------------------------------------------

from test_audio_routing import build_livekit_pipeline, light_models, pipeline_config  # noqa: E402,F401


def test_the_pipeline_builds_the_same_way_with_dubbing_off_and_on(light_models, monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")

    def build(**interpreter):
        config = pipeline_config(
            mode="interpreter",
            interpreter={"languages": ["en", "ko"], "direction": "one_way", **interpreter},
            participants={ALICE: spec("en")},
        )
        transport, worker = build_livekit_pipeline(config)
        return transport, [type(p).__name__ for p in worker._pipeline._processors]

    off_transport, off_names = build()
    on_transport, on_names = build(dubbing="subtle")
    assert off_transport.output()._router.dubbing is None  # default: the direct path
    assert on_transport.output()._router.dubbing == PROFILES["subtle"]
    assert on_names == off_names  # same pipeline: the setting only reaches the router


def test_the_input_tap_copies_audio_after_the_lane_has_its_own(monkeypatch) -> None:
    """The lane's frame (VAD, STT) is pushed unchanged; dubbing only gets a copy of it."""

    from src.runtime.participant_lanes import PerParticipantLiveKitInput

    seen_by_router, pushed = [], []

    class Router:
        dubbing = PROFILES["balanced"]

        async def feed_original(self, speaker, pcm, rate):
            seen_by_router.append((speaker, pcm, rate))

    class Client:
        async def get_next_audio_frame(self):
            for _ in range(10):  # the stream resampler holds back its first chunk while its filter fills
                data = (3000 * np.sin(np.arange(480) / 7)).astype(np.int16)
                event = type("E", (), {"frame": type("F", (), {"data": data, "sample_rate": 48000, "num_channels": 1})()})()
                yield event, ALICE

    async def scenario(router):
        node = PerParticipantLiveKitInput.__new__(PerParticipantLiveKitInput)
        node._participant_resamplers = {}
        node._router = router
        node._client = Client()
        node._params = type("P", (), {"audio_in_sample_rate": 16000})()
        node._sample_rate = 16000

        async def push_audio_frame(frame):
            pushed.append(frame)

        node.push_audio_frame = push_audio_frame
        await node._audio_in_task_handler()

    asyncio.run(scenario(Router()))
    assert pushed and seen_by_router
    assert [c[1] for c in seen_by_router] == [f.audio for f in pushed]  # a copy of exactly what the lane got
    assert all(c[0] == ALICE and c[2] == 16000 for c in seen_by_router)
    seen_by_router.clear()
    pushed.clear()
    off = Router()
    off.dubbing = None
    asyncio.run(scenario(off))
    assert pushed and not seen_by_router  # off: nothing is taken


def test_a_silent_microphone_keeps_the_mixer_and_the_track_idle() -> None:
    async def scenario():
        source, channel = make_channel()
        for _ in range(100):  # a second of microphone hiss, 10 ms frames
            channel.push_original((np.random.default_rng(0).standard_normal(240) * 4).astype(np.int16).tobytes())
            await asyncio.sleep(0.01)
        silent_frames = len(source.frames)
        buffered = channel.mixer.buffered_bytes
        channel.push_original(tone(0.02, 6000))  # then the person speaks
        await asyncio.sleep(0.5)
        spoke = len(source.frames)
        await channel.aclose()
        return silent_frames, buffered, spoke

    silent_frames, buffered, spoke = run_async(scenario())
    assert silent_frames == 0 and buffered == 0  # nothing queued, nothing written
    assert spoke > 0
