"""Per-listener audio routing and the audio ownership contract (RM-001, RM-002).

Contract (docs/BACKEND_ROADMAP.md): originals reach Runtime only, ``A_translation`` only B,
``B_translation`` only A, nobody hears their own translation, ``hear_original`` opts the
two people into each other's original microphone.
"""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")
livekit_rtc = pytest.importorskip("livekit.rtc")

from pipecat.frames.frames import (
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.livekit.transport import LiveKitParams

from src.runtime import audio_routing
from src.runtime.audio_routing import (
    AudioRoutingError,
    ListenerAudioRouter,
    LiveKitApiSettings,
)
from src.runtime.interpreter import InterpreterSettings, TranslationProcessor
from src.runtime.participant_lanes import (
    ListenerRoutedLiveKitOutput,
    PerParticipantLiveKitTransport,
    TTSLaneGate,
)
from src.runtime.participants import (
    ParticipantDirectory,
    translation_destination,
    translation_track_name,
)
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.session import RuntimeSession

ALICE = "browser-1"  # speaks English
BOB = "guest-2"  # speaks Korean
OBSERVER = "observer-3"
DEFAULT_VOICE = "agent-voice"
AUDIO = livekit_rtc.TrackKind.KIND_AUDIO


def spec(language, voice_id=None, display_name=None):
    return {"language": language, "voice_id": voice_id, "display_name": display_name}


def directory_with(*entries, pair=("en", "ko")):
    return ParticipantDirectory.from_mapping(pair, dict(entries))


# -- participant context ------------------------------------------------------------


def test_context_describes_each_participant_and_their_listener() -> None:
    directory = directory_with((ALICE, spec("en", "alice-voice", "Alice")), (BOB, spec("ko")))
    alice, bob = directory.contexts()

    assert (alice.identity, alice.lane, alice.display_name) == (ALICE, 0, "Alice")
    assert (alice.source_language, alice.target_language, alice.voice_id) == ("en", "ko", "alice-voice")
    assert (alice.output_track, alice.output_destination) == (
        "translation-for-browser-1",
        "translation-lane-0",
    )
    # A's words are heard by B (on B's track), B's by A.
    assert (alice.listener, bob.listener) == (BOB, ALICE)
    assert (bob.source_language, bob.target_language, bob.voice_id) == ("ko", "en", None)
    assert bob.output_track == translation_track_name(BOB) == "translation-for-guest-2"
    assert bob.output_destination == translation_destination(1)


def test_listener_follows_presence_once_it_is_reported() -> None:
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
    assert directory.listener_for(ALICE) == BOB  # presence unknown: everyone counts
    directory.set_present({ALICE})
    assert directory.listener_for(ALICE) is None  # Bob is not in the room
    assert directory.listener_for(BOB) == ALICE
    assert not directory.is_present(BOB) and not directory.is_present("stranger")
    directory.set_present({ALICE, BOB})
    assert directory.listener_for(ALICE) == BOB


def test_single_participant_has_no_listener() -> None:
    directory = directory_with((ALICE, spec("en")))
    (alice,) = directory.contexts()
    assert alice.listener is None and alice.target_language == "ko"


# -- the translator tags speech with the listener's lane --------------------------------


class FakeSession:
    def __init__(self) -> None:
        self.event_client = object()
        self.session_id = uuid4()
        self.events: list[tuple[str, dict]] = []

    def event(self, event_type, payload=None):
        return (event_type, payload or {})

    def enqueue_event(self, event) -> None:
        self.events.append(event)


def translator(directory, *, route=True, session=None):
    async def translate(messages):
        return "T:" + messages[-1]["content"]

    return TranslationProcessor(
        settings=InterpreterSettings(languages=directory.languages, direction="auto"),
        translate=translate,
        model="test-model",
        session=session,
        initial_tts_language=directory.languages[1],
        participants=directory,
        default_voice_id=DEFAULT_VOICE,
        route_by_listener=route,
    )


def turn(text, user_id):
    started, stopped = UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()
    started.metadata["user_id"] = stopped.metadata["user_id"] = user_id
    return [
        started,
        SleepFrame(sleep=0.02),
        TranscriptionFrame(text, user_id, "now", None, finalized=True),
        SleepFrame(sleep=0.02),
        stopped,
        SleepFrame(sleep=0.06),
    ]


def run(processor, frames):
    return asyncio.run(run_test(processor, frames_to_send=frames, expected_down_frames=None))


def routed(down):
    out = []
    for frame in down:
        lane = frame.metadata.get("listener_lane") if isinstance(
            frame, (TTSSpeakFrame, TTSUpdateSettingsFrame)
        ) else None
        if isinstance(frame, TTSUpdateSettingsFrame):
            delta = {
                name: getattr(frame.delta, name)
                for name in ("language", "voice")
                if isinstance(getattr(frame.delta, name), str)
            }
            out.append(("settings", lane, delta))
        elif isinstance(frame, TTSSpeakFrame):
            out.append(("say", lane, frame.text))
    return out


def test_each_translation_goes_to_the_other_participants_lane_only() -> None:
    directory = directory_with(
        (ALICE, spec("en", "alice-voice")), (BOB, spec("ko", "bob-voice"))
    )
    frames = turn("Hello.", ALICE) + turn("안녕.", BOB) + turn("Again.", ALICE)
    down, _ = run(translator(directory), frames)

    assert routed(down) == [
        # Alice speaks: Korean, Alice's voice, on Bob's lane (1). Lane 1 already speaks ko.
        ("settings", 1, {"voice": "alice-voice"}),
        ("say", 1, "T:Hello."),
        # Bob speaks: English, Bob's voice, on Alice's lane (0). Lane 0 already speaks en.
        ("settings", 0, {"voice": "bob-voice"}),
        ("say", 0, "T:안녕."),
        # Alice again: lane 1 keeps language and voice, nothing to resend.
        ("say", 1, "T:Again."),
    ]


def test_nobody_hears_their_own_translation() -> None:
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
    down, _ = run(translator(directory), turn("one", ALICE) + turn("둘", BOB))
    lanes_by_speaker = {text: lane for kind, lane, text in routed(down) if kind == "say"}
    assert lanes_by_speaker == {"T:one": directory.lane_of(BOB), "T:둘": directory.lane_of(ALICE)}


def test_lane_voice_and_language_follow_the_speaker_per_lane() -> None:
    directory = directory_with((ALICE, spec("en", "alice-voice")), (BOB, spec("ko")))
    down, _ = run(translator(directory), turn("a", ALICE) + turn("ㄱ", BOB) + turn("b", ALICE))
    settings = [(lane, delta) for kind, lane, delta in routed(down) if kind == "settings"]
    # Bob has no voice of his own: the agent voice is already on lane 0, nothing is sent.
    assert settings == [(1, {"voice": "alice-voice"})]


def test_speech_before_the_second_person_joins_is_not_spoken_but_is_reported() -> None:
    directory = directory_with((ALICE, spec("en")))
    session = FakeSession()
    down, _ = run(translator(directory, session=session), turn("anyone there?", ALICE))
    assert routed(down) == []  # nobody to hear it (and Alice must not hear herself)
    texts = [p["content"] for kind, p in session.events if kind == "TranscriptFinal"]
    assert texts == ["anyone there?", "T:anyone there?"]


def test_late_joiner_hears_only_speech_after_joining() -> None:
    directory = directory_with((ALICE, spec("en", "alice-voice")))

    class Join(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSSpeakFrame) and frame.text == "join-now":
                await directory.update(BOB, spec("ko", "bob-voice"))
                return
            await self.push_frame(frame, direction)

    pipeline = Pipeline([Join(), translator(directory)])
    frames = turn("before", ALICE) + [TTSSpeakFrame("join-now")] + [SleepFrame(sleep=0.02)] + turn("after", ALICE)
    down, _ = asyncio.run(run_test(pipeline, frames_to_send=frames, expected_down_frames=None))
    assert [(kind, lane, text) for kind, lane, text in routed(down) if kind == "say" and lane is not None] == [
        ("say", 1, "T:after")
    ]


def test_listener_who_left_the_room_is_not_spoken_to() -> None:
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
    directory.set_present({ALICE})
    down, _ = run(translator(directory), turn("hello?", ALICE) + turn("네", BOB))
    # Alice's speech has no listener; Bob (who is not present) still gets heard by Alice.
    assert routed(down) == [("say", 0, "T:네")]


def test_without_route_by_listener_frames_are_untagged() -> None:
    directory = directory_with((ALICE, spec("en", "alice-voice")), (BOB, spec("ko")))
    down, _ = run(translator(directory, route=False), turn("Hello.", ALICE))
    assert routed(down) == [("settings", None, {"voice": "alice-voice"}), ("say", None, "T:Hello.")]


# -- parallel TTS lanes: audio is stamped with the listener's destination ------------------


class FakeLaneTTS(FrameProcessor):
    """Stands in for a TTS service: speech becomes audio for its transport destination."""

    def __init__(self, destination) -> None:
        super().__init__()
        self.destination = destination
        self.said: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            self.said.append(frame.text)
            audio = OutputAudioRawFrame(b"\0\0" * 160, 16000, 1)
            audio.transport_destination = self.destination
            audio.metadata["text"] = frame.text
            await self.push_frame(audio)
            return
        if isinstance(frame, TTSUpdateSettingsFrame):
            return
        await self.push_frame(frame, direction)


def test_translation_audio_reaches_only_the_listeners_destination() -> None:
    directory = directory_with(
        (ALICE, spec("en", "alice-voice")), (BOB, spec("ko", "bob-voice"))
    )
    lanes = [FakeLaneTTS(translation_destination(lane)) for lane in range(2)]
    pipeline = Pipeline(
        [
            translator(directory),
            ParallelPipeline(*[[TTSLaneGate(lane), lanes[lane]] for lane in range(2)]),
        ]
    )
    frames = turn("Hello.", ALICE) + turn("안녕.", BOB)
    down, _ = asyncio.run(run_test(pipeline, frames_to_send=frames, expected_down_frames=None))

    audio = [f for f in down if isinstance(f, OutputAudioRawFrame)]
    assert [(f.transport_destination, f.metadata["text"]) for f in audio] == [
        ("translation-lane-1", "T:Hello."),  # Alice's English-to-Korean goes to Bob
        ("translation-lane-0", "T:안녕."),  # Bob's Korean-to-English goes to Alice
    ]
    assert lanes[1].said == ["T:Hello."] and lanes[0].said == ["T:안녕."]


# -- the router: tracks, permissions and subscriptions against a fake LiveKit ------------------


class FakeSource:
    def __init__(self, sample_rate, channels, queue_size_ms=1000) -> None:
        self.frames: list = []
        self.closed = False

    async def capture_frame(self, frame) -> None:
        self.frames.append(frame)

    async def aclose(self) -> None:
        self.closed = True


class FakePublication:
    def __init__(self, sid, name="mic", kind=AUDIO) -> None:
        self.sid, self.name, self.kind = sid, name, kind


class FakeRemote:
    def __init__(self, identity, *, kind=0, mic_sid=None) -> None:
        self.identity, self.kind = identity, kind
        self.track_publications = {}
        if mic_sid:
            self.track_publications[mic_sid] = FakePublication(mic_sid)


class FakeLocalParticipant:
    def __init__(self) -> None:
        self.published = []  # (name, sid)
        self.unpublished = []
        self.permission_calls = []
        self._counter = 0

    async def publish_track(self, track, options):
        self._counter += 1
        sid = f"TR_{track.name}"
        self.published.append((track.name, sid))
        return SimpleNamespace(sid=sid)

    async def unpublish_track(self, sid) -> None:
        self.unpublished.append(sid)

    def set_track_subscription_permissions(self, *, allow_all_participants, participant_permissions=None):
        assert allow_all_participants is False
        self.permission_calls.append(
            {p.participant_identity: list(p.allowed_track_sids) for p in participant_permissions or []}
            | {"__allow_all__": allow_all_participants}
        )


class FakeRoom:
    name = "voice-room"

    def __init__(self) -> None:
        self.remote_participants: dict[str, FakeRemote] = {}
        self.local_participant = FakeLocalParticipant()
        self.handlers: dict[str, list] = {}

    def on(self, event, callback) -> None:
        self.handlers.setdefault(event, []).append(callback)

    def fire(self, event, *args) -> None:
        for callback in self.handlers.get(event, []):
            callback(*args)

    def join(self, remote: FakeRemote) -> None:
        self.remote_participants[remote.identity] = remote
        self.fire("participant_connected", remote)

    def leave(self, identity) -> None:
        remote = self.remote_participants.pop(identity)
        self.fire("participant_disconnected", remote)

    @property
    def last_permissions(self):
        calls = self.local_participant.permission_calls
        return {k: v for k, v in calls[-1].items() if k != "__allow_all__"} if calls else None


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], bool]] = []
        self.fail = False
        self.closed = False
        self.room = self

    async def update_subscriptions(self, request) -> None:
        if self.fail:
            raise RuntimeError("livekit unavailable")
        assert request.room == "voice-room"
        self.calls.append((request.identity, tuple(request.track_sids), request.subscribe))

    async def aclose(self) -> None:
        self.closed = True

    def unsubscribed(self, identity) -> set[str]:
        return {s for who, sids, sub in self.calls if who == identity and not sub for s in sids}

    def subscribed(self, identity) -> set[str]:
        return {s for who, sids, sub in self.calls if who == identity and sub for s in sids}


@pytest.fixture
def fake_rtc(monkeypatch):
    created: list = []

    def create_audio_track(name, source):
        track = SimpleNamespace(name=name, source=source)
        created.append(track)
        return track

    monkeypatch.setattr(audio_routing.rtc, "AudioSource", FakeSource)
    monkeypatch.setattr(audio_routing.rtc.LocalAudioTrack, "create_audio_track", staticmethod(create_audio_track))
    return created


def make_router(directory, *, hear_original=False):
    api = FakeApi()
    router = ListenerAudioRouter(
        directory,
        hear_original=hear_original,
        api_settings=LiveKitApiSettings("ws://localhost:7880", "key", "secret"),
        api_factory=lambda _settings: api,
        burst=(0.0,),
    )
    return router, api


async def settle(router) -> None:
    runner = router._runner
    if runner is not None:
        await runner


def run_async(coro):
    return asyncio.run(coro)


def test_tracks_are_closed_by_default_then_opened_for_one_listener_each(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        router, api = make_router(directory)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        room.remote_participants[BOB] = FakeRemote(BOB, mic_sid="MIC_B")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        return room, api, router

    room, api, router = run_async(scenario())
    permission_calls = room.local_participant.permission_calls
    # 1. nobody may subscribe to anything before a track exists ...
    assert permission_calls[0] == {"__allow_all__": False}
    # 2. ... then each listener may subscribe to their own translation track only.
    assert room.last_permissions == {
        ALICE: [f"TR_translation-for-{ALICE}"],
        BOB: [f"TR_translation-for-{BOB}"],
    }
    assert {name for name, _ in room.local_participant.published} == {
        f"translation-for-{ALICE}",
        f"translation-for-{BOB}",
    }
    # Neither person is allowed the other's translation track (rule 1).
    assert f"TR_translation-for-{BOB}" not in room.last_permissions[ALICE]
    assert f"TR_translation-for-{ALICE}" not in room.last_permissions[BOB]
    # Originals are Runtime only: each is unsubscribed from the other's microphone.
    assert api.unsubscribed(ALICE) == {"MIC_B"} and api.unsubscribed(BOB) == {"MIC_A"}
    # Each listener is also (re)subscribed to their translation track server-side.
    assert api.subscribed(ALICE) == {f"TR_translation-for-{ALICE}"}
    assert router._directory.listener_for(ALICE) == BOB


def test_hear_original_keeps_the_other_persons_microphone_but_not_for_observers(fake_rtc) -> None:
    async def scenario(hear_original):
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        router, api = make_router(directory, hear_original=hear_original)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        room.remote_participants[BOB] = FakeRemote(BOB, mic_sid="MIC_B")
        room.remote_participants[OBSERVER] = FakeRemote(OBSERVER)
        room.remote_participants["egress"] = FakeRemote(
            "egress", kind=livekit_rtc.ParticipantKind.PARTICIPANT_KIND_EGRESS
        )
        await router.attach(room, sample_rate=24000)
        await settle(router)
        return api

    off = run_async(scenario(False))
    assert off.unsubscribed(ALICE) == {"MIC_B"} and off.unsubscribed(BOB) == {"MIC_A"}
    assert off.unsubscribed(OBSERVER) == {"MIC_A", "MIC_B"}
    assert off.unsubscribed("egress") == set()  # trusted server-side participants keep working

    on = run_async(scenario(True))
    assert on.unsubscribed(ALICE) == set() and on.unsubscribed(BOB) == set()
    # Originals are still Runtime-only for everyone who is not one of the two people.
    assert on.unsubscribed(OBSERVER) == {"MIC_A", "MIC_B"}


def test_late_join_publishes_the_new_listeners_track_and_unsubscribes_both(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")))
        router, api = make_router(directory)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        before = (room.last_permissions, list(api.calls), directory.listener_for(ALICE))

        # Bob is added to the mapping (Platform call) and joins the LiveKit room.
        await directory.update(BOB, spec("ko", "bob-voice"))
        room.join(FakeRemote(BOB, mic_sid="MIC_B"))
        await settle(router)
        return room, api, router, directory, before

    room, api, router, directory, before = run_async(scenario())
    permissions, calls, listener = before
    assert permissions == {ALICE: [f"TR_translation-for-{ALICE}"]}
    assert listener is None
    assert not any(not subscribe for _, _, subscribe in calls)  # nothing to hide from Alice yet

    assert room.last_permissions == {
        ALICE: [f"TR_translation-for-{ALICE}"],
        BOB: [f"TR_translation-for-{BOB}"],
    }
    assert api.unsubscribed(ALICE) == {"MIC_B"} and api.unsubscribed(BOB) == {"MIC_A"}
    assert directory.listener_for(ALICE) == BOB and directory.listener_for(BOB) == ALICE


def test_directory_entry_added_before_the_person_connects_publishes_nothing_yet(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")))
        router, api = make_router(directory)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        await directory.update(BOB, spec("ko"))  # Platform told us, Bob has not connected
        await settle(router)
        return room, directory

    room, directory = run_async(scenario())
    assert [name for name, _ in room.local_participant.published] == [f"translation-for-{ALICE}"]
    assert directory.listener_for(ALICE) is None  # not in the room: not a listener yet


def test_leaving_unpublishes_the_track_and_a_rejoin_gets_a_fresh_one(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        router, api = make_router(directory)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        room.remote_participants[BOB] = FakeRemote(BOB, mic_sid="MIC_B")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        bob_source = router._tracks[BOB].source

        room.leave(BOB)
        await settle(router)
        after_leave = (room.last_permissions, directory.listener_for(ALICE), bob_source.closed)

        room.join(FakeRemote(BOB, mic_sid="MIC_B2"))
        await settle(router)
        return room, api, after_leave

    room, api, (permissions, listener, closed) = run_async(scenario())
    assert room.local_participant.unpublished == [f"TR_translation-for-{BOB}"]
    assert permissions == {ALICE: [f"TR_translation-for-{ALICE}"]}  # Bob can hear nothing now
    assert listener is None and closed
    names = [name for name, _ in room.local_participant.published]
    assert names.count(f"translation-for-{BOB}") == 2
    assert api.unsubscribed(ALICE) >= {"MIC_B", "MIC_B2"}  # the new microphone is hidden too
    assert api.unsubscribed(BOB) == {"MIC_A"}


def test_republished_original_track_is_unsubscribed_again(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        router, api = make_router(directory)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        bob = room.remote_participants[BOB] = FakeRemote(BOB, mic_sid="MIC_B")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        # Bob mutes/reconnects his microphone: a new track sid LiveKit auto-subscribes Alice to.
        bob.track_publications = {"MIC_B_NEW": FakePublication("MIC_B_NEW")}
        room.fire("track_published", SimpleNamespace(sid="MIC_B_NEW"), bob)
        await settle(router)
        return api

    api = run_async(scenario())
    assert api.unsubscribed(ALICE) == {"MIC_B", "MIC_B_NEW"}


def test_policy_is_reapplied_in_a_burst_because_the_server_can_race_the_autosubscription(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        api = FakeApi()
        router = ListenerAudioRouter(
            directory,
            api_settings=LiveKitApiSettings("ws://x", "k", "s"),
            api_factory=lambda _s: api,
            burst=(0.0, 0.01, 0.01),
        )
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        room.remote_participants[BOB] = FakeRemote(BOB, mic_sid="MIC_B")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        return api

    api = run_async(scenario())
    unsubscribe_alice = [c for c in api.calls if c[0] == ALICE and not c[2]]
    assert len(unsubscribe_alice) >= 4  # initial sync + attach burst of three


def test_audio_frames_go_to_the_listeners_own_track(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        router, _ = make_router(directory)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        room.remote_participants[BOB] = FakeRemote(BOB, mic_sid="MIC_B")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        to_bob = await router.write(translation_destination(directory.lane_of(BOB)), "frame-for-bob")
        to_alice = await router.write(translation_destination(directory.lane_of(ALICE)), "frame-for-alice")
        unknown = await router.write("translation-lane-7", "stray")
        garbage = await router.write("somewhere-else", "stray")
        return router, (to_bob, to_alice, unknown, garbage)

    router, results = run_async(scenario())
    assert results == (True, True, True, True)
    assert router._tracks[BOB].source.frames == ["frame-for-bob"]
    assert router._tracks[ALICE].source.frames == ["frame-for-alice"]


def test_audio_for_a_listener_who_is_not_in_the_room_is_dropped(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        router, _ = make_router(directory)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        ok = await router.write(translation_destination(1), "frame-for-absent-bob")
        return router, ok

    router, ok = run_async(scenario())
    assert ok and list(router._tracks) == [ALICE] and router._tracks[ALICE].source.frames == []


def test_server_api_failure_is_logged_not_raised_and_retried_by_the_next_sync(fake_rtc, caplog) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        router, api = make_router(directory)
        api.fail = True
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        room.remote_participants[BOB] = FakeRemote(BOB, mic_sid="MIC_B")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        failed_calls = len(api.calls)
        api.fail = False
        await router.sync()
        return api, failed_calls

    with caplog.at_level("WARNING"):
        api, failed_calls = run_async(scenario())
    assert failed_calls == 0 and "update_subscriptions" in caplog.text
    assert api.unsubscribed(ALICE) == {"MIC_B"} and api.unsubscribed(BOB) == {"MIC_A"}


def test_close_releases_the_tracks_and_the_api_client(fake_rtc) -> None:
    async def scenario():
        directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
        router, api = make_router(directory)
        room = FakeRoom()
        room.remote_participants[ALICE] = FakeRemote(ALICE, mic_sid="MIC_A")
        await router.attach(room, sample_rate=24000)
        await settle(router)
        sources = [entry.source for entry in router._tracks.values()]
        await router.close()
        await router.sync()  # a closed router does nothing
        return api, sources, router

    api, sources, router = run_async(scenario())
    assert all(source.closed for source in sources) and api.closed and not router._tracks


def test_api_settings_come_from_the_environment(monkeypatch) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AudioRoutingError):
        LiveKitApiSettings.from_environment("ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    settings = LiveKitApiSettings.from_environment("wss://example.livekit.cloud")
    assert settings.http_url == "https://example.livekit.cloud"
    assert LiveKitApiSettings("ws://localhost:7880", "k", "s").http_url == "http://localhost:7880"


# -- the output transport ---------------------------------------------------------------------


def test_output_routes_destination_frames_to_the_router_and_keeps_the_default_track() -> None:
    class Router:
        def __init__(self) -> None:
            self.writes = []

        async def write(self, destination, frame):
            self.writes.append((destination, frame))
            return True

    class Client:
        def __init__(self) -> None:
            self.published = []

        async def publish_audio(self, frame):
            self.published.append(frame)
            return True

    router, client = Router(), Client()
    output = ListenerRoutedLiveKitOutput(
        SimpleNamespace(), client, LiveKitParams(audio_out_enabled=True), router=router
    )
    output._sample_rate = 16000

    default = OutputAudioRawFrame(b"\1\0" * 160, 16000, 1)
    lane = OutputAudioRawFrame(b"\2\0" * 160, 16000, 1)
    lane.transport_destination = "translation-lane-1"

    async def go():
        assert await output.write_audio_frame(default)
        assert await output.write_audio_frame(lane)

    asyncio.run(go())
    assert len(client.published) == 1 and [d for d, _ in router.writes] == ["translation-lane-1"]
    assert router.writes[0][1].data.tobytes() == b"\2\0" * 160


def test_transport_without_a_router_keeps_the_stock_output() -> None:
    transport = PerParticipantLiveKitTransport("ws://x", "t", "room", params=LiveKitParams())
    assert type(transport.output()).__name__ == "LiveKitOutputTransport"
    routed_transport = PerParticipantLiveKitTransport(
        "ws://x", "t", "room", params=LiveKitParams(), router=object()
    )
    assert isinstance(routed_transport.output(), ListenerRoutedLiveKitOutput)


# -- pipeline builder and configuration -----------------------------------------------------


@pytest.fixture
def light_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer)
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)


def pipeline_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        "ws://localhost:7880", "token", "room", "deepgram", "openai", "cartesia",
        DEFAULT_VOICE, "prompt", **kwargs,
    )  # fmt: skip


def top_level(worker):
    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    return [p for p in pipeline._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}]


def test_two_person_pipeline_has_one_tts_per_listener_and_routed_output(light_models, monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    config = pipeline_config(
        mode="interpreter",
        interpreter={"languages": ["en", "ko"], "direction": "one_way", "hear_original": True},
        participants={ALICE: spec("en", "alice-voice")},
    )
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    session.participants = ParticipantDirectory.from_mapping(("en", "ko"), config.participants)
    transport, worker = build_livekit_pipeline(config, session)

    processors = top_level(worker)
    assert type(transport).__name__ == "PerParticipantLiveKitTransport"
    assert type(processors[-1]).__name__ == "ListenerRoutedLiveKitOutput"
    assert processors[2].__class__.__name__ == "TranslationProcessor" and processors[2]._route
    tts_lanes = processors[3].processors
    assert len(tts_lanes) == 2
    destinations, languages = [], []
    for lane in tts_lanes:
        names = [type(p).__name__ for p in lane._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}]
        assert names == ["TTSLaneGate", "CartesiaTTSService"]
        tts = next(p for p in lane._processors if type(p).__name__ == "CartesiaTTSService")
        destinations.append(tts._transport_destination)
        languages.append(tts._settings.language)
    assert destinations == ["translation-lane-0", "translation-lane-1"]
    assert languages == ["en", "ko"]  # each lane speaks its listener's language
    router = transport.output()._router
    assert router.hear_original is True and router._directory is session.participants
    assert transport.output()._params.audio_out_destinations == destinations


def test_hear_original_defaults_to_off(light_models, monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    config = pipeline_config(
        mode="interpreter",
        interpreter={"languages": ["en", "ko"], "direction": "one_way"},
        participants={ALICE: spec("en")},
    )
    transport, _ = build_livekit_pipeline(config)
    assert transport.output()._router.hear_original is False


def test_two_person_pipeline_refuses_to_run_without_the_means_to_enforce(light_models, monkeypatch) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    config = pipeline_config(
        mode="interpreter",
        interpreter={"languages": ["en", "ko"], "direction": "one_way"},
        participants={ALICE: spec("en")},
    )
    with pytest.raises(AudioRoutingError):
        build_livekit_pipeline(config)


def test_single_participant_interpreter_pipeline_is_unchanged(light_models, monkeypatch) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    config = pipeline_config(
        mode="interpreter", interpreter={"languages": ["en", "ru"], "direction": "auto"}
    )
    transport, worker = build_livekit_pipeline(config)
    assert type(transport).__name__ == "LiveKitTransport"
    assert [type(p).__name__ for p in top_level(worker)] == [
        "LiveKitInputTransport", "VADProcessor", "DeepgramSTTService", "UserTurnProcessor",
        "TranslationProcessor", "CartesiaTTSService", "LiveKitOutputTransport",
    ]  # fmt: skip
    tts = top_level(worker)[5]
    assert tts._transport_destination is None  # one track, heard by the caller (rule 4)
    assert transport._params.audio_out_destinations == []
    assert top_level(worker)[4]._route is False


def test_hear_original_setting_is_validated() -> None:
    base = {"languages": ["en", "ko"], "direction": "one_way"}
    assert InterpreterSettings.from_mapping(base).hear_original is False
    assert InterpreterSettings.from_mapping({**base, "hear_original": True}).hear_original is True
    with pytest.raises(RuntimeError, match="hear_original"):
        InterpreterSettings.from_mapping({**base, "hear_original": "yes"})


def test_hear_original_travels_in_the_session_contract(monkeypatch) -> None:
    from src.runtime import api

    interpreter = {"languages": ["en", "ko"], "direction": "one_way", "hear_original": True}
    request = api.SessionStart(
        session_id=uuid4(), conversation_id=uuid4(), agent_id=uuid4(), room_name="room",
        mode="interpreter", interpreter=interpreter,
        participants={ALICE: api.ParticipantSpec(language="en")},
    )  # fmt: skip
    assert request.model_dump(mode="json")["interpreter"]["hear_original"] is True
    for name, value in {
        "LIVEKIT_URL": "ws://localhost:7880", "DEEPGRAM_API_KEY": "d", "OPENAI_API_KEY": "o",
        "CARTESIA_API_KEY": "c", "CARTESIA_VOICE_ID": "v",
    }.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    config = PipelineConfig.from_environment(
        {
            "room_name": "room", "livekit_token": "t", "mode": "interpreter",
            "interpreter": interpreter, "participants": {ALICE: spec("en")},
        }
    )  # fmt: skip
    assert config.interpreter["hear_original"] is True
