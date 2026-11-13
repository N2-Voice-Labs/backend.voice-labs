"""Two-person interpreter sessions (TR-020): mapping, routing, voices and the API."""

import asyncio
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    STTUpdateSettingsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test

from src.runtime import api
from src.runtime.interpreter import InterpreterSettings, TranslationProcessor
from src.runtime.participant_lanes import LaneTurnTagger, ParticipantLaneGate
from src.runtime.participants import (
    MAX_PARTICIPANTS,
    ParticipantDirectory,
    ParticipantError,
)
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.session import RuntimeSession

ALICE = "browser-1"
BOB = "guest-2"
DEFAULT_VOICE = "agent-voice"


def spec(language, voice_id=None, display_name=None):
    return {"language": language, "voice_id": voice_id, "display_name": display_name}


def directory_with(*entries, pair=("en", "ko")):
    return ParticipantDirectory.from_mapping(pair, dict(entries))


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


# -- participant directory -----------------------------------------------------


def test_directory_lanes_follow_insertion_order_and_are_capped() -> None:
    directory = directory_with((ALICE, spec("en")))
    assert directory.lane_of(ALICE) == 0
    assert directory.lane_of(BOB) is None
    assert directory.identity_for_lane(1) is None

    directory.set(BOB, spec("ko"))
    assert (directory.lane_of(BOB), directory.identity_for_lane(1)) == (1, BOB)
    assert len(directory) == MAX_PARTICIPANTS == 2

    with pytest.raises(ParticipantError) as full:
        directory.set("third", spec("en"))
    assert full.value.code == "session_full"
    # Re-sending an existing participant replaces it instead of counting twice.
    assert directory.set(BOB, spec("ko", "v1")).voice_id == "v1"
    assert len(directory) == 2


def test_directory_languages_voices_and_targets() -> None:
    directory = directory_with((ALICE, spec("en-US", "alice-voice", "Alice")))
    alice = directory.get(ALICE)
    assert (alice.language, alice.voice_id, alice.display_name) == ("en", "alice-voice", "Alice")
    # Before the second person joins, the target is the other language of the pair and
    # the empty lane waits in the language the first speaker does not use.
    assert directory.target_for(ALICE) == "ko"
    assert directory.language_for_lane(1) == "ko"

    directory.set(BOB, spec("ko"))
    assert directory.get(BOB).voice_id is None
    assert directory.target_for(ALICE) == "ko" and directory.target_for(BOB) == "en"
    assert directory.language_for_lane(0) == "en" and directory.language_for_lane(1) == "ko"
    assert directory.as_contract() == {
        ALICE: {"language": "en", "voice_id": "alice-voice", "display_name": "Alice"},
        BOB: {"language": "ko", "voice_id": None, "display_name": None},
    }


@pytest.mark.parametrize("language", ["", None, "english-please", "1"])
def test_directory_rejects_invalid_language(language) -> None:
    with pytest.raises(ParticipantError) as error:
        directory_with((ALICE, spec(language)))
    assert error.value.code == "invalid_participant_language"


def test_directory_update_notifies_listeners() -> None:
    directory = directory_with((ALICE, spec("en")))
    seen = []

    async def listener(participant) -> None:
        seen.append((participant.identity, participant.language))

    directory.add_listener(listener)
    asyncio.run(directory.update(BOB, spec("ko")))
    assert seen == [(BOB, "ko")]


# -- translation: per-speaker language, voice and events ---------------------------


def translator_with(directory, *, session=None, default_voice=DEFAULT_VOICE, calls=None, speaker_language="declared", route=False):
    async def translate(messages):
        if calls is not None:
            calls.append(messages)
        return "T:" + messages[-1]["content"]

    return TranslationProcessor(
        settings=InterpreterSettings(languages=directory.languages, direction="auto", speaker_language=speaker_language),
        translate=translate,
        model="test-model",
        session=session,
        initial_tts_language=directory.languages[1],
        participants=directory,
        default_voice_id=default_voice,
        route_by_listener=route,
        echo_guard=False,  # the fake translator returns its input, so a repeated sentence would look like an echo
    )


def segment(text, user_id, finalized=True, language=None):
    return TranscriptionFrame(text, user_id, "now", language, finalized=finalized)


def turn(text, user_id, language=None):
    """One speaker turn as the lane delivers it: boundary frames tagged with the speaker."""

    started, stopped = UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()
    started.metadata["user_id"] = stopped.metadata["user_id"] = user_id
    return [
        started,
        SleepFrame(sleep=0.02),
        segment(text, user_id, language=language),
        SleepFrame(sleep=0.02),
        stopped,
        SleepFrame(sleep=0.06),
    ]


def run(processor, frames):
    return asyncio.run(run_test(processor, frames_to_send=frames, expected_down_frames=None))


def sequence(down):
    out = []
    for frame in down:
        if isinstance(frame, TTSUpdateSettingsFrame):
            delta = {
                name: getattr(frame.delta, name)
                for name in ("language", "voice")
                if isinstance(getattr(frame.delta, name), str)
            }
            out.append(("settings", delta))
        elif isinstance(frame, TTSSpeakFrame):
            out.append(("say", frame.text))
    return out


def test_each_speaker_is_translated_into_the_other_language_in_their_own_voice() -> None:
    directory = directory_with(
        (ALICE, spec("en", "alice-voice", "Alice")), (BOB, spec("ko", "bob-voice"))
    )
    calls: list = []
    session = FakeSession()
    processor = translator_with(directory, session=session, calls=calls)
    frames = (
        turn("Hello there.", ALICE, language="ko")  # STT hint is ignored: declared wins
        + turn("안녕하세요.", BOB)
        + turn("How are you?", ALICE)
    )
    down, _ = run(processor, frames)

    assert sequence(down) == [
        # first speaker: target ko is already current; only the voice changes
        ("settings", {"voice": "alice-voice"}),
        ("say", "T:Hello there."),
        ("settings", {"language": "en", "voice": "bob-voice"}),
        ("say", "T:안녕하세요."),
        ("settings", {"language": "ko", "voice": "alice-voice"}),
        ("say", "T:How are you?"),
    ]
    assert "English (en) to Korean (ko)" in calls[0][0]["content"]
    assert "Korean (ko) to English (en)" in calls[1][0]["content"]

    users = session.of("TranscriptFinal")
    users = [p for p in users if p["role"] == "user"]
    assert [(p["user_id"], p["primary_language"], p["target_language"]) for p in users] == [
        (ALICE, "en", "ko"),
        (BOB, "ko", "en"),
        (ALICE, "en", "ko"),
    ]
    assert users[0]["speaker_name"] == "Alice" and "speaker_name" not in users[1]
    assistants = [p for p in session.of("TranscriptFinal") if p["role"] == "assistant"]
    assert [
        (p["speaker_user_id"], p["source_language"], p["target_language"], p["primary_language"])
        for p in assistants
    ] == [(ALICE, "en", "ko", "ko"), (BOB, "ko", "en", "en"), (ALICE, "en", "ko", "ko")]
    assert assistants[0]["speaker_name"] == "Alice"


def test_speaker_without_a_voice_uses_the_agent_voice_and_voice_is_not_resent() -> None:
    directory = directory_with((ALICE, spec("en", "alice-voice")), (BOB, spec("ko")))
    processor = translator_with(directory)
    down, _ = run(
        processor,
        turn("one", ALICE) + turn("two", ALICE) + turn("셋", BOB) + turn("four", ALICE),
    )

    assert sequence(down) == [
        ("settings", {"voice": "alice-voice"}),
        ("say", "T:one"),
        ("say", "T:two"),
        ("settings", {"language": "en", "voice": DEFAULT_VOICE}),
        ("say", "T:셋"),
        ("settings", {"language": "ko", "voice": "alice-voice"}),
        ("say", "T:four"),
    ]


def test_participant_added_while_running_is_used_for_the_next_utterance() -> None:
    directory = directory_with((ALICE, spec("en")))
    processor = translator_with(directory)

    class AddGuest(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame) and frame.text == "add-guest":
                await directory.update(BOB, spec("ko", "bob-voice"))
            await self.push_frame(frame, direction)

    pipeline = Pipeline([AddGuest(), processor])
    frames = (
        turn("hi", ALICE)
        + [TextFrame("add-guest"), SleepFrame(sleep=0.02)]
        + turn("네", BOB)
    )
    down, _ = run(pipeline, frames)
    assert ("say", "T:네") in sequence(down)
    assert ("settings", {"language": "en", "voice": "bob-voice"}) in sequence(down)


def test_overlapping_speakers_are_buffered_and_flushed_separately() -> None:
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
    calls: list = []
    processor = translator_with(directory, calls=calls)

    def started(user_id):
        frame = UserStartedSpeakingFrame()
        frame.metadata["user_id"] = user_id
        return frame

    def stopped(user_id):
        frame = UserStoppedSpeakingFrame()
        frame.metadata["user_id"] = user_id
        return frame

    frames = [
        started(ALICE),
        SleepFrame(sleep=0.02),
        segment("part one,", ALICE, finalized=False),
        started(BOB),
        SleepFrame(sleep=0.02),
        segment("안녕", BOB),
        segment("part two", ALICE),
        SleepFrame(sleep=0.02),
        stopped(BOB),
        SleepFrame(sleep=0.02),
        stopped(ALICE),
        SleepFrame(sleep=0.1),
    ]
    run(processor, frames)

    texts = sorted(messages[-1]["content"] for messages in calls)
    assert texts == ["part one, part two", "안녕"]


def test_unknown_speaker_falls_back_to_detection_and_the_agent_voice() -> None:
    directory = directory_with((ALICE, spec("en", "alice-voice")))
    processor = translator_with(directory)
    down, _ = run(processor, turn("привет", "someone-else", language="ru"))
    # Not a participant: single-participant resolution (script guess; ru is not in the
    # en/ko pair so the previous/first language applies) and no voice switch.
    assert [s for s in sequence(down) if s[0] == "say"] == [("say", "T:привет")]
    assert not any(s[0] == "settings" and "voice" in s[1] for s in sequence(down))


def test_single_participant_processor_is_unchanged() -> None:
    async def translate(messages):
        return "T:" + messages[-1]["content"]

    processor = TranslationProcessor(
        settings=InterpreterSettings(languages=("en", "ru"), direction="auto"),
        translate=translate,
        model="m",
        initial_tts_language="ru",
    )
    session = FakeSession()
    processor._session = session
    down, _ = run(processor, turn("hello", "speaker", language="en"))
    assert sequence(down) == [("say", "T:hello")]
    users = [p for p in session.of("TranscriptFinal") if p["role"] == "user"]
    assert "target_language" not in users[0]
    assistant = [p for p in session.of("TranscriptFinal") if p["role"] == "assistant"][0]
    assert set(assistant) == {"content", "role", "primary_language", "detected_languages"}


# -- lanes: audio routing and STT language ---------------------------------------


class FakeSTT(FrameProcessor):
    """Records the audio it receives and the settings updates addressed to it."""

    def __init__(self, name) -> None:
        super().__init__(name=name)
        self.audio_users: list[str] = []
        self.language_updates: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, STTUpdateSettingsFrame) and frame.service is self:
            self.language_updates.append(str(frame.delta.language))
            return
        if isinstance(frame, UserAudioRawFrame):
            self.audio_users.append(frame.user_id)
        await self.push_frame(frame, direction)


def audio(user_id):
    return UserAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, user_id=user_id)


def build_lanes(directory):
    stts, lanes = [], []
    for lane in range(MAX_PARTICIPANTS):
        language = directory.language_for_lane(lane)
        stt = FakeSTT(f"STT{lane}")
        stts.append(stt)
        lanes.append(
            [
                ParticipantLaneGate(directory, lane, stt=stt, language=language),
                stt,
                LaneTurnTagger(directory, lane),
            ]
        )
    return stts, ParallelPipeline(*lanes)


def test_each_participants_audio_reaches_only_its_own_stt_lane() -> None:
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
    stts, parallel = build_lanes(directory)
    frames = [audio(ALICE), audio(BOB), audio(ALICE), audio("stranger"), SleepFrame(sleep=0.05)]
    run(parallel, frames)
    assert stts[0].audio_users == [ALICE, ALICE]
    assert stts[1].audio_users == [BOB]


def test_participant_joining_later_retargets_its_lane_stt_language() -> None:
    # Lane 1 starts with a placeholder language and switches when the guest is added;
    # the other lane's STT is left alone.
    directory = directory_with((ALICE, spec("en")), pair=("en", "ko"))
    stts, parallel = build_lanes(directory)

    class AddGuest(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame) and frame.text == "add-guest":
                await directory.update(BOB, spec("ru"))
            await self.push_frame(frame, direction)

    frames = [
        audio(ALICE),
        TextFrame("add-guest"),
        SleepFrame(sleep=0.05),
        audio(BOB),
        SleepFrame(sleep=0.05),
    ]
    run(Pipeline([AddGuest(), parallel]), frames)
    assert stts[1].language_updates == ["ru"]
    assert stts[0].language_updates == []
    assert stts[1].audio_users == [BOB]


def test_turn_boundaries_are_tagged_with_the_lane_participant() -> None:
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
    for lane, identity in ((0, ALICE), (1, BOB)):
        down, _ = run(
            LaneTurnTagger(directory, lane),
            [UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), SleepFrame(sleep=0.02)],
        )
        boundaries = [
            f for f in down if isinstance(f, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame))
        ]
        assert len(boundaries) == 2
        assert all(f.metadata["user_id"] == identity for f in boundaries)


# -- pipeline builder / config ------------------------------------------------------


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


def test_builder_creates_one_stt_lane_per_participant(light_models, monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    config = pipeline_config(
        mode="interpreter",
        interpreter={"languages": ["en", "ko"], "direction": "one_way"},
        participants={ALICE: spec("en", "alice-voice")},
    )
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    session.participants = ParticipantDirectory.from_mapping(("en", "ko"), config.participants)
    _, worker = build_livekit_pipeline(config, session)
    processors = top_level(worker)
    assert [type(p).__name__ for p in processors] == [
        "PerParticipantLiveKitInput",
        "ParallelPipeline",
        "TranslationProcessor",
        "ParallelPipeline",  # one TTS per listener (RM-001)
        "RuntimeEventProcessor",
        "ListenerRoutedLiveKitOutput",
    ]
    lanes = processors[1].processors
    assert len(lanes) == 2
    stt_languages = []
    for lane in lanes:
        names = [type(p).__name__ for p in lane._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}]
        assert names == [
            "ParticipantLaneGate",
            "VADProcessor",
            "DeepgramSTTService",
            "UserTurnProcessor",
            "LaneTurnTagger",
        ]
        stt = next(p for p in lane._processors if type(p).__name__ == "DeepgramSTTService")
        stt_languages.append(stt._settings.language)
    # Fixed per-participant languages: no "multi" detection, so non-code-switching
    # pairs such as ko <-> en work. The not-yet-joined lane waits in the other language.
    assert stt_languages == ["en", "ko"]
    translator = processors[2]
    assert translator._participants is session.participants
    assert translator._default_voice_id == DEFAULT_VOICE


def test_builder_keeps_single_participant_pipeline_without_a_mapping(light_models) -> None:
    config = pipeline_config(
        mode="interpreter", interpreter={"languages": ["en", "ru"], "direction": "auto"}
    )
    _, worker = build_livekit_pipeline(config)
    names = [type(p).__name__ for p in top_level(worker)]
    assert names[:5] == [
        "LiveKitInputTransport", "VADProcessor", "DeepgramSTTService", "UserTurnProcessor",
        "TranslationProcessor",
    ]  # fmt: skip


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "LIVEKIT_URL": "ws://localhost:7880",
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }.items():
        monkeypatch.setenv(key, value)


def test_config_participants_require_a_cascaded_interpreter(monkeypatch) -> None:
    _env(monkeypatch)
    base = {"room_name": "room", "livekit_token": "t"}
    participants = {ALICE: spec("en", "alice-voice")}
    interpreter = {"languages": ["en", "ko"], "direction": "one_way"}
    config = PipelineConfig.from_environment(
        {**base, "mode": "interpreter", "interpreter": interpreter, "participants": participants}
    )
    assert config.participants == participants
    assert PipelineConfig.from_environment(base).participants is None
    assert (
        PipelineConfig.from_environment(
            {**base, "mode": "interpreter", "interpreter": interpreter, "participants": {}}
        ).participants
        is None
    )
    for bad in (
        {**base, "participants": participants},  # agent mode
        {
            **base,
            "mode": "interpreter",
            "interpreter": {**interpreter, "engine": "realtime"},
            "participants": participants,
        },
        {**base, "mode": "interpreter", "interpreter": interpreter, "participants": {ALICE: {}}},
    ):
        with pytest.raises(RuntimeError):
            PipelineConfig.from_environment(bad)


# -- HTTP API -------------------------------------------------------------------------


@pytest.fixture
def runtime_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    started: list[str] = []

    async def fake_run_session(request) -> None:
        started.append(str(request.session_id))
        await asyncio.sleep(30)

    monkeypatch.setattr(api, "run_session", fake_run_session)
    yield {"X-Runtime-Token": "test-runtime-token"}
    for key in started:
        api._participant_directories.pop(key, None)
        api._session_contracts.pop(key, None)
        api._session_states.pop(key, None)
        api._active_sessions.pop(key, None)


def start_payload(**extra):
    return {
        "session_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "voice-test-room",
        "mode": "interpreter",
        "interpreter": {"languages": ["en", "ko"], "direction": "one_way", "engine": "cascaded"},
        **extra,
    }


def test_start_accepts_participants_and_add_route_updates_the_live_mapping(runtime_api) -> None:
    payload = start_payload(participants={ALICE: spec("en", "alice-voice", "Alice")})
    key = payload["session_id"]
    with TestClient(api.app) as client:
        assert client.post("/sessions", json=payload, headers=runtime_api).status_code == 202
        directory = api._participant_directories[key]
        assert directory.get(ALICE).voice_id == "alice-voice"

        added = client.post(
            f"/sessions/{key}/participants",
            json={"identity": BOB, "language": "ko", "voice_id": "bob-voice"},
            headers=runtime_api,
        )
        assert added.status_code == 200 and added.json() == {"status": "updated", "participants": 2}
        assert directory.get(BOB).language == "ko"
        assert directory.target_for(ALICE) == "ko"

        # The Platform's retry now carries both participants and must not conflict.
        retry = payload | {
            "participants": {
                ALICE: spec("en", "alice-voice", "Alice"),
                BOB: spec("ko", "bob-voice"),
            }
        }
        assert client.post("/sessions", json=retry, headers=runtime_api).json() == {
            "status": "already_running"
        }

        full = client.post(
            f"/sessions/{key}/participants",
            json={"identity": "third", "language": "en"},
            headers=runtime_api,
        )
        assert (full.status_code, full.json()["detail"]) == (409, "session_full")
        bad = client.post(
            f"/sessions/{key}/participants",
            json={"identity": BOB, "language": "!!"},
            headers=runtime_api,
        )
        assert (bad.status_code, bad.json()["detail"]) == (422, "invalid_participant_language")


def test_add_participant_route_errors(runtime_api) -> None:
    plain = start_payload()
    with TestClient(api.app) as client:
        body = {"identity": BOB, "language": "ko"}
        assert client.post(f"/sessions/{uuid4()}/participants", json=body, headers=runtime_api).status_code == 404
        assert client.post(f"/sessions/{uuid4()}/participants", json=body).status_code == 401
        assert (
            client.post(
                f"/sessions/{uuid4()}/participants", json=body, headers={"X-Runtime-Token": "nope"}
            ).status_code
            == 401
        )
        assert client.post("/sessions", json=plain, headers=runtime_api).status_code == 202
        response = client.post(
            f"/sessions/{plain['session_id']}/participants", json=body, headers=runtime_api
        )
        assert (response.status_code, response.json()["detail"]) == (409, "participants_not_enabled")


def test_participants_need_a_cascaded_interpreter(runtime_api) -> None:
    with TestClient(api.app) as client:
        realtime = start_payload(
            participants={ALICE: spec("en")},
            interpreter={"languages": ["en", "ko"], "direction": "one_way", "engine": "realtime"},
        )
        agent = start_payload(participants={ALICE: spec("en")}, mode="agent", interpreter=None)
        for payload in (realtime, agent):
            response = client.post("/sessions", json=payload, headers=runtime_api)
            assert (response.status_code, response.json()["detail"]) == (
                422,
                "participants_require_cascaded_interpreter",
            )


def test_session_without_participants_has_unchanged_contract(runtime_api) -> None:
    payload = start_payload()
    request = api.SessionStart(**payload)
    assert request.participants is None
    assert "participants" not in request.model_dump(mode="json")
    with_participants = api.SessionStart(**payload, participants={ALICE: spec("en")})
    assert with_participants.model_dump(mode="json")["participants"] == {ALICE: spec("en")}
    with TestClient(api.app) as client:
        assert client.post("/sessions", json=payload, headers=runtime_api).status_code == 202
    assert payload["session_id"] not in api._participant_directories


# -- lanes feeding the translator, end to end (fake STT, no providers) ------------------


class SpeakingSTT(FrameProcessor):
    """On each audio chunk, acts like Deepgram plus the turn processor for a short turn."""

    def __init__(self, name, words) -> None:
        super().__init__(name=name)
        self.words = words

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserAudioRawFrame):
            await self.push_frame(UserStartedSpeakingFrame())
            await self.push_frame(
                TranscriptionFrame(self.words, frame.user_id, "now", None, finalized=True)
            )
            await self.push_frame(UserStoppedSpeakingFrame())
            return
        await self.push_frame(frame, direction)


def test_two_lanes_feed_one_translator_with_the_right_speaker_each() -> None:
    directory = directory_with(
        (ALICE, spec("en", "alice-voice")), (BOB, spec("ko", "bob-voice"))
    )
    session = FakeSession()
    translator = translator_with(directory, session=session)
    lanes = [
        [
            ParticipantLaneGate(directory, lane, stt=stt, language="x"),
            stt,
            LaneTurnTagger(directory, lane),
        ]
        for lane, stt in enumerate(
            (SpeakingSTT("A", "hello from alice"), SpeakingSTT("B", "밥의 말"))
        )
    ]
    frames = [
        audio(ALICE),
        SleepFrame(sleep=0.15),
        audio(BOB),
        SleepFrame(sleep=0.15),
        audio(ALICE),
        SleepFrame(sleep=0.15),
    ]
    down, _ = run(Pipeline([ParallelPipeline(*lanes), translator]), frames)

    assert [s[1] for s in sequence(down) if s[0] == "say"] == [
        "T:hello from alice",
        "T:밥의 말",
        "T:hello from alice",
    ]
    voices = [s[1]["voice"] for s in sequence(down) if s[0] == "settings" and "voice" in s[1]]
    assert voices == ["alice-voice", "bob-voice", "alice-voice"]
    assert [
        (p["speaker_user_id"], p["target_language"])
        for p in session.of("TranscriptFinal")
        if p["role"] == "assistant"
    ] == [(ALICE, "ko"), (BOB, "en"), (ALICE, "ko")]


# -- session lifecycle ------------------------------------------------------------------


def test_two_person_session_survives_one_participant_leaving(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    class Client:
        def __init__(self) -> None:
            self.events = []

        async def emit(self, event) -> None:
            self.events.append(event)

    class Transport:
        def __init__(self) -> None:
            self.handlers = {}

        def add_event_handler(self, name, handler) -> None:
            self.handlers[name] = handler

        async def cleanup(self) -> None:
            return None

    class Worker:
        def __init__(self) -> None:
            self.cancel_reasons = []

        def cancel(self, *, reason=None) -> None:
            self.cancel_reasons.append(reason)

    transport, worker = Transport(), Worker()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, worker))
    outcomes = []

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, _worker) -> None:
            return None

        async def run(self) -> None:
            handlers = transport.handlers
            await handlers["on_connected"](transport)
            await handlers["on_audio_track_subscribed"](transport, ALICE)  # already in the room
            await handlers["on_participant_connected"](transport, BOB)  # joined later
            await handlers["on_participant_disconnected"](transport, BOB)
            outcomes.append(list(worker.cancel_reasons))
            await handlers["on_participant_disconnected"](transport, ALICE)

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), Client())
    session.participants = directory_with((ALICE, spec("en")))
    asyncio.run(session.run_pipeline(pipeline_config()))

    assert outcomes == [[]]  # BOB leaving did not end the call
    assert worker.cancel_reasons == ["participant_disconnected"]  # the last one leaving did


# -- regression: interleaved speakers must not bleed through the audio resampler --------


def _chunk(samples):
    import numpy as np
    from types import SimpleNamespace

    return SimpleNamespace(
        frame=SimpleNamespace(data=np.asarray(samples, dtype=np.int16), sample_rate=48000, num_channels=1)
    )


def _interleaved_events(chunks=120):
    """Participant A speaks (tone), participant B streams digital silence, interleaved."""

    import numpy as np

    t = np.arange(chunks * 480)
    tone = (8000 * np.sin(2 * np.pi * 440 * t / 48000)).astype(np.int16)
    events = []
    for index in range(chunks):
        events.append((_chunk(tone[index * 480 : (index + 1) * 480]), ALICE))
        events.append((_chunk(np.zeros(480)), BOB))
    return events, tone


def _drive_input(input_cls, events):
    """Run the transport input's audio loop over fake LiveKit events; return frames per user."""

    from types import SimpleNamespace

    class FakeClient:
        async def get_next_audio_frame(self):
            for event in events:
                yield event

    transport = input_cls(
        SimpleNamespace(), FakeClient(), SimpleNamespace(audio_in_enabled=True)
    )
    transport._sample_rate = 16000
    received: dict[str, bytes] = {}

    async def capture(frame) -> None:
        received[frame.user_id] = received.get(frame.user_id, b"") + frame.audio

    transport.push_audio_frame = capture  # type: ignore[method-assign]
    asyncio.run(transport._audio_in_task_handler())
    return received


def _peak(audio: bytes) -> int:
    import numpy as np

    return int(np.abs(np.frombuffer(audio, dtype=np.int16)).max()) if audio else 0


def test_interleaved_participants_are_resampled_independently() -> None:
    from src.runtime.participant_lanes import PerParticipantLiveKitInput

    events, tone = _interleaved_events()
    received = _drive_input(PerParticipantLiveKitInput, events)

    # B only ever sent digital silence: nothing of A may leak into B's audio (the
    # resampler's own rounding noise is +-1) ...
    assert _peak(received[BOB]) <= 2
    # ... and A's audio is the same as resampling A on its own.
    import numpy as np
    from pipecat.audio.utils import create_stream_resampler

    alone = asyncio.run(create_stream_resampler().resample(tone.tobytes(), 48000, 16000))
    mine = np.frombuffer(received[ALICE], dtype=np.int16).astype(float)
    ref = np.frombuffer(alone, dtype=np.int16).astype(float)
    size = min(len(mine), len(ref))
    assert size > 1000
    assert np.abs(mine[:size] - ref[:size]).max() <= 8  # rounding only (tone peak is 8000)


def test_stock_shared_resampler_does_bleed_between_participants() -> None:
    # Documents the pipecat behaviour the per-participant input works around: one
    # stateful resampler fed interleaved chunks puts A's speech into B's frames.
    from pipecat.audio.utils import create_stream_resampler

    events, _ = _interleaved_events()
    shared = create_stream_resampler()

    async def run_shared():
        out: dict[str, bytes] = {}
        for event, user in events:
            out[user] = out.get(user, b"") + await shared.resample(
                event.frame.data.tobytes(), 48000, 16000
            )
        return out

    assert _peak(asyncio.run(run_shared())[BOB]) > 1000


def test_private_listening_contract_needs_participants(runtime_api) -> None:
    with TestClient(api.app) as client:
        response = client.post(
            "/sessions", json=start_payload(private_listening=True), headers=runtime_api
        )
        assert (response.status_code, response.json()["detail"]) == (
            422,
            "private_listening_requires_participants",
        )
        payload = start_payload(participants={ALICE: spec("en")}, private_listening=True)
        assert client.post("/sessions", json=payload, headers=runtime_api).status_code == 202
        # The Platform's retry of the same contract is recognised, not a conflict.
        assert client.post("/sessions", json=payload, headers=runtime_api).json() == {
            "status": "already_running"
        }


def test_phone_room_participants_need_private_listening(runtime_api) -> None:
    phone = {**spec("ko"), "phone_room": True}
    with TestClient(api.app) as client:
        response = client.post(
            "/sessions",
            json=start_payload(participants={ALICE: spec("en"), BOB: phone}),
            headers=runtime_api,
        )
        assert (response.status_code, response.json()["detail"]) == (
            422,
            "phone_room_requires_private_listening",
        )
        payload = start_payload(
            participants={ALICE: spec("en"), BOB: phone}, private_listening=True
        )
        assert client.post("/sessions", json=payload, headers=runtime_api).status_code == 202
        key = payload["session_id"]
        assert api._participant_directories[key].phone_room_identities() == [BOB]
        # The stored contract keeps the flag (and only where it is set).
        stored = api._session_contracts[key]["participants"]
        assert stored[BOB]["phone_room"] is True and "phone_room" not in stored[ALICE]


def test_add_phone_participant_to_a_private_listening_session(runtime_api) -> None:
    body = {"identity": "sip-guest-1", "language": "ko", "phone_room": True}
    plain = start_payload(participants={ALICE: spec("en")})
    private = start_payload(participants={ALICE: spec("en")}, private_listening=True)
    with TestClient(api.app) as client:
        for payload in (plain, private):
            assert client.post("/sessions", json=payload, headers=runtime_api).status_code == 202
        refused = client.post(
            f"/sessions/{plain['session_id']}/participants", json=body, headers=runtime_api
        )
        assert (refused.status_code, refused.json()["detail"]) == (
            422,
            "phone_room_requires_private_listening",
        )
        key = private["session_id"]
        added = client.post(f"/sessions/{key}/participants", json=body, headers=runtime_api)
        assert added.status_code == 200
        assert api._participant_directories[key].phone_room_identities() == ["sip-guest-1"]
        assert api._session_contracts[key]["participants"]["sip-guest-1"]["phone_room"] is True


def test_unanswered_phone_participant_can_be_withdrawn(runtime_api) -> None:
    phone = {"identity": "sip-guest-1", "language": "ko", "phone_room": True}
    private = start_payload(participants={ALICE: spec("en")}, private_listening=True)
    key = private["session_id"]
    with TestClient(api.app) as client:
        assert client.post("/sessions", json=private, headers=runtime_api).status_code == 202
        url = f"/sessions/{key}/participants"
        assert client.post(url, json=phone, headers=runtime_api).status_code == 200
        directory = api._participant_directories[key]

        refused = client.delete(f"{url}/{ALICE}", headers=runtime_api)  # not a phone
        assert (refused.status_code, refused.json()["detail"]) == (409, "participant_not_removable")
        assert client.delete(f"{url}/sip-guest-1").status_code == 401
        assert client.delete(f"{url}/nobody", headers=runtime_api).status_code == 404

        removed = client.delete(f"{url}/sip-guest-1", headers=runtime_api)
        assert removed.json() == {"status": "removed", "participants": 1}
        assert "sip-guest-1" not in directory
        assert set(api._session_contracts[key]["participants"]) == {ALICE}
        # The slot is free again for the retry.
        assert client.post(url, json=phone, headers=runtime_api).status_code == 200


@pytest.mark.parametrize(
    "voice",
    ["../../v1/user", "a/b", "a?b=c", "%2e%2e", "a b", "x" * 129, "ok\n", "../"],
)
def test_directory_rejects_voice_ids_that_could_alter_a_provider_url(voice) -> None:
    directory = ParticipantDirectory(["en", "ko"])
    with pytest.raises(ParticipantError) as raised:
        directory.set(ALICE, spec("en", voice))
    assert raised.value.code == "invalid_participant_voice"
    assert directory.get(ALICE) is None  # nothing was registered


@pytest.mark.parametrize(
    "voice", ["JBFqnCBsd6RMkjVDRZzb", "0a7b9c3e-1f4d-4a8e-9c2b-7d6e5f4a3b21", "alice-voice", "v_1"]
)
def test_directory_accepts_real_looking_provider_voice_ids(voice) -> None:
    directory = ParticipantDirectory(["en", "ko"])
    assert directory.set(ALICE, spec("en", voice)).voice_id == voice


def test_elevenlabs_session_with_unsafe_default_voice_is_refused_before_any_state(
    runtime_api,
) -> None:
    for bad in ("../v1/x", "voice id", "a" * 129, "ok\n"):
        payload = start_payload(voice_id=bad, providers={"tts": "elevenlabs"})
        with TestClient(api.app) as client:
            response = client.post("/sessions", json=payload, headers=runtime_api)
        assert (response.status_code, response.json()["detail"]) == (422, "invalid_voice_id")
        key = payload["session_id"]
        assert key not in api._session_contracts
        assert key not in api._session_states
        assert key not in api._active_sessions
        assert key not in api._participant_directories


def test_elevenlabs_session_with_safe_default_voice_is_accepted(runtime_api) -> None:
    payload = start_payload(voice_id="Abc_123-x", providers={"tts": "elevenlabs"})
    with TestClient(api.app) as client:
        assert client.post("/sessions", json=payload, headers=runtime_api).status_code == 202


def test_voice_id_check_does_not_affect_other_tts_providers(runtime_api) -> None:
    with TestClient(api.app) as client:
        for providers in ({"tts": "cartesia"}, {}):
            payload = start_payload(voice_id="../weird id.with:chars", providers=providers)
            response = client.post("/sessions", json=payload, headers=runtime_api)
            assert response.status_code == 202, providers


@pytest.mark.parametrize("mode", ["auto", "declared"])
def test_speaker_language_settings(mode):
    assert InterpreterSettings.from_mapping({"languages": ["en", "ru"], "speaker_language": mode}).speaker_language == mode
    assert InterpreterSettings.from_mapping({"languages": ["en", "ru"]}).speaker_language == "declared"


@pytest.mark.parametrize("mode", ["invalid", None, 1, []])
def test_invalid_speaker_language_settings(mode):
    with pytest.raises(RuntimeError, match="speaker_language"):
        InterpreterSettings.from_mapping({"languages": ["en", "ru"], "speaker_language": mode})


@pytest.mark.parametrize("detected", ["ru", "rus", None, "en", "fra"])
def test_auto_russian_passthrough_to_russian_listener(detected):
    from src.runtime.echo_guard import EchoGuard

    directory = directory_with((ALICE, spec("en", "alice-voice")), (BOB, spec("ru")), pair=("en", "ru"))
    calls = []
    session = FakeSession()
    processor = translator_with(directory, calls=calls, session=session, speaker_language="auto", route=True)
    processor._echo = EchoGuard()
    text = "Здравствуйте, как ваши дела?"
    down, _ = run(processor, turn(text, ALICE, detected))
    assert calls == []
    speech = [f for f in down if isinstance(f, TTSSpeakFrame)]
    assert [(f.text, f.metadata["listener_lane"]) for f in speech] == [(text, 1)]
    assert processor._echo.is_echo(text)
    events = session.of("TranscriptFinal")
    assert [(e["role"], e["content"], e["primary_language"], e["target_language"]) for e in events] == [
        ("user", text, "ru", "ru"), ("assistant", text, "ru", "ru")]
    assert not session.of("LatencyMeasured")
    assert list(processor._history) == [("ru", text, "ru", text)]


@pytest.mark.parametrize("text,detected,expected", [
    ("Hello, how are you?", "rus", "en"),
    ("12345", "eng", "en"),
    ("12345", "rus", "ru"),
    ("12345", None, "en"),
])
def test_auto_source_detection_and_fallback(text, detected, expected):
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ru")), pair=("en", "ru"))
    calls = []
    session = FakeSession()
    processor = translator_with(directory, calls=calls, session=session, speaker_language="auto")
    run(processor, turn(text, ALICE, detected))
    assert session.of("TranscriptFinal")[0]["primary_language"] == expected
    assert len(calls) == (expected != "ru")


def test_auto_gate_never_updates_declared_language():
    directory = directory_with((ALICE, spec("en")))
    stt = FakeSTT("STT")
    gate = ParticipantLaneGate(directory, 1, stt=stt, language="en", sync_language=False)
    run(Pipeline([gate, stt]), [SleepFrame(sleep=0.02)])
    # Both StartFrame synchronization and a later participant notification are disabled.
    asyncio.run(directory.update(BOB, spec("ko")))
    assert stt.language_updates == []


@pytest.mark.parametrize("provider", ["deepgram", "elevenlabs"])
@pytest.mark.parametrize("mode", ["auto", "declared"])
def test_builder_auto_lanes_override_fixed_provider_language(light_models, monkeypatch, provider, mode):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    config = pipeline_config(
        mode="interpreter", interpreter={"languages": ["en", "ru"], "speaker_language": mode},
        participants={ALICE: spec("en")}, provider_settings={"stt": {"language": "en"}},
        providers={"stt": provider, "llm": "openai", "tts": "cartesia"},
        elevenlabs_api_key="test-key",
    )
    _, worker = build_livekit_pipeline(config)
    for index, lane in enumerate(top_level(worker)[1].processors):
        stt = next(p for p in lane._processors if type(p).__name__ in {"DeepgramSTTService", "VoiceLabsElevenLabsSTTService"})
        gate = next(p for p in lane._processors if isinstance(p, ParticipantLaneGate))
        expected = ("multi" if provider == "deepgram" else None) if mode == "auto" else ("en", "ru")[index]
        from src.runtime.participants import normalize_language
        actual = stt._settings.language
        assert (normalize_language(actual) if mode == "declared" else actual) == expected
        assert gate._sync_declared_language == (mode == "declared")
        if provider == "elevenlabs":
            assert stt._include_language_detection == (mode == "auto")


@pytest.mark.parametrize("code,expected", [("eng", "en"), ("rus", "ru"), ("kor", "ko")])
def test_iso_three_letter_language_normalization(code, expected):
    from src.runtime.participants import normalize_language
    assert normalize_language(code) == expected


@pytest.mark.parametrize("pair,text,expected", [
    (("en", "ru"), "Hello Ж", "en"),
    (("en", "ru"), "Hi Привет", "ru"),
    (("en", "ru"), "Hi Да", None),
    (("en", "es"), "Hello", None),
])
def test_auto_script_vote_requires_clear_pair_candidate(pair, text, expected):
    from src.runtime.interpreter import guess_language_from_script
    assert guess_language_from_script(text, pair, clear_vote=True) == expected


def test_passthrough_timeline_tracks_tts_without_translation_marks():
    from unittest.mock import AsyncMock
    from src.runtime.interpreter import _Utterance
    from src.runtime.timeline import TimelineTracker, UtteranceTimeline

    session = FakeSession()
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ru")), pair=("en", "ru"))
    processor = translator_with(directory, session=session, speaker_language="auto", route=True)
    processor.push_frame = AsyncMock()
    processor._timeline = TimelineTracker(session, "cascaded")
    timeline = UtteranceTimeline("cascaded", 0.0, speaker=ALICE, turn_id=7)
    asyncio.run(processor.process_utterance(_Utterance("Здравствуйте!", "rus", ALICE, "now", timeline)))
    assert timeline.speech_complete and timeline.chunks_pushed == 1
    assert timeline.listener == BOB and timeline.destination == "translation-lane-1"
    assert set(timeline.marks) == {"tts_request"}
    assert all(e["turn_id"] == 7 for e in session.of("TranscriptFinal"))
