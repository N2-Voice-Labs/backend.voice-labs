"""ElevenLabs as a live TTS provider: config, factory, per-participant voices, interruption,
cleanup and failure handling.

Everything here runs the REAL Pipecat ``ElevenLabsTTSService`` against a FAKE WebSocket
(``FakeSocket``) injected at ``service._websocket_connect``. No network is used and no
ElevenLabs behaviour is verified: the fake only answers the way Pipecat's own protocol code
expects (audio for text, ``isFinal`` for ``close_context``). What the real API does with the
voice id / ``language_code`` in the URL is out of reach of these tests.
"""

import asyncio
import base64
import json
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("pipecat")
pytest.importorskip("pipecat.services.elevenlabs.tts")

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.tests.utils import SleepFrame
from pipecat.workers.runner import WorkerRunner
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close
from websockets.protocol import State

import src.runtime.pipeline
from src.runtime import pipeline as pipeline_module
from src.runtime import api
from src.runtime.interpreter import InterpreterSettings, TranslationProcessor
from src.runtime.participants import ParticipantDirectory
from src.runtime.pipeline import PipelineConfig
from src.runtime.provider_failures import (
    cascaded_fallback_config,
    provider_kind,
    watch_provider_failures,
)
from src.runtime.session import RuntimeSession, SessionState
from src.runtime.speech_providers import (
    build_tts,
    provider_capabilities,
    provider_setting_types,
    tts_names,
)

TIMEOUT = 10
ELEVEN_PROVIDERS = {"stt": "deepgram", "llm": "openai-compatible", "tts": "elevenlabs"}
SECRET = "xi-SECRET-key-123"

ENV_NAMES = (
    "LIVEKIT_URL",
    "RUNTIME_LIVEKIT_TOKEN",
    "LIVEKIT_ROOM_NAME",
    "DEEPGRAM_API_KEY",
    "OPENAI_API_KEY",
    "CARTESIA_API_KEY",
    "CARTESIA_VOICE_ID",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
)


def make_config(**overrides) -> PipelineConfig:
    values = dict(
        livekit_url="ws://localhost:7880",
        livekit_token="token",
        room_name="room",
        deepgram_api_key="dg-key",
        openai_api_key="oa-key",
        cartesia_api_key="",
        cartesia_voice_id="",
        system_instruction="prompt",
        providers=dict(ELEVEN_PROVIDERS),
        elevenlabs_api_key=SECRET,
        elevenlabs_voice_id="default-voice",
    )
    values.update(overrides)
    return PipelineConfig(**values)


# -- config: credentials follow the selected TTS provider ----------------------------


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("RUNTIME_LIVEKIT_TOKEN", "token")
    monkeypatch.setenv("LIVEKIT_ROOM_NAME", "room")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    return monkeypatch


def test_elevenlabs_session_needs_only_elevenlabs_credentials(clean_env) -> None:
    clean_env.setenv("ELEVENLABS_API_KEY", "el-key")
    clean_env.setenv("ELEVENLABS_VOICE_ID", "el-voice")
    config = PipelineConfig.from_environment({"providers": {"tts": "elevenlabs"}})
    assert (config.tts_api_key, config.tts_voice_id) == ("el-key", "el-voice")
    assert (config.cartesia_api_key, config.cartesia_voice_id) == ("", "")


def test_elevenlabs_session_reports_only_elevenlabs_names_when_missing(clean_env) -> None:
    clean_env.setenv("CARTESIA_API_KEY", "ct")  # irrelevant to this session
    clean_env.setenv("CARTESIA_VOICE_ID", "ct-voice")
    with pytest.raises(RuntimeError) as error:
        PipelineConfig.from_environment({"providers": {"tts": "elevenlabs"}})
    assert "ELEVENLABS_API_KEY" in str(error.value)
    assert "ELEVENLABS_VOICE_ID" in str(error.value)
    assert "CARTESIA" not in str(error.value)


def test_cartesia_session_does_not_need_elevenlabs_credentials(clean_env) -> None:
    clean_env.setenv("CARTESIA_API_KEY", "ct")
    clean_env.setenv("CARTESIA_VOICE_ID", "ct-voice")
    config = PipelineConfig.from_environment({})  # default provider: cartesia
    assert (config.tts_api_key, config.tts_voice_id) == ("ct", "ct-voice")
    assert config.elevenlabs_api_key == "" and config.elevenlabs_voice_id == ""
    clean_env.delenv("CARTESIA_API_KEY")
    with pytest.raises(RuntimeError) as error:
        PipelineConfig.from_environment({})
    assert "CARTESIA_API_KEY" in str(error.value) and "ELEVENLABS" not in str(error.value)


def test_session_voice_id_replaces_only_the_selected_providers_default(clean_env) -> None:
    clean_env.setenv("ELEVENLABS_API_KEY", "el-key")
    clean_env.setenv("ELEVENLABS_VOICE_ID", "el-voice")
    clean_env.setenv("CARTESIA_API_KEY", "ct")
    clean_env.setenv("CARTESIA_VOICE_ID", "ct-voice")
    eleven = PipelineConfig.from_environment(
        {"providers": {"tts": "elevenlabs"}, "voice_id": "session-voice"}
    )
    assert eleven.tts_voice_id == "session-voice"
    assert eleven.cartesia_voice_id == "ct-voice"  # the other provider's default is untouched
    cartesia = PipelineConfig.from_environment({"voice_id": "session-voice"})
    assert cartesia.tts_voice_id == cartesia.cartesia_voice_id == "session-voice"
    assert cartesia.elevenlabs_voice_id == "el-voice"


def test_realtime_engine_needs_no_tts_credentials_for_either_provider(clean_env) -> None:
    clean_env.delenv("DEEPGRAM_API_KEY")
    for tts in ("cartesia", "elevenlabs"):
        config = PipelineConfig.from_environment(
            {
                "mode": "interpreter",
                "interpreter": {"engine": "realtime", "languages": ["en", "ko"], "direction": "one_way"},
                "providers": {"tts": tts},
            }
        )
        assert config.mode == "interpreter"


def test_existing_cartesia_constructor_keeps_working() -> None:
    config = PipelineConfig(
        "ws://livekit", "token", "room", "deepgram", "openai", "ct-key", "ct-voice", "prompt"
    )
    assert config.providers["tts"] == "cartesia"
    assert (config.tts_api_key, config.tts_voice_id) == ("ct-key", "ct-voice")
    assert config.elevenlabs_api_key == ""
    # the legacy constructor drives the factory and the interpreter default voice
    from pipecat.services.cartesia.tts import CartesiaTTSService

    service = build_tts(config, "en")
    assert isinstance(service, CartesiaTTSService)
    assert service._api_key == "ct-key" and service._settings.voice == "ct-voice"


# -- factory ----------------------------------------------------------------------------


def test_elevenlabs_is_registered_with_honest_capabilities() -> None:
    assert "elevenlabs" in tts_names()
    capabilities = provider_capabilities("tts", "elevenlabs")
    assert capabilities.supports("streaming") and capabilities.supports("transport_destination")
    assert not capabilities.supports("voice_clone")
    assert set(provider_setting_types("tts", "elevenlabs")) == {"model", "language"}
    # registered credentials mapping + the factory builds against the right URL fields
    assert pipeline_module._TTS_CREDENTIALS["elevenlabs"] == ("elevenlabs_api_key", "elevenlabs_voice_id")
    service = build_tts(make_config(), "en")
    url = urlparse(service._build_websocket_url())
    assert url.scheme == "wss" and url.netloc == "api.elevenlabs.io"
    assert url.path == "/v1/text-to-speech/default-voice/multi-stream-input"
    assert parse_qs(url.query)["model_id"] == ["eleven_flash_v2_5"]


@pytest.mark.parametrize("bad", ["../../v1/user", "a/b", "a?b=c", "%2e%2e", "", "a b", "x" * 129])
def test_factory_rejects_a_voice_id_that_could_alter_the_websocket_path(bad) -> None:
    with pytest.raises(RuntimeError, match="invalid elevenlabs voice id"):
        build_tts(make_config(elevenlabs_voice_id=bad), "en")


def test_factory_builds_the_real_service_with_voice_model_language_and_destination() -> None:
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    service = build_tts(make_config(), "ko", transport_destination="lane-1")
    assert isinstance(service, ElevenLabsTTSService)
    assert service._api_key == SECRET
    assert service._transport_destination == "lane-1"
    assert service._settings.voice == "default-voice"
    assert service._settings.model == "eleven_flash_v2_5"  # default model
    assert service._settings.language == "ko"
    assert provider_kind(service) == "tts"


def test_factory_honours_session_model_and_does_not_mutate_the_settings() -> None:
    config = make_config(provider_settings={"tts": {"model": "eleven_v3", "language": "en"}})
    service = build_tts(config)
    assert service._settings.model == "eleven_v3"
    assert service._settings.language == "en"
    assert config.provider_settings["tts"] == {"model": "eleven_v3", "language": "en"}
    assert build_tts(make_config())._transport_destination is None


def test_session_start_rejects_cartesia_only_or_unknown_elevenlabs_settings(clean_env) -> None:
    clean_env.setenv("ELEVENLABS_API_KEY", "el-key")
    clean_env.setenv("ELEVENLABS_VOICE_ID", "el-voice")
    config = PipelineConfig.from_environment(
        {
            "providers": {"tts": "elevenlabs"},
            "provider_settings": {"tts": {"model": "eleven_flash_v2_5", "language": "ru"}},
        }
    )
    assert config.provider_settings["tts"]["model"] == "eleven_flash_v2_5"
    with pytest.raises(RuntimeError, match="unsupported tts provider setting: speed"):
        PipelineConfig.from_environment(
            {"providers": {"tts": "elevenlabs"}, "provider_settings": {"tts": {"speed": 1.1}}}
        )


# -- fake WebSocket rig -----------------------------------------------------------------


class FakeSocket:
    """Answers the ElevenLabs multi-context protocol the way Pipecat's code expects."""

    def __init__(self, uri: str, kwargs: dict, *, answer_audio: bool = True) -> None:
        self.uri = uri
        self.kwargs = kwargs
        self.sent: list[dict] = []
        self.state = State.OPEN
        self.closed = False
        self.answer_audio = answer_audio
        self._incoming: asyncio.Queue = asyncio.Queue()

    # -- what the tests look at
    @property
    def voice(self) -> str:
        return urlparse(self.uri).path.split("/")[3]

    @property
    def language_code(self) -> str | None:
        return (parse_qs(urlparse(self.uri).query).get("language_code") or [None])[0]

    @property
    def texts(self) -> list[str]:
        return [m["text"] for m in self.sent if m.get("text", "").strip() and "context_id" in m]

    def messages(self, key: str) -> list[dict]:
        return [m for m in self.sent if m.get(key)]

    # -- websocket surface used by Pipecat
    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        context = message.get("context_id")
        if message.get("text", "").strip() and context and self.answer_audio:
            await self.server_audio(context)
        if message.get("close_context"):
            await self._incoming.put(json.dumps({"contextId": context, "isFinal": True}))
        if message.get("close_socket"):
            await self._incoming.put(None)

    async def server_audio(self, context: str) -> None:
        audio = base64.b64encode(b"\x00\x01" * 480).decode()
        await self._incoming.put(json.dumps({"contextId": context, "audio": audio}))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        message = await self._incoming.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def wait_closed(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True
        self.state = State.CLOSED


class Speaker(FrameProcessor):
    """Stands in for the transport output: records frames and reports the bot stopped
    speaking after each TTSStoppedFrame (which un-pauses the TTS, as the real output does)."""

    def __init__(self) -> None:
        super().__init__(name="Speaker")
        self.frames: list[Frame] = []

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)
        if type(frame).__name__ == "TTSStoppedFrame":
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)


class Rig:
    """A worker running ``[*before, service, Speaker]`` with ``service`` wired to fake sockets."""

    def __init__(self, service, *before, socket_options=None) -> None:
        self.service = service
        self.sockets: list[FakeSocket] = []
        self.socket_options = socket_options or {}
        self.speaker = Speaker()
        self.connect_error: BaseException | None = None

        async def connect(uri, **kwargs):
            if self.connect_error is not None:
                raise self.connect_error
            socket = FakeSocket(uri, kwargs, **self.socket_options)
            self.sockets.append(socket)
            return socket

        service._websocket_connect = connect
        self.worker = PipelineWorker(
            Pipeline([*before, service, self.speaker]),
            params=PipelineParams(enable_metrics=False),
            cancel_on_idle_timeout=False,
        )
        self.runner = WorkerRunner()
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> "Rig":
        started = asyncio.Event()

        @self.worker.event_handler("on_pipeline_started")
        async def _started(*_):
            started.set()

        await self.runner.add_workers(self.worker)
        self._task = asyncio.create_task(self.runner.run())
        await asyncio.wait_for(started.wait(), TIMEOUT)
        return self

    async def __aexit__(self, *_) -> None:
        if not self.worker.has_finished():
            await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self._task, TIMEOUT)

    async def send(self, *frames: Frame) -> None:
        for frame in frames:
            if isinstance(frame, SleepFrame):
                await asyncio.sleep(frame.sleep)
            else:
                await self.worker.queue_frame(frame)

    def audio_contexts(self) -> list[str | None]:
        return [f.context_id for f in self.speaker.frames if isinstance(f, TTSAudioRawFrame)]


def spoken(rig: Rig) -> list[tuple[str, str | None, list[str]]]:
    """Per socket that carried text: (voice id, language_code, texts) -- what was asked for."""

    return [(s.voice, s.language_code, s.texts) for s in rig.sockets if s.texts]


# -- voice and language changes at runtime -------------------------------------------------


def test_voice_and_plain_string_language_updates_reconnect_with_the_new_url() -> None:
    async def scenario():
        service = build_tts(make_config(), "ko", transport_destination="lane-1")
        async with Rig(service) as rig:
            await rig.send(TTSSpeakFrame("안녕하세요.", append_to_context=False), SleepFrame(0.3))
            # exactly what interpreter.py pushes: plain code strings, no Language enum
            await rig.send(
                TTSUpdateSettingsFrame(delta=TTSSettings(language="ru", voice="voice-b")),
                SleepFrame(0.2),
                TTSSpeakFrame("Привет.", append_to_context=False),
                SleepFrame(0.3),
            )
            return rig, service

    rig, service = asyncio.run(scenario())
    assert spoken(rig) == [
        ("default-voice", "ko", ["안녕하세요."]),
        ("voice-b", "ru", ["Привет."]),
    ]
    assert service._settings.language == "ru" and service._settings.voice == "voice-b"
    assert rig.sockets[0].closed  # the first connection was closed, not leaked
    assert {f.transport_destination for f in rig.speaker.frames if isinstance(f, TTSAudioRawFrame)} == {
        "lane-1"
    }
    assert rig.sockets[0].kwargs["additional_headers"] == {"xi-api-key": SECRET}


def test_language_the_model_does_not_list_is_dropped_not_sent() -> None:
    async def scenario():
        service = build_tts(make_config(), "sw")  # Swahili: not in eleven_flash_v2_5's table
        async with Rig(service) as rig:
            await rig.send(TTSSpeakFrame("Habari.", append_to_context=False), SleepFrame(0.3))
            return rig

    rig = asyncio.run(scenario())
    assert spoken(rig) == [("default-voice", None, ["Habari."])]


def test_multilingual_v2_never_gets_a_language_code() -> None:
    async def scenario():
        config = make_config(provider_settings={"tts": {"model": "eleven_multilingual_v2"}})
        async with Rig(build_tts(config, "ko")) as rig:
            await rig.send(TTSSpeakFrame("안녕하세요.", append_to_context=False), SleepFrame(0.3))
            return rig

    rig = asyncio.run(scenario())
    assert spoken(rig) == [("default-voice", None, ["안녕하세요."])]
    assert "model_id=eleven_multilingual_v2" in rig.sockets[0].uri


# -- per-participant voices through the interpreter -----------------------------------------

ALICE, BOB = "browser-1", "guest-2"


def spec(language, voice_id=None):
    return {"language": language, "voice_id": voice_id, "display_name": None}


def turn(text, user_id):
    started, stopped = UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()
    started.metadata["user_id"] = stopped.metadata["user_id"] = user_id
    return [
        started,
        SleepFrame(0.02),
        TranscriptionFrame(text, user_id, "now", None, finalized=True),
        SleepFrame(0.02),
        stopped,
        SleepFrame(0.5),  # translation + fake synthesis + bot-stopped round trip
    ]


def translator(directory, config):
    async def translate(messages):
        return "T:" + messages[-1]["content"]

    return TranslationProcessor(
        settings=InterpreterSettings(languages=directory.languages, direction="auto"),
        translate=translate,
        model="test-model",
        initial_tts_language=directory.languages[1],
        participants=directory,
        default_voice_id=config.tts_voice_id,
    )


def test_each_speakers_voice_is_used_for_their_translation_with_elevenlabs() -> None:
    directory = ParticipantDirectory.from_mapping(
        ("en", "ko"),
        {ALICE: spec("en", "alice-voice"), BOB: spec("ko", "bob-voice")},
    )

    async def scenario():
        config = make_config()
        tts = build_tts(config, directory.languages[1])  # as translation_engine builds lane 0
        async with Rig(tts, translator(directory, config)) as rig:
            await rig.send(
                *turn("Hello there.", ALICE),
                *turn("안녕하세요.", BOB),
                *turn("How are you?", ALICE),
            )
            return rig

    rig = asyncio.run(scenario())
    assert spoken(rig) == [
        ("alice-voice", "ko", ["T:Hello there."]),  # A's voice, translated into ko
        ("bob-voice", "en", ["T:안녕하세요."]),  # B's voice, translated into en
        ("alice-voice", "ko", ["T:How are you?"]),
    ]


def test_participants_without_a_voice_fall_back_to_the_default_voice() -> None:
    directory = ParticipantDirectory.from_mapping(("en", "ko"), {ALICE: spec("en"), BOB: spec("ko")})

    async def scenario():
        config = make_config(elevenlabs_voice_id="session-voice")
        async with Rig(build_tts(config, "ko"), translator(directory, config)) as rig:
            await rig.send(*turn("Hello.", ALICE), *turn("안녕.", BOB))
            return rig

    rig = asyncio.run(scenario())
    assert spoken(rig) == [("session-voice", "ko", ["T:Hello."]), ("session-voice", "en", ["T:안녕."])]


def test_translation_engine_builds_elevenlabs_lanes_with_destinations() -> None:
    """``make_tts`` in translation_engine goes through build_tts with transport_destination."""

    lane_tts = [build_tts(make_config(), "ko", transport_destination=f"lane-{i}") for i in (0, 1)]
    assert [t._transport_destination for t in lane_tts] == ["lane-0", "lane-1"]
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    # a subclass of Pipecat's service (it adds fast reporting of a rejected API key)
    assert all(isinstance(t, ElevenLabsTTSService) for t in lane_tts)
    assert all(provider_kind(t) == "tts" for t in lane_tts)


# -- interruption and cleanup ---------------------------------------------------------------


def test_pipecat_library_interruption_closes_the_inflight_context_and_drops_late_audio() -> None:
    """Library behaviour (Pipecat InterruptionFrame handling with the ElevenLabs protocol)."""
    async def scenario():
        service = build_tts(make_config(), "en")
        async with Rig(service, socket_options={"answer_audio": False}) as rig:
            # an LLM answer still streaming: no LLMFullResponseEndFrame yet, so the turn's
            # context stays open on the ElevenLabs side
            await rig.send(
                LLMFullResponseStartFrame(), TextFrame("A long answer. And then more"), SleepFrame(0.3)
            )
            socket = rig.sockets[0]
            [opened] = [m for m in socket.sent if m.get("text") == " "]
            context = opened["context_id"]
            assert service.audio_context_available(context)  # synthesis is in flight
            await rig.send(InterruptionFrame(), SleepFrame(0.2))
            closed = [m.get("context_id") for m in socket.messages("close_context")]
            gone = not service.audio_context_available(context)
            # audio the server still had in flight for the cancelled context arrives late
            await socket.server_audio(context)
            await rig.send(SleepFrame(0.2))
            late = [c for c in rig.audio_contexts() if c == context]
            # the service is usable afterwards: a new utterance opens a new context
            socket.answer_audio = True
            await rig.send(TTSSpeakFrame("Next.", append_to_context=False), SleepFrame(0.3))
            still_open = not socket.closed
            return rig, context, closed, gone, late, socket, still_open

    rig, context, closed, gone, late, socket, still_open = asyncio.run(scenario())
    assert context in closed  # ElevenLabs told to stop that context (close_context)
    assert gone  # and Pipecat's audio context for it was removed
    assert late == []  # late audio of the cancelled context never reached the output
    assert [t.strip() for t in socket.texts] == ["A long answer.", "Next."]
    assert len({m["context_id"] for m in socket.sent if m.get("text") == " "}) == 2
    assert still_open  # an interruption does not drop the connection


def test_pipecat_library_end_frame_closes_the_socket_and_stops_the_tasks() -> None:
    """Library behaviour (Pipecat ElevenLabsTTSService on EndFrame), not Voice Labs teardown;
    the Voice Labs path is test_session_end_request_tears_down_via_run_pipeline_exactly_once."""
    async def scenario():
        service = build_tts(make_config(), "en")
        async with Rig(service) as rig:
            await rig.send(TTSSpeakFrame("Hello.", append_to_context=False), SleepFrame(0.3))
            socket = rig.sockets[0]
            assert service._websocket is socket and service._receive_task and service._keepalive_task
            receive, keepalive = service._receive_task, service._keepalive_task
        return rig, service, socket, receive, keepalive

    rig, service, socket, receive, keepalive = asyncio.run(scenario())
    assert socket.messages("close_socket") and socket.closed
    assert service._websocket is None
    assert service._receive_task is None and service._keepalive_task is None
    assert receive.done() and keepalive.done()
    assert len(rig.sockets) == 1  # no reconnect on the way out


def test_pipecat_library_worker_cancel_closes_the_socket_too() -> None:
    """Library behaviour (Pipecat worker.cancel), not Voice Labs teardown."""
    async def scenario():
        service = build_tts(make_config(), "en")
        rig = Rig(service)
        async with rig:
            await rig.send(TTSSpeakFrame("Hello.", append_to_context=False), SleepFrame(0.2))
            await rig.worker.cancel(reason="test")
        return rig, service

    rig, service = asyncio.run(scenario())
    assert rig.sockets[0].closed
    assert service._websocket is None and service._receive_task is None


# -- failures: the same handling a Cartesia failure gets ---------------------------------------

PAYLOAD_KEYS = {"provider", "kind", "category", "stage", "retryable", "attempt", "message"}


class RecordingClient:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.event_type for e in self.events]

    def of(self, event_type: str) -> list[dict]:
        return [e.payload for e in self.events if e.event_type == event_type]


class FakeTransport:
    def __init__(self) -> None:
        self.cleaned = False

    def add_event_handler(self, name, handler) -> None:
        pass

    async def cleanup(self) -> None:
        self.cleaned = True


def handshake_rejected(status: int) -> Exception:
    """What ``websockets`` raises when the opening handshake gets an HTTP error status."""

    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    return InvalidStatus(Response(status, "Rejected", Headers(), b""))


async def run_failing_session(monkeypatch, rig: Rig, frames, config=None):
    client = RecordingClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    watch_provider_failures(rig.worker, session, ELEVEN_PROVIDERS)
    transport = FakeTransport()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, rig.worker))
    started = asyncio.Event()

    @rig.worker.event_handler("on_pipeline_started")
    async def _started(*_):
        started.set()

    run = asyncio.create_task(session.run_pipeline(config or make_config()))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    for frame in frames:
        if rig.worker.has_finished():
            break
        await rig.worker.queue_frame(frame)
        await asyncio.sleep(0.3)
    await asyncio.wait_for(run, TIMEOUT)  # no hang: the failure ends the call by itself
    return client, session, transport


def assert_failed_like_any_provider(client, session, transport, *, reason: str) -> None:
    failed = client.of("ProviderError")[-1]
    assert (failed["kind"], failed["provider"], failed["category"]) == ("tts", "elevenlabs", reason)
    assert set(failed) == PAYLOAD_KEYS and failed["retryable"] is False
    assert client.events, "no events recorded: the secrecy check would pass vacuously"
    assert "ProviderError" in client.types()
    serialized = json.dumps([[e.event_type, e.payload] for e in client.events], default=str)
    assert SECRET not in serialized
    assert SECRET not in repr(client.events)
    assert SECRET not in failed["message"]
    assert client.of("SessionFailed") == [{"outcome": "failed", "reason": "provider_failed"}]
    assert session.state == SessionState.ENDED
    assert transport.cleaned and session._pipeline_worker is None


def test_session_end_request_tears_down_via_run_pipeline_exactly_once(monkeypatch) -> None:
    """Voice Labs' own path: RuntimeSession.run_pipeline + request_end with the real
    ElevenLabs service on a fake socket: socket closed, transport cleaned, terminal event once."""

    async def scenario():
        service = build_tts(make_config(), "en")
        rig = Rig(service)
        client = RecordingClient()
        session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
        transport = FakeTransport()
        monkeypatch.setattr(
            src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, rig.worker)
        )
        started = asyncio.Event()

        @rig.worker.event_handler("on_pipeline_started")
        async def _started(*_):
            started.set()

        run = asyncio.create_task(session.run_pipeline(make_config()))
        await asyncio.wait_for(started.wait(), TIMEOUT)
        await rig.worker.queue_frame(TTSSpeakFrame("Hello.", append_to_context=False))
        await asyncio.sleep(0.3)
        assert rig.sockets and not rig.sockets[0].closed
        await session.request_end("test end")
        await asyncio.wait_for(run, TIMEOUT)
        return client, session, transport, rig, service

    client, session, transport, rig, service = asyncio.run(scenario())
    assert len(rig.sockets) == 1 and rig.sockets[0].closed
    assert rig.sockets[0].messages("close_socket")
    assert service._websocket is None and service._receive_task is None
    assert transport.cleaned and session._pipeline_worker is None
    assert session.state == SessionState.ENDED
    assert client.types().count("SessionEnded") == 1
    assert client.of("SessionFailed") == [] and client.of("ProviderError") == []


def test_rejected_api_key_on_connect_ends_the_session_as_an_auth_failure(monkeypatch) -> None:
    async def scenario():
        rig = Rig(build_tts(make_config(), "en"))
        rig.connect_error = handshake_rejected(401)  # the connect in setup() fails
        return await run_failing_session(monkeypatch, rig, [])

    assert_failed_like_any_provider(*asyncio.run(scenario()), reason="auth")


class RejectingSocket(FakeSocket):
    """What the REAL ElevenLabs API does with an invalid key (observed 2026-11-05): the handshake
    succeeds, then the server closes with 1008 "Invalid API key" once text arrives."""

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        if message.get("text", "").strip() and message.get("context_id"):
            await self._incoming.put(ConnectionClosedError(Close(1008, "Invalid API key"), None))

    async def __anext__(self) -> str:
        item = await self._incoming.get()
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item


def test_in_band_key_rejection_ends_the_session_at_once_as_an_auth_failure(monkeypatch) -> None:
    """The real API rejects a bad key after the handshake (1008). The call must fail as `auth`
    on the first rejection, not be retried and then reported as a generic failure."""

    async def scenario():
        service = build_tts(make_config(), "en")
        rig = Rig(service)

        async def connect(uri, **kwargs):
            socket = RejectingSocket(uri, kwargs)
            rig.sockets.append(socket)
            return socket

        service._websocket_connect = connect
        speak = TTSSpeakFrame("Hello.", append_to_context=False)
        client, session, transport = await run_failing_session(monkeypatch, rig, [speak, speak])
        return client, session, transport, rig

    client, session, transport, rig = asyncio.run(scenario())
    assert_failed_like_any_provider(client, session, transport, reason="auth")
    assert len(rig.sockets) == 1  # no reconnect attempts against a key that will not work


def test_quota_error_text_from_elevenlabs_ends_the_session(monkeypatch) -> None:
    async def scenario():
        rig = Rig(build_tts(make_config(), "en"))
        rig.connect_error = RuntimeError("quota_exceeded: this request exceeds your quota")
        return await run_failing_session(monkeypatch, rig, [])

    assert_failed_like_any_provider(*asyncio.run(scenario()), reason="quota")


def test_dropped_websocket_mid_call_is_a_retried_unavailable_error_then_ends_the_call(
    monkeypatch,
) -> None:
    async def scenario():
        service = build_tts(make_config(), "en")
        rig = Rig(service)

        async def fail_send(raw):
            raise ConnectionClosedError(Close(1011, "internal error"), None)

        # the socket drops as soon as text is sent; every reconnect drops again
        original_connect = service._websocket_connect

        async def connect(uri, **kwargs):
            socket = await original_connect(uri, **kwargs)
            socket.send = fail_send
            return socket

        service._websocket_connect = connect
        speak = TTSSpeakFrame("Hello.", append_to_context=False)
        return await run_failing_session(monkeypatch, rig, [speak, speak, speak, speak])

    client, session, transport = asyncio.run(scenario())
    errors = client.of("ProviderError")
    assert errors[0]["retryable"] is True and errors[0]["attempt"] == 1  # announced, call goes on
    assert errors[0]["category"] == "unavailable"  # a dropped socket, like Cartesia's
    assert [e["kind"] for e in errors] == ["tts"] * len(errors)
    assert {e["provider"] for e in errors} == {"elevenlabs"}
    last = errors[-1]
    assert (last["category"], last["attempt"], last["retryable"]) == ("unavailable", 2, False)
    assert session.state == SessionState.ENDED and transport.cleaned
    assert client.of("SessionFailed") == [{"outcome": "failed", "reason": "provider_failed"}]


def test_cascaded_fallback_uses_the_selected_tts_providers_credentials() -> None:
    realtime = {"engine": "realtime", "languages": ["en", "ko"]}
    eleven = make_config(mode="interpreter", interpreter=realtime)
    assert cascaded_fallback_config(eleven).interpreter["engine"] == "cascaded"
    no_voice = make_config(mode="interpreter", interpreter=realtime, elevenlabs_voice_id="")
    assert cascaded_fallback_config(no_voice) is None
    # a Cartesia-only key does not make an ElevenLabs session feasible
    wrong_key = make_config(
        mode="interpreter", interpreter=realtime, elevenlabs_api_key="", cartesia_api_key="ct", cartesia_voice_id="v"
    )
    assert cascaded_fallback_config(wrong_key) is None


# -- unsupported voice operations for ElevenLabs --------------------------------------------


@pytest.fixture
def runtime_api(monkeypatch):
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    calls: list = []

    class Boom:
        def __init__(self, *args, **kwargs) -> None:
            calls.append("httpx")
            raise AssertionError("no provider call expected")

    monkeypatch.setattr(httpx, "AsyncClient", Boom)
    with TestClient(api.app) as client:
        yield client, calls


def test_cloning_an_elevenlabs_voice_is_rejected_without_a_provider_call(runtime_api) -> None:
    client, calls = runtime_api
    response = client.post(
        "/voices/clone",
        json={
            "provider": "elevenlabs",
            "name": "Anna",
            "language": "en",
            "sample_url": "https://example.com/sample.wav",
        },
        headers={"X-Runtime-Token": "token"},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported_tts_provider"}
    assert calls == []


def test_deleting_an_elevenlabs_voice_is_unknown_provider_without_a_provider_call(runtime_api) -> None:
    client, calls = runtime_api
    response = client.delete("/voices/elevenlabs/voice123", headers={"X-Runtime-Token": "token"})
    assert response.status_code == 404
    assert response.json() == {"code": "unknown_provider"}
    assert calls == []
