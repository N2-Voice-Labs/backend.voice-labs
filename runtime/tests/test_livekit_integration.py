"""Opt-in LiveKit transport acceptance checks with inference providers stubbed."""

import asyncio
import contextlib
import importlib
import math
import os
from array import array
from uuid import uuid4

import pytest
import httpx

frame_processor = pytest.importorskip("pipecat.processors.frame_processor")
pipecat_frames = pytest.importorskip("pipecat.frames.frames")
livekit_api = pytest.importorskip("livekit.api")
livekit_rtc = pytest.importorskip("livekit.rtc")
FrameProcessor = frame_processor.FrameProcessor
InputAudioRawFrame = pipecat_frames.InputAudioRawFrame
TranscriptionFrame = pipecat_frames.TranscriptionFrame
PipelineConfig = importlib.import_module("src.runtime.pipeline").PipelineConfig
RuntimeSession = importlib.import_module("src.runtime.session").RuntimeSession
runtime_api = importlib.import_module("src.runtime.api")

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
LIVEKIT_CONFIGURED = bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET)


class NoProviderService(FrameProcessor):
    """Stand in for external inference APIs while keeping Pipecat and LiveKit real."""

    audio_received = None
    transcription_emitted = False

    class Settings:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs

    instances = []

    def __init__(self, *args, **kwargs) -> None:
        self.settings = kwargs.get("settings")
        self.instances.append(self)
        super().__init__(*args, **kwargs)

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            if self.audio_received is not None:
                self.audio_received.set()
            if not NoProviderService.transcription_emitted:
                NoProviderService.transcription_emitted = True
                await self.push_frame(
                    TranscriptionFrame(
                        "synthetic ingress transcript",
                        "participant-1",
                        "integration-test",
                        finalized=True,
                    ),
                    direction,
                )


class RecordingEventClient:
    def __init__(self) -> None:
        self.events = []
        self.runtime_started = asyncio.Event()
        self.transcript_received = asyncio.Event()

    async def emit(self, event) -> dict[str, str]:
        self.events.append(event)
        if event.event_type == "RuntimeStarted":
            self.runtime_started.set()
        if event.event_type == "TranscriptFinal":
            self.transcript_received.set()
        return {"status": "accepted"}

    async def aclose(self) -> None:
        return None


def test_runtime_joins_requested_livekit_room_and_leaves_after_explicit_end(
    monkeypatch,
) -> None:
    service_modules = _provider_service_modules()
    if not LIVEKIT_CONFIGURED:
        pytest.skip("Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET to run")

    asyncio.run(
        _verify_livekit_session(monkeypatch, service_modules, disconnect_remotely=False)
    )


def test_runtime_ends_when_livekit_removes_its_participant(monkeypatch) -> None:
    service_modules = _provider_service_modules()
    if not LIVEKIT_CONFIGURED:
        pytest.skip("Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET to run")

    asyncio.run(
        _verify_livekit_session(monkeypatch, service_modules, disconnect_remotely=True)
    )


def test_runtime_ends_when_browser_participant_disconnects(monkeypatch) -> None:
    service_modules = _provider_service_modules()
    if not LIVEKIT_CONFIGURED:
        pytest.skip("Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET to run")

    asyncio.run(
        _verify_livekit_session(
            monkeypatch,
            service_modules,
            disconnect_remotely=False,
            disconnect_browser=True,
            browser_removed_remotely=False,
        )
    )


def test_runtime_ends_when_livekit_removes_browser_participant(monkeypatch) -> None:
    service_modules = _provider_service_modules()
    if not LIVEKIT_CONFIGURED:
        pytest.skip("Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET to run")

    asyncio.run(
        _verify_livekit_session(
            monkeypatch,
            service_modules,
            disconnect_remotely=False,
            disconnect_browser=True,
            browser_removed_remotely=True,
        )
    )


def test_runtime_receives_audio_published_to_its_livekit_room(monkeypatch) -> None:
    service_modules = _provider_service_modules()
    if not LIVEKIT_CONFIGURED:
        pytest.skip("Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET to run")

    asyncio.run(_verify_livekit_audio_ingress(monkeypatch, service_modules))


def test_authenticated_runtime_handoff_auto_joins_requested_room(monkeypatch) -> None:
    service_modules = _provider_service_modules()
    if not LIVEKIT_CONFIGURED:
        pytest.skip("Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET to run")

    asyncio.run(_verify_runtime_api_handoff(monkeypatch, service_modules))


def _provider_service_modules():
    return (
        pytest.importorskip("pipecat.services.deepgram.stt"),
        pytest.importorskip("pipecat.services.openai.llm"),
        pytest.importorskip("pipecat.services.cartesia.tts"),
    )


async def _verify_livekit_audio_ingress(monkeypatch, service_modules) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "integration-test-stub")
    monkeypatch.setenv("OPENAI_API_KEY", "integration-test-stub")
    monkeypatch.setenv("CARTESIA_API_KEY", "integration-test-stub")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "integration-test-stub")
    monkeypatch.setenv("RUNTIME_LIVEKIT_TOKEN", "")

    NoProviderService.audio_received = asyncio.Event()
    NoProviderService.transcription_emitted = False
    room_name = f"voice-labs-audio-test-{uuid4()}"
    config = PipelineConfig.from_environment(
        {"room_name": room_name, "system_prompt": "test", "voice_id": "stub"}
    )
    event_client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), event_client)
    runtime_task = None
    publisher = livekit_rtc.Room()
    source = livekit_rtc.AudioSource(sample_rate=48000, num_channels=1)

    try:
        deepgram_stt, openai_llm, cartesia_tts = service_modules
        monkeypatch.setattr(deepgram_stt, "DeepgramSTTService", NoProviderService)
        monkeypatch.setattr(openai_llm, "OpenAILLMService", NoProviderService)
        monkeypatch.setattr(cartesia_tts, "CartesiaTTSService", NoProviderService)

        runtime_task = asyncio.create_task(session.run_pipeline(config))
        await asyncio.wait_for(event_client.runtime_started.wait(), timeout=15)

        publisher_identity = f"voice-labs-test-publisher-{uuid4()}"
        publisher_token = (
            livekit_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(publisher_identity)
            .with_grants(
                livekit_api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=False,
                )
            )
            .to_jwt()
        )
        await publisher.connect(LIVEKIT_URL, publisher_token)
        track = livekit_rtc.LocalAudioTrack.create_audio_track(
            "integration-test-tone", source
        )
        await publisher.local_participant.publish_track(track)

        sample_rate = 48000
        samples_per_frame = 480
        for frame_index in range(30):
            samples = array(
                "h",
                (
                    int(
                        8000
                        * math.sin(
                            2
                            * math.pi
                            * 440
                            * (frame_index * samples_per_frame + index)
                            / sample_rate
                        )
                    )
                    for index in range(samples_per_frame)
                ),
            )
            await source.capture_frame(
                livekit_rtc.AudioFrame(
                    data=samples.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=samples_per_frame,
                )
            )

        await asyncio.wait_for(NoProviderService.audio_received.wait(), timeout=10)
        await asyncio.wait_for(event_client.transcript_received.wait(), timeout=1)
        assert any(
            event.event_type == "RuntimeStarted" for event in event_client.events
        )
        transcript = next(
            event
            for event in event_client.events
            if event.event_type == "TranscriptFinal"
        )
        assert transcript.payload["content"] == "synthetic ingress transcript"
        assert transcript.payload["role"] == "user"
    finally:
        await publisher.disconnect()
        await source.aclose()
        if runtime_task is not None and not runtime_task.done():
            with contextlib.suppress(Exception):
                await session.request_end()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(runtime_task, timeout=10)


async def _verify_runtime_api_handoff(monkeypatch, service_modules) -> None:
    runtime_token = "integration-runtime-token"
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", runtime_token)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "integration-test-stub")
    monkeypatch.setenv("OPENAI_API_KEY", "integration-test-stub")
    monkeypatch.setenv("CARTESIA_API_KEY", "integration-test-stub")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "integration-test-stub")
    monkeypatch.setenv("RUNTIME_LIVEKIT_TOKEN", "")

    event_client = RecordingEventClient()
    event_clients = []

    def create_event_client(platform_url, token):
        assert token == runtime_token
        event_clients.append(event_client)
        return event_client

    monkeypatch.setattr(runtime_api, "PlatformEventClient", create_event_client)
    NoProviderService.audio_received = None
    NoProviderService.instances = []
    for module, service_name in zip(
        service_modules,
        ("DeepgramSTTService", "OpenAILLMService", "CartesiaTTSService"),
    ):
        monkeypatch.setattr(module, service_name, NoProviderService)

    room_name = f"voice-labs-api-handoff-{uuid4()}"
    session_id = uuid4()
    headers = {"X-Runtime-Token": runtime_token}
    payload = {
        "session_id": str(session_id),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": room_name,
        "system_prompt": "integration prompt",
        "voice_id": "integration voice",
        "providers": {
            "stt": "deepgram",
            "llm": "openai-compatible",
            "tts": "cartesia",
        },
        "provider_settings": {
            "stt": {"model": "nova-3", "language": "multi"},
            "llm": {"model": "gpt-4o-mini", "temperature": 0.4},
            "tts": {"model": "sonic-2", "language": "en"},
        },
    }
    livekit = livekit_api.LiveKitAPI(
        _livekit_http_url(LIVEKIT_URL),
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

    try:
        transport = httpx.ASGITransport(app=runtime_api.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://runtime.test"
        ) as client:
            accepted = await client.post("/sessions", json=payload, headers=headers)
            assert accepted.status_code == 202
            assert accepted.json() == {"status": "accepted"}
            await asyncio.wait_for(event_client.runtime_started.wait(), timeout=15)

            status = await client.get(f"/sessions/{session_id}", headers=headers)
            assert status.json()["status"] == "running"
            participants = await livekit.room.list_participants(
                livekit_api.ListParticipantsRequest(room=room_name)
            )
            assert len(participants.participants) == 1
            assert participants.participants[0].identity.startswith("voice-labs-agent-")
            assert event_client.events[0].payload["room_name"] == room_name
            assert len(event_clients) == 1

            assert len(NoProviderService.instances) == 3
            service_settings = {
                instance.settings.values.get("model"): instance.settings.values
                for instance in NoProviderService.instances
            }
            assert service_settings["nova-3"]["language"] == "multi"
            assert service_settings["gpt-4o-mini"]["temperature"] == 0.4
            assert service_settings["sonic-2"]["language"] == "en"

            ended = await client.post(f"/sessions/{session_id}/end", headers=headers)
            assert ended.status_code == 200
            assert ended.json() == {"status": "ended"}
            assert [event.event_type for event in event_client.events][
                -1
            ] == "SessionEnded"
            remaining = await livekit.room.list_participants(
                livekit_api.ListParticipantsRequest(room=room_name)
            )
            assert remaining.participants == []
    finally:
        task = runtime_api._active_sessions.get(str(session_id))
        if task and not task.done():
            session = runtime_api._runtime_sessions.get(str(session_id))
            if session is not None:
                await session.request_end()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=10)
        await livekit.aclose()


async def _verify_livekit_session(
    monkeypatch,
    service_modules,
    *,
    disconnect_remotely: bool,
    disconnect_browser: bool = False,
    browser_removed_remotely: bool = False,
) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "integration-test-stub")
    monkeypatch.setenv("OPENAI_API_KEY", "integration-test-stub")
    monkeypatch.setenv("CARTESIA_API_KEY", "integration-test-stub")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "integration-test-stub")
    monkeypatch.setenv("RUNTIME_LIVEKIT_TOKEN", "")

    room_name = f"voice-labs-runtime-test-{uuid4()}"
    config = PipelineConfig.from_environment(
        {"room_name": room_name, "system_prompt": "test", "voice_id": "stub"}
    )
    event_client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), event_client)
    runtime_task = None
    livekit = livekit_api.LiveKitAPI(
        _livekit_http_url(LIVEKIT_URL),
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

    try:
        deepgram_stt, openai_llm, cartesia_tts = service_modules
        monkeypatch.setattr(deepgram_stt, "DeepgramSTTService", NoProviderService)
        monkeypatch.setattr(openai_llm, "OpenAILLMService", NoProviderService)
        monkeypatch.setattr(cartesia_tts, "CartesiaTTSService", NoProviderService)
        runtime_task = asyncio.create_task(session.run_pipeline(config))
        await asyncio.wait_for(event_client.runtime_started.wait(), timeout=15)

        participants = await livekit.room.list_participants(
            livekit_api.ListParticipantsRequest(room=room_name)
        )
        assert len(participants.participants) == 1
        participant = participants.participants[0]
        assert participant.identity.startswith("voice-labs-agent-")
        started_event = next(
            event
            for event in event_client.events
            if event.event_type == "RuntimeStarted"
        )
        assert started_event.payload["room_name"] == room_name

        if disconnect_browser:
            browser = livekit_rtc.Room()
            browser_identity = f"browser-{session.session_id}"
            browser_token = (
                livekit_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                .with_identity(browser_identity)
                .with_grants(
                    livekit_api.VideoGrants(
                        room_join=True,
                        room=room_name,
                        can_publish=True,
                        can_subscribe=True,
                    )
                )
                .to_jwt()
            )
            await browser.connect(LIVEKIT_URL, browser_token)
            joined = await livekit.room.list_participants(
                livekit_api.ListParticipantsRequest(room=room_name)
            )
            assert {item.identity for item in joined.participants} == {
                participant.identity,
                browser_identity,
            }
            if browser_removed_remotely:
                await livekit.room.remove_participant(
                    livekit_api.RoomParticipantIdentity(
                        room=room_name, identity=browser_identity
                    )
                )
            else:
                await browser.disconnect()
            for _ in range(50):
                current = await livekit.room.list_participants(
                    livekit_api.ListParticipantsRequest(room=room_name)
                )
                if browser_identity not in {
                    item.identity for item in current.participants
                }:
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("LiveKit did not remove the disconnected browser")
        elif disconnect_remotely:
            await livekit.room.remove_participant(
                livekit_api.RoomParticipantIdentity(
                    room=room_name, identity=participant.identity
                )
            )
        else:
            assert await session.request_end()

        disconnect_timeout = (
            10 if browser_removed_remotely else 15 if disconnect_browser else 10
        )
        await asyncio.wait_for(runtime_task, timeout=disconnect_timeout)
        terminal_events = [
            event.event_type
            for event in event_client.events
            if event.event_type in {"SessionEnded", "SessionFailed"}
        ]
        assert terminal_events == ["SessionEnded"]
        if disconnect_browser:
            terminal_event = next(
                event
                for event in event_client.events
                if event.event_type == "SessionEnded"
            )
            assert terminal_event.payload["outcome"] == "ended"
            assert terminal_event.payload["reason"] in {
                "Browser participant disconnected",
                "LiveKit disconnected",
            }
        elif not disconnect_remotely:
            terminal_event = next(
                event
                for event in event_client.events
                if event.event_type == "SessionEnded"
            )
            assert terminal_event.payload == {
                "outcome": "ended",
                "reason": "Session ended by request",
            }
        remaining = await livekit.room.list_participants(
            livekit_api.ListParticipantsRequest(room=room_name)
        )
        assert remaining.participants == []
    finally:
        if disconnect_browser and "browser" in locals():
            with contextlib.suppress(Exception):
                await browser.disconnect()
        if runtime_task is not None and not runtime_task.done():
            with contextlib.suppress(Exception):
                await session.request_end()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(runtime_task, timeout=10)
        await livekit.aclose()


def _livekit_http_url(url: str) -> str:
    if url.startswith("ws://"):
        return f"http://{url.removeprefix('ws://')}"
    if url.startswith("wss://"):
        return f"https://{url.removeprefix('wss://')}"
    return url
