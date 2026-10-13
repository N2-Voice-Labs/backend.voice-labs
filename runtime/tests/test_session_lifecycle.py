import asyncio
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from src.runtime import api
from src.runtime.pipeline import PipelineConfig
from src.runtime.session import RuntimeSession, SessionState


class RecordingEventClient:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


class FakeTransport:
    def __init__(self) -> None:
        self.handlers = {}
        self.pending_event_tasks = []

    def add_event_handler(self, name, handler) -> None:
        self.handlers[name] = handler

    async def cleanup(self) -> None:
        if self.pending_event_tasks:
            await asyncio.gather(*self.pending_event_tasks)


class FakePipelineTask:
    def __init__(self) -> None:
        self.cancel_reasons = []

    def cancel(self, *, reason=None) -> None:
        self.cancel_reasons.append(reason)


def pipeline_config() -> PipelineConfig:
    return PipelineConfig("ws://livekit", "token", "room-42", "deepgram", "openai", "cartesia", "voice", "prompt")


def test_runtime_started_follows_livekit_connection_and_ends_once(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    transport = FakeTransport()
    task = FakePipelineTask()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, task))

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            assert worker is task

        async def run(self) -> None:
            assert client.events == []
            await transport.handlers["on_connected"]()

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(session.run_pipeline(pipeline_config()))

    assert [event.event_type for event in client.events] == ["RuntimeStarted", "SessionEnded"]
    assert client.events[0].payload == {"room_name": "room-42"}
    assert session.state == SessionState.ENDED


def test_pipeline_waits_for_background_runtime_started_delivery(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    class BlockingEventClient(RecordingEventClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def emit(self, event) -> None:
            if event.event_type == "RuntimeStarted":
                self.started.set()
                await self.release.wait()
            self.events.append(event)

    client = BlockingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    transport = FakeTransport()
    task = FakePipelineTask()
    transcript_enqueued = asyncio.Event()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, task))

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            assert worker is task

        async def run(self) -> None:
            transport.pending_event_tasks.append(
                asyncio.create_task(transport.handlers["on_connected"](transport))
            )
            await client.started.wait()
            session.enqueue_event(session.event("TranscriptFinal", {"content": "hello"}))
            transcript_enqueued.set()

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)

    async def exercise_pipeline() -> None:
        pipeline_run = asyncio.create_task(session.run_pipeline(pipeline_config()))
        await client.started.wait()
        await transcript_enqueued.wait()
        assert not pipeline_run.done()
        assert client.events == []
        client.release.set()
        await pipeline_run

    asyncio.run(exercise_pipeline())

    assert [event.event_type for event in client.events] == [
        "RuntimeStarted",
        "TranscriptFinal",
        "SessionEnded",
    ]


def test_livekit_disconnect_cancels_pipecat_task(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    transport = FakeTransport()
    task = FakePipelineTask()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, task))

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            assert worker is task

        async def run(self) -> None:
            await transport.handlers["on_connected"]()
            await transport.handlers["on_disconnected"]()

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(session.run_pipeline(pipeline_config()))

    assert task.cancel_reasons == ["LiveKit disconnected"]
    assert [event.event_type for event in client.events] == ["RuntimeStarted", "SessionEnded"]
    assert session.state == SessionState.ENDED


def test_livekit_participant_disconnect_cancels_pipecat_task(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    transport = FakeTransport()
    task = FakePipelineTask()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, task))

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            assert worker is task

        async def run(self) -> None:
            await transport.handlers["on_connected"](transport)
            await transport.handlers["on_participant_disconnected"](
                transport, "PA_remote-participant-sid"
            )

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(session.run_pipeline(pipeline_config()))

    assert task.cancel_reasons == ["participant_disconnected"]
    assert [event.event_type for event in client.events] == ["RuntimeStarted", "SessionEnded"]
    assert session.state == SessionState.ENDED


def test_pipeline_exception_emits_one_session_failure(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    transport = FakeTransport()
    task = FakePipelineTask()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, task))

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            assert worker is task

        async def run(self) -> None:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(session.run_pipeline(pipeline_config()))

    assert [event.event_type for event in client.events] == ["SessionFailed"]
    assert client.events[0].payload == {"outcome": "failed", "reason": "RuntimeError"}
    assert session.state == SessionState.ENDED


def test_request_end_cancels_native_pipeline_worker(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    transport = FakeTransport()
    task = FakePipelineTask()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, task))

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            assert worker is task

        async def run(self) -> None:
            assert await session.request_end()
            assert task.cancel_reasons == ["Session ended by request"]

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(session.run_pipeline(pipeline_config()))


def test_request_end_awaits_async_pipecat_cancellation(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    transport = FakeTransport()

    class AsyncPipelineTask:
        def __init__(self) -> None:
            self.cancellation_completed = False

        async def cancel(self, reason: str) -> None:
            await asyncio.sleep(0)
            self.cancellation_completed = True

    task = AsyncPipelineTask()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, task))

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            assert worker is task

        async def run(self) -> None:
            assert await session.request_end()
            assert task.cancellation_completed

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(session.run_pipeline(pipeline_config()))


def test_provider_configuration_failure_does_not_emit_runtime_started(monkeypatch) -> None:
    class FakePlatformEventClient(RecordingEventClient):
        def __init__(self, platform_url, runtime_token) -> None:
            super().__init__()
            client_instance[0] = self

        async def aclose(self) -> None:
            self.closed = True

    client_instance = [None]
    monkeypatch.setattr(api, "PlatformEventClient", FakePlatformEventClient)
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "runtime-token")
    for name in (
        "LIVEKIT_URL",
        "RUNTIME_LIVEKIT_TOKEN",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "CARTESIA_API_KEY",
        "CARTESIA_VOICE_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    request = api.SessionStart(
        session_id=uuid4(),
        conversation_id=uuid4(),
        agent_id=uuid4(),
        room_name="room-42",
    )
    with pytest.raises(RuntimeError, match="runtime provider configuration is missing"):
        asyncio.run(api.run_session(request))

    emitted = client_instance[0].events
    assert [event.event_type for event in emitted] == ["SessionFailed"]
    assert emitted[0].payload == {"outcome": "failed", "reason": "RuntimeError"}
    assert client_instance[0].closed is True
    assert api._session_states[str(request.session_id)] == "failed"
