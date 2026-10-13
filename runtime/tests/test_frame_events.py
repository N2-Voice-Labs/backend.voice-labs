import asyncio
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transcriptions.language import Language

from src.runtime.frame_events import RuntimeEventProcessor
from src.runtime.session import RuntimeSession, SessionState


class RecordingEventClient:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


async def process(processor, frame) -> None:
    processor.push_frame = lambda forwarded, direction: _record_forwarded(
        processor, forwarded, direction
    )
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)


async def _record_forwarded(processor, frame, direction) -> None:
    processor.forwarded.append((frame, direction))


def test_final_user_transcription_emits_persistable_transcript_event() -> None:
    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    processor = RuntimeEventProcessor(session)
    processor.forwarded = []

    async def run_frames() -> None:
        await process(
            processor,
            TranscriptionFrame("hello there", "participant-1", "now", finalized=False),
        )
        await process(
            processor,
            TranscriptionFrame(
                "hello there",
                "participant-1",
                "now",
                language=Language.EN,
                finalized=True,
            ),
        )
        await session.close_event_dispatcher()

    asyncio.run(run_frames())

    assert [(event.event_type, event.payload) for event in client.events] == [
        (
            "TranscriptFinal",
            {
                "content": "hello there",
                "role": "user",
                "user_id": "participant-1",
                "timestamp": "now",
                "primary_language": "en",
                "detected_languages": ["en"],
            },
        )
    ]
    assert len(processor.forwarded) == 2


def test_final_transcript_frame_is_forwarded_while_platform_delivery_is_slow() -> None:
    class BlockingEventClient(RecordingEventClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def emit(self, event) -> None:
            self.started.set()
            await self.release.wait()
            self.events.append(event)

    client = BlockingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    processor = RuntimeEventProcessor(session)
    processor.forwarded = []

    async def run_frame() -> None:
        process_task = asyncio.create_task(
            process(
                processor,
                TranscriptionFrame("hello there", "participant-1", "now", finalized=True),
            )
        )
        try:
            await client.started.wait()
            await process(processor, UserStartedSpeakingFrame())
            assert len(processor.forwarded) == 2
        finally:
            client.release.set()
        await process_task
        await session.close_event_dispatcher()

    asyncio.run(run_frame())
    assert [event.event_type for event in client.events] == [
        "TranscriptFinal",
        "SpeechStarted",
    ]


def test_platform_event_failure_cancels_the_live_pipeline() -> None:
    class FailingEventClient(RecordingEventClient):
        async def emit(self, event) -> None:
            raise ConnectionError("Platform is unavailable")

    class PipelineTask:
        def __init__(self) -> None:
            self.cancel_reason = None
            self.cancel_completed = False

        async def cancel(self, *, reason=None) -> None:
            self.cancel_reason = reason
            self.cancel_completed = True

    client = FailingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    pipeline_task = PipelineTask()
    session._pipeline_worker = pipeline_task
    processor = RuntimeEventProcessor(session)
    processor.forwarded = []

    async def run_frame() -> None:
        await process(
            processor,
            TranscriptionFrame("hello there", "participant-1", "now", finalized=True),
        )
        await asyncio.sleep(0)
        await session.close_event_dispatcher()

    asyncio.run(run_frame())

    assert pipeline_task.cancel_reason == "Platform event delivery failed"
    assert pipeline_task.cancel_completed is True


def test_event_queue_overflow_cancels_the_live_pipeline() -> None:
    class PipelineTask:
        def __init__(self) -> None:
            self.cancel_reason = None
            self.cancel_completed = False

        async def cancel(self, *, reason=None) -> None:
            self.cancel_reason = reason
            self.cancel_completed = True

    session = RuntimeSession(uuid4(), uuid4(), uuid4(), RecordingEventClient())
    pipeline_task = PipelineTask()
    session._pipeline_worker = pipeline_task

    async def fill_queue() -> int:
        accepted = 0
        try:
            for _ in range(1024):
                session.enqueue_event(session.event("SpeechStarted"))
                accepted += 1
        except RuntimeError as exc:
            assert str(exc) == "runtime event queue capacity exceeded"
        else:
            pytest.fail("event queue accepted more events than its bounded capacity")
        await session.close_event_dispatcher()
        return accepted

    accepted = asyncio.run(fill_queue())

    assert 0 < accepted < 1024
    assert pipeline_task.cancel_reason == "Runtime event queue capacity exceeded"
    assert pipeline_task.cancel_completed is True


def test_event_dispatcher_shutdown_bounds_wait_for_cancellation_resistant_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancellationResistantEventClient(RecordingEventClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def emit(self, event) -> None:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
            self.events.append(event)

    monkeypatch.setattr("src.runtime.session._EVENT_DRAIN_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("src.runtime.session._EVENT_WORKER_STOP_TIMEOUT_SECONDS", 0.01)
    client = CancellationResistantEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)

    async def close_dispatcher() -> None:
        session.enqueue_event(session.event("SpeechStarted"))
        await client.started.wait()
        worker = session._event_delivery_task
        shutdown = asyncio.create_task(session.close_event_dispatcher())
        completed, _ = await asyncio.wait({shutdown}, timeout=0.2)
        assert shutdown in completed
        assert await shutdown is False
        assert worker is not None and not worker.done()
        client.release.set()
        await asyncio.sleep(0)
        assert await session.close_event_dispatcher() is True
        assert worker.done()

    asyncio.run(close_dispatcher())


def test_assistant_text_emits_once_at_full_response_end() -> None:
    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    processor = RuntimeEventProcessor(
        session, assistant_output=True
    )
    processor.forwarded = []

    async def run_frames() -> None:
        await process(processor, TextFrame("Good "))
        await process(processor, TextFrame("morning."))
        assert client.events == []
        await process(processor, LLMFullResponseEndFrame())
        await session.close_event_dispatcher()

    asyncio.run(run_frames())

    assert [(event.event_type, event.payload) for event in client.events] == [
        ("TranscriptFinal", {"content": "Good morning.", "role": "assistant"})
    ]


def test_speech_and_interruption_frames_update_session_lifecycle(monkeypatch) -> None:
    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    processor = RuntimeEventProcessor(session)
    processor.forwarded = []

    async def process_base_frame(self, frame, direction) -> None:
        pass

    monkeypatch.setattr(FrameProcessor, "process_frame", process_base_frame)

    async def run_frames() -> None:
        await process(processor, UserStartedSpeakingFrame())
        assert session.state == SessionState.LISTENING
        await process(processor, BotStartedSpeakingFrame())
        assert session.state == SessionState.SPEAKING
        await process(processor, InterruptionFrame())
        assert session.state == SessionState.LISTENING
        assert session.cancel_requested is True
        await process(processor, UserStoppedSpeakingFrame())
        await process(processor, BotStoppedSpeakingFrame())
        await session.close_event_dispatcher()

    asyncio.run(run_frames())

    assert [event.event_type for event in client.events] == [
        "SpeechStarted",
        "AgentSpeechStarted",
        "UserInterrupted",
        "AgentSpeechCancelled",
        "SpeechStopped",
        "TurnCompleted",
    ]


def test_interruption_without_active_assistant_speech_does_not_emit_cancellation(
    monkeypatch,
) -> None:
    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    processor = RuntimeEventProcessor(session)
    processor.forwarded = []

    async def process_base_frame(self, frame, direction) -> None:
        pass

    monkeypatch.setattr(FrameProcessor, "process_frame", process_base_frame)

    async def run_frames() -> None:
        await process(processor, UserStartedSpeakingFrame())
        await process(processor, InterruptionFrame())
        await session.close_event_dispatcher()

    asyncio.run(run_frames())

    assert [event.event_type for event in client.events] == [
        "SpeechStarted",
        "UserInterrupted",
    ]
