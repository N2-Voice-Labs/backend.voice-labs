"""Event delivery policy: critical events end the call when lost, best-effort ones never do."""

import asyncio
import logging
from uuid import uuid4

import httpx
import pytest

from src.runtime import session as session_module
from src.runtime.events import CRITICAL_EVENT_TYPES, EVENT_TYPES, is_critical_event
from src.runtime.session import _EVENT_QUEUE_CAPACITY, RuntimeSession


class Worker:
    def __init__(self) -> None:
        self.cancel_reasons: list[str | None] = []

    async def cancel(self, *, reason=None) -> None:
        self.cancel_reasons.append(reason)


class Client:
    """Records delivered events; ``fail(event)`` may return an exception to raise."""

    def __init__(self, fail=None) -> None:
        self.events = []
        self.fail = fail
        self.release = asyncio.Event()
        self.block = False

    async def emit(self, event) -> None:
        if self.block:
            await self.release.wait()
        error = self.fail(event) if self.fail else None
        if error is not None:
            raise error
        self.events.append(event)


def make_session(client) -> tuple[RuntimeSession, Worker]:
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    worker = Worker()
    session._pipeline_worker = worker
    return session, worker


def status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://platform/runtime/events")
    return httpx.HTTPStatusError("x", request=request, response=httpx.Response(code, request=request))


def test_event_classification() -> None:
    critical = {
        "RuntimeStarted", "TranscriptFinal", "ToolCalled", "HandoffRequested",
        "OutcomeRecorded", "SessionEnded", "SessionFailed", "RuntimeFailed",
    }
    assert CRITICAL_EVENT_TYPES == critical
    for best_effort in ("LatencyMeasured", "TurnDebug", "SpeechStarted", "SpeechStopped",
                        "TurnCompleted", "AgentSpeechStarted", "UserInterrupted",
                        "AgentSpeechCancelled", "ProviderError", "EngineFallback"):
        assert best_effort in EVENT_TYPES and not is_critical_event(best_effort)


def test_best_effort_overflow_is_dropped_counted_and_never_ends_the_call(caplog) -> None:
    client = Client()
    client.block = True
    session, worker = make_session(client)

    async def flood() -> None:
        for _ in range(_EVENT_QUEUE_CAPACITY * 3):
            session.enqueue_event(session.event("LatencyMeasured"))  # must not raise
        client.release.set()
        await session.close_event_dispatcher()

    with caplog.at_level(logging.WARNING, logger=session_module.__name__):
        asyncio.run(flood())

    assert worker.cancel_reasons == []
    assert session.event_delivery_error is None
    dropped = session.dropped_events["LatencyMeasured"]
    assert dropped >= _EVENT_QUEUE_CAPACITY
    drop_warnings = [r for r in caplog.records if "best-effort" in r.getMessage()]
    assert 1 <= len(drop_warnings) < dropped  # rate limited


def test_critical_event_evicts_queued_best_effort_instead_of_failing() -> None:
    client = Client()
    client.block = True
    session, worker = make_session(client)

    async def scenario() -> None:
        session.enqueue_event(session.event("SpeechStarted"))  # in flight, blocked
        await asyncio.sleep(0)
        for _ in range(_EVENT_QUEUE_CAPACITY):
            session.enqueue_event(session.event("LatencyMeasured"))
        session.enqueue_event(session.event("TranscriptFinal", {"content": "hi"}))
        client.release.set()
        await session.close_event_dispatcher()

    asyncio.run(scenario())

    assert worker.cancel_reasons == []
    assert session.dropped_events["LatencyMeasured"] == 1
    assert "TranscriptFinal" in [e.event_type for e in client.events]


def test_queue_full_of_critical_events_still_ends_the_call() -> None:
    client = Client()
    client.block = True
    session, worker = make_session(client)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="capacity"):
            for _ in range(_EVENT_QUEUE_CAPACITY * 2):
                session.enqueue_event(session.event("TranscriptFinal"))
        client.release.set()
        await session.close_event_dispatcher()

    asyncio.run(scenario())

    assert worker.cancel_reasons == ["Runtime event queue capacity exceeded"]


def test_best_effort_delivery_failure_does_not_end_the_call_or_stop_delivery() -> None:
    client = Client(
        fail=lambda e: ConnectionError("down") if e.event_type == "LatencyMeasured" else None
    )
    session, worker = make_session(client)

    async def scenario() -> None:
        session.enqueue_event(session.event("LatencyMeasured"))
        session.enqueue_event(session.event("TranscriptFinal", {"content": "hi"}))
        await session.close_event_dispatcher()

    asyncio.run(scenario())

    assert worker.cancel_reasons == [] and session.event_delivery_error is None
    assert [e.event_type for e in client.events] == ["TranscriptFinal"]
    assert session.dropped_events["LatencyMeasured"] == 1


@pytest.mark.parametrize("event_type", ["ToolCalled", "TranscriptFinal", "SpeechStarted"])
def test_a_4xx_rejection_is_logged_and_dropped_without_ending_the_call(
    event_type, caplog
) -> None:
    client = Client(fail=lambda e: status_error(422) if e.event_type == event_type else None)
    session, worker = make_session(client)

    async def scenario() -> None:
        session.enqueue_event(session.event(event_type))
        session.enqueue_event(session.event("OutcomeRecorded", {"outcome": "resolved"}))
        await session.close_event_dispatcher()

    with caplog.at_level(logging.WARNING, logger=session_module.__name__):
        asyncio.run(scenario())

    assert worker.cancel_reasons == [] and session.event_delivery_error is None
    assert [e.event_type for e in client.events] == ["OutcomeRecorded"]
    assert any(event_type in r.getMessage() for r in caplog.records)


def test_a_critical_event_lost_after_retries_still_ends_the_call() -> None:
    client = Client(fail=lambda e: status_error(503))
    session, worker = make_session(client)

    async def scenario() -> None:
        session.enqueue_event(session.event("TranscriptFinal"))
        await asyncio.sleep(0.05)
        await session.close_event_dispatcher()

    asyncio.run(scenario())

    assert worker.cancel_reasons == ["Platform event delivery failed"]
    assert session.event_delivery_error is not None
