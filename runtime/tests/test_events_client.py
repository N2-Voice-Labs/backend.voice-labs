import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from src.runtime import events_client
from src.runtime.events import RuntimeEvent
from src.runtime.events_client import PlatformEventClient


def test_event_client_posts_versioned_event_with_runtime_auth(monkeypatch) -> None:
    requests = []
    original_async_client = httpx.AsyncClient

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"status": "accepted"})

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        "src.runtime.events_client.httpx.AsyncClient",
        lambda **kwargs: original_async_client(transport=transport, **kwargs),
    )
    event_client = PlatformEventClient("http://platform.local", "runtime-secret")
    event = RuntimeEvent(
        event_type="TranscriptFinal",
        session_id=uuid4(),
        conversation_id=uuid4(),
        agent_id=uuid4(),
        payload={"content": "hello", "role": "user"},
    )

    async def send_event() -> None:
        assert await event_client.emit(event) == {"status": "accepted"}
        await event_client.aclose()

    asyncio.run(send_event())

    assert len(requests) == 1
    assert requests[0].headers["X-Runtime-Token"] == "runtime-secret"
    payload = json.loads(requests[0].content)
    assert payload["schema_version"] == 1
    assert payload["event_type"] == "TranscriptFinal"
    assert payload["payload"] == {"content": "hello", "role": "user"}


def test_event_client_retries_transient_platform_failure_with_same_event(monkeypatch) -> None:
    requests = []
    original_async_client = httpx.AsyncClient

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"detail": "temporary outage"})
        return httpx.Response(202, json={"status": "accepted"})

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        "src.runtime.events_client.httpx.AsyncClient",
        lambda **kwargs: original_async_client(transport=transport, **kwargs),
    )
    event_client = PlatformEventClient("http://platform.local", "runtime-secret")
    event = RuntimeEvent(
        event_type="TranscriptFinal",
        session_id=uuid4(),
        conversation_id=uuid4(),
        agent_id=uuid4(),
        payload={"content": "hello", "role": "user"},
    )

    async def send_event() -> None:
        assert await event_client.emit(event) == {"status": "accepted"}
        await event_client.aclose()

    asyncio.run(send_event())

    assert len(requests) == 2
    assert [request.headers["X-Runtime-Token"] for request in requests] == [
        "runtime-secret",
        "runtime-secret",
    ]
    assert requests[0].content == requests[1].content


def test_event_client_honors_bounded_retry_after_for_rate_limits(monkeypatch) -> None:
    requests = []
    delays = []
    original_async_client = httpx.AsyncClient

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "5"},
                json={"detail": "rate limited"},
            )
        return httpx.Response(202, json={"status": "accepted"})

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        "src.runtime.events_client.httpx.AsyncClient",
        lambda **kwargs: original_async_client(transport=transport, **kwargs),
    )
    monkeypatch.setattr("src.runtime.events_client.asyncio.sleep", record_delay)
    event_client = PlatformEventClient("http://platform.local", "runtime-secret")
    event = RuntimeEvent(
        event_type="RuntimeStarted",
        session_id=uuid4(),
        conversation_id=uuid4(),
        agent_id=uuid4(),
        payload={"room_name": "voice-room"},
    )

    async def send_event() -> None:
        assert await event_client.emit(event) == {"status": "accepted"}
        await event_client.aclose()

    asyncio.run(send_event())

    assert len(requests) == 2
    assert delays == [1.0]


def make_event(event_type: str) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=event_type,
        session_id=uuid4(),
        conversation_id=uuid4(),
        agent_id=uuid4(),
        payload={},
    )


def install_client(monkeypatch, handle, *, clock: list[float] | None = None):
    """A PlatformEventClient on a mock transport; sleeping advances a fake clock."""

    from types import SimpleNamespace

    original_async_client = httpx.AsyncClient
    now = clock if clock is not None else [0.0]
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        now[0] += delay

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        "src.runtime.events_client.httpx.AsyncClient",
        lambda **kwargs: original_async_client(transport=transport, **kwargs),
    )
    monkeypatch.setattr("src.runtime.events_client.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "src.runtime.events_client.time", SimpleNamespace(monotonic=lambda: now[0])
    )
    return PlatformEventClient("http://platform.local", "runtime-secret"), delays


def test_non_retryable_4xx_is_raised_immediately_without_retry(monkeypatch) -> None:
    requests = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(422, json={"detail": "invalid_tool_event"})

    client, delays = install_client(monkeypatch, handle)

    with pytest.raises(httpx.HTTPStatusError) as raised:
        asyncio.run(client.emit(make_event("ToolCalled")))

    assert raised.value.response.status_code == 422
    assert len(requests) == 1 and delays == []


def test_best_effort_event_gives_up_after_a_few_attempts(monkeypatch) -> None:
    requests = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503)

    client, _ = install_client(monkeypatch, handle)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.emit(make_event("LatencyMeasured")))

    assert len(requests) == events_client.BEST_EFFORT_ATTEMPTS


def test_critical_event_keeps_retrying_with_exponential_backoff(monkeypatch) -> None:
    requests = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) <= 7:
            return httpx.Response(503)
        return httpx.Response(202, json={"status": "accepted"})

    client, delays = install_client(monkeypatch, handle)

    assert asyncio.run(client.emit(make_event("TranscriptFinal"))) == {"status": "accepted"}
    assert len(requests) == 8
    assert delays == sorted(delays) and delays[0] == 0.1 and delays[-1] > delays[0] * 10


def test_critical_event_gives_up_after_the_retry_budget(monkeypatch) -> None:
    clock = [0.0]

    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("platform down", request=request)

    client, delays = install_client(monkeypatch, handle, clock=clock)

    with pytest.raises(httpx.ConnectError):
        asyncio.run(client.emit(make_event("OutcomeRecorded")))

    assert 50 <= sum(delays) <= events_client.CRITICAL_RETRY_BUDGET_SECONDS
    assert max(delays) <= events_client._MAX_BACKOFF_SECONDS
