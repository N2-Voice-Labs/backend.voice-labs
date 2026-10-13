import asyncio
import json
from uuid import uuid4

import httpx

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
