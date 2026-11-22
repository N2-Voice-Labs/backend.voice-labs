import asyncio
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.runtime import api


@pytest.mark.parametrize("token", [None, "wrong-runtime-token"])
@pytest.mark.parametrize("endpoint", ["start", "status", "end"])
def test_runtime_control_routes_reject_missing_or_wrong_tokens(
    monkeypatch: pytest.MonkeyPatch, token: str | None, endpoint: str
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "expected-runtime-token")
    session_id = uuid4()
    payload = {
        "session_id": str(session_id),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "voice-test-room",
    }
    headers = {"X-Runtime-Token": token} if token else {}

    with TestClient(api.app) as client:
        if endpoint == "start":
            response = client.post("/sessions", json=payload, headers=headers)
        elif endpoint == "status":
            response = client.get(f"/sessions/{session_id}", headers=headers)
        else:
            response = client.post(f"/sessions/{session_id}/end", headers=headers)

        assert response.status_code == 401
        assert response.json() == {"detail": "invalid_runtime_token"}


def test_runtime_retains_only_recent_terminal_session_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    monkeypatch.setattr(api, "_MAX_TERMINAL_SESSION_HISTORY", 1, raising=False)
    session_ids = [uuid4(), uuid4()]
    headers = {"X-Runtime-Token": "test-runtime-token"}

    with TestClient(api.app) as client:
        for session_id in session_ids:
            ended = client.post(f"/sessions/{session_id}/end", headers=headers)
            assert ended.status_code == 200
            assert ended.json() == {"status": "ended"}
        oldest = client.get(f"/sessions/{session_ids[0]}", headers=headers)
        newest = client.get(f"/sessions/{session_ids[1]}", headers=headers)

    assert oldest.status_code == 200
    assert oldest.json()["status"] == "not_running"
    assert newest.status_code == 200
    assert newest.json()["status"] == "ended"


def test_session_cleanup_removes_runtime_state_when_event_client_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    for name in (
        "LIVEKIT_URL",
        "RUNTIME_LIVEKIT_TOKEN",
        "LIVEKIT_ROOM_NAME",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "CARTESIA_API_KEY",
        "CARTESIA_VOICE_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    class EventClient:
        def __init__(self) -> None:
            self.events = []

        async def emit(self, runtime_event) -> None:
            self.events.append(runtime_event)

        async def aclose(self) -> None:
            raise RuntimeError("event client close failed")

    event_client = EventClient()
    monkeypatch.setattr(
        api,
        "PlatformEventClient",
        lambda *_args, **_kwargs: event_client,
    )
    request = api.SessionStart(
        session_id=uuid4(),
        conversation_id=uuid4(),
        agent_id=uuid4(),
        room_name="voice-test-room",
    )
    session_key = str(request.session_id)

    with pytest.raises(RuntimeError, match="event client close failed"):
        asyncio.run(api.run_session(request))

    assert [runtime_event.event_type for runtime_event in event_client.events] == [
        "SessionFailed"
    ]
    assert api._session_states[session_key] == "failed"
    assert session_key not in api._active_sessions
    assert session_key not in api._runtime_sessions


@pytest.mark.parametrize("terminal_status", ["ended", "failed", "cancelled"])
def test_terminal_session_id_cannot_be_restarted(
    monkeypatch: pytest.MonkeyPatch, terminal_status: str
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    session_id = uuid4()
    api._session_states[str(session_id)] = terminal_status
    payload = {
        "session_id": str(session_id),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "voice-test-room",
    }

    with TestClient(api.app) as client:
        response = client.post(
            "/sessions",
            json=payload,
            headers={"X-Runtime-Token": "test-runtime-token"},
        )
        session_status = client.get(
            f"/sessions/{session_id}",
            headers={"X-Runtime-Token": "test-runtime-token"},
        )

    assert response.status_code == 202
    assert response.json() == {
        "status": "already_finished",
        "session_status": terminal_status,
    }
    assert session_status.status_code == 200
    assert session_status.json()["status"] == terminal_status


def test_active_session_duplicate_remains_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    started_sessions: list[api.SessionStart] = []

    async def hold_session(request: api.SessionStart) -> None:
        started_sessions.append(request)
        await asyncio.Event().wait()

    monkeypatch.setattr(api, "run_session", hold_session)
    payload = {
        "session_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "voice-test-room",
        "providers": {"stt": "deepgram"},
        "provider_settings": {"stt": {"model": "nova-3", "language": "multi"}},
    }
    session_key = payload["session_id"]
    headers = {"X-Runtime-Token": "test-runtime-token"}

    with TestClient(api.app) as client:
        first = client.post("/sessions", json=payload, headers=headers)
        duplicate = client.post("/sessions", json=payload, headers=headers)

    assert first.status_code == 202
    assert first.json() == {"status": "accepted"}
    assert duplicate.status_code == 202
    assert duplicate.json() == {"status": "already_running"}
    assert started_sessions[0].providers == {"stt": "deepgram"}
    assert started_sessions[0].provider_settings == {
        "stt": {"model": "nova-3", "language": "multi"}
    }
    api._active_sessions.pop(session_key, None)
    api._session_states.pop(session_key, None)
    api._session_contracts.pop(session_key, None)


def test_active_session_id_rejects_a_different_room_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")

    async def hold_session(request: api.SessionStart) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(api, "run_session", hold_session)
    payload = {
        "session_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "voice-test-room",
    }
    session_key = payload["session_id"]
    headers = {"X-Runtime-Token": "test-runtime-token"}

    with TestClient(api.app) as client:
        accepted = client.post("/sessions", json=payload, headers=headers)
        conflicting = client.post(
            "/sessions", json={**payload, "room_name": "wrong-room"}, headers=headers
        )

    assert accepted.status_code == 202
    assert accepted.json() == {"status": "accepted"}
    assert conflicting.status_code == 409
    assert conflicting.json() == {"detail": "session_contract_conflict"}
    api._active_sessions.pop(session_key, None)
    api._session_states.pop(session_key, None)
    api._session_contracts.pop(session_key, None)


def test_end_session_preserves_an_existing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    session_id = uuid4()
    api._session_states[str(session_id)] = "failed"

    client = TestClient(api.app)
    response = client.post(
        f"/sessions/{session_id}/end",
        headers={"X-Runtime-Token": "test-runtime-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "failed"}
    assert api._session_states[str(session_id)] == "failed"


def test_session_start_accepts_tools_and_knowledge_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    started: list[api.SessionStart] = []

    async def hold_session(request: api.SessionStart) -> None:
        started.append(request)
        await asyncio.Event().wait()

    monkeypatch.setattr(api, "run_session", hold_session)
    tool = {
        "name": "lookup_order",
        "description": "Look up",
        "tool_type": "http",
        "input_schema": {"type": "object"},
        "configuration": {"url": "https://orders.test"},
    }
    payload = {
        "session_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "voice-test-room",
        "tools": [tool],
        "knowledge_enabled": True,
    }
    legacy = {**payload, "session_id": str(uuid4())}
    del legacy["tools"], legacy["knowledge_enabled"]
    headers = {"X-Runtime-Token": "test-runtime-token"}

    with TestClient(api.app) as client:
        assert client.post("/sessions", json=payload, headers=headers).status_code == 202
        assert client.post("/sessions", json=legacy, headers=headers).status_code == 202
    assert [(r.tools, r.knowledge_enabled) for r in started] == [([tool], True), ([], False)]
    for key in (payload["session_id"], legacy["session_id"]):
        api._active_sessions.pop(key, None)
        api._session_states.pop(key, None)
        api._session_contracts.pop(key, None)


@pytest.mark.parametrize("mode_given", [False, True])
def test_start_warns_when_the_mode_is_left_to_the_agent_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode_given: bool
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    payload = {
        "session_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "voice-test-room",
        # Rejected right after the check, so no session is ever started.
        "private_listening": True,
    }
    if mode_given:
        payload["mode"] = "interpreter"
    with caplog.at_level("WARNING", logger="src.runtime.api"):
        with TestClient(api.app) as client:
            response = client.post(
                "/sessions", json=payload, headers={"X-Runtime-Token": "test-runtime-token"}
            )
    assert response.status_code == 422
    warned = any("without an explicit mode" in record.message for record in caplog.records)
    assert warned is (not mode_given)


def test_failed_session_is_logged_with_its_id_and_no_secrets(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "platform-token-SECRET")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-SECRET")
    for name in ("LIVEKIT_URL", "RUNTIME_LIVEKIT_TOKEN", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)

    class EventClient:
        async def emit(self, runtime_event) -> None:
            return None

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(api, "PlatformEventClient", lambda *_a, **_k: EventClient())
    request = api.SessionStart(
        session_id=uuid4(), conversation_id=uuid4(), agent_id=uuid4(), room_name="voice-test-room"
    )

    with caplog.at_level("ERROR", logger=api.logger.name):
        with pytest.raises(RuntimeError):
            asyncio.run(api.run_session(request))

    records = [r for r in caplog.records if str(request.session_id) in r.getMessage()]
    assert records and records[0].exc_info is not None
    assert "SECRET" not in caplog.text.replace("OPENAI_API_KEY", "")


def test_consume_session_result_logs_the_task_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def boom() -> None:
        raise ValueError("pipeline exploded")

    async def scenario() -> None:
        task = asyncio.create_task(boom(), name="voice-session-abc")
        task.add_done_callback(api._consume_session_result)
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

    with caplog.at_level("ERROR", logger=api.logger.name):
        asyncio.run(scenario())

    record = next(r for r in caplog.records if "voice-session-abc" in r.getMessage())
    assert record.exc_info and record.exc_info[0] is ValueError


def test_end_session_returns_without_waiting_forever_for_a_stuck_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    monkeypatch.setattr(api, "END_WAIT_TIMEOUT_SECONDS", 0.05)
    session_id = uuid4()
    key = str(session_id)

    async def scenario() -> None:
        release = asyncio.Event()

        async def stubborn() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()  # slow to stop

        task = asyncio.create_task(stubborn())
        await asyncio.sleep(0)
        api._active_sessions[key] = task
        try:
            result = await asyncio.wait_for(
                api.end_session(session_id, x_runtime_token="test-runtime-token"), 2
            )
            assert result == {"status": "ending"}
            assert not task.done()  # the wait timeout did not cancel it again
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            api._active_sessions.pop(key, None)
            api._session_states.pop(key, None)

    asyncio.run(scenario())


def test_shutdown_is_bounded_when_a_session_will_not_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "SHUTDOWN_WAIT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(api.sip_transfer, "install_if_configured", lambda: None)

    async def idle() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(api.provider_health, "refresh_forever", idle)
    key = str(uuid4())

    async def scenario() -> None:
        release = asyncio.Event()

        async def stubborn() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()  # swallows the first cancellation and never stops alone

        task = asyncio.create_task(stubborn())
        await asyncio.sleep(0)
        api._active_sessions[key] = task
        try:
            async with api.lifespan(api.app):
                pass
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            api._active_sessions.pop(key, None)

    asyncio.run(asyncio.wait_for(scenario(), 3))
