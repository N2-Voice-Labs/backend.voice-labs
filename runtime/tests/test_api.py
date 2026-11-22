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


# -- POST /voices/clone-upload ----------------------------------------------------------------

UPLOAD_TOKEN = {"X-Runtime-Token": "upload-token"}
UPLOAD_FIELDS = {"provider": "elevenlabs", "name": "Anna", "language": "en"}


@pytest.fixture
def upload_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "upload-token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    monkeypatch.setenv("CARTESIA_API_KEY", "ct-key")
    calls: list[dict] = []

    async def fake_clone(**kwargs):
        calls.append(kwargs)
        if kwargs.get("fail") is not None:
            raise kwargs["fail"]
        return "provider-voice-1"

    monkeypatch.setattr(api, "clone_voice", fake_clone)
    return calls


def upload(
    fields=None,
    *,
    content=b"RIFFaudio",
    content_type="audio/webm;codecs=opus",
    filename="rec.webm",
    headers=UPLOAD_TOKEN,
    with_file=True,
):
    files = {"file": (filename, content, content_type)} if with_file else None
    with TestClient(api.app) as client:
        return client.post(
            "/voices/clone-upload",
            data={**UPLOAD_FIELDS, **(fields or {})},
            files=files,
            headers=headers,
        )


def test_clone_upload_success_forwards_bytes_without_storing(upload_env) -> None:
    response = upload({"description": "calm"})
    assert response.status_code == 200
    assert response.json() == {"provider_voice_id": "provider-voice-1"}
    (call,) = upload_env
    assert call == {
        "provider": "elevenlabs",
        "api_key": "el-key",
        "name": "Anna",
        "language": "en",
        "sample": b"RIFFaudio",
        "content_type": "audio/webm",
        "filename": "rec.webm",
        "description": "calm",
    }


def test_clone_upload_cartesia_uses_its_own_key_and_octet_stream_is_accepted(upload_env) -> None:
    response = upload({"provider": "cartesia"}, content_type="application/octet-stream")
    assert response.status_code == 200
    assert upload_env[0]["provider"] == "cartesia"
    assert upload_env[0]["api_key"] == "ct-key"
    assert upload_env[0]["description"] is None


@pytest.mark.parametrize("headers", [{}, {"X-Runtime-Token": "wrong"}])
def test_clone_upload_requires_runtime_token(upload_env, headers) -> None:
    response = upload(headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_runtime_token"}
    assert upload_env == []


def test_clone_upload_fails_closed_without_configured_token(upload_env, monkeypatch) -> None:
    monkeypatch.delenv("RUNTIME_PLATFORM_TOKEN")
    assert upload().status_code == 401
    assert upload_env == []


def test_clone_upload_unknown_provider(upload_env) -> None:
    response = upload({"provider": "other"})
    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported_tts_provider"}
    assert upload_env == []


@pytest.mark.parametrize("provider", ["elevenlabs", "cartesia"])
def test_clone_upload_missing_key(upload_env, monkeypatch, provider) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY" if provider == "elevenlabs" else "CARTESIA_API_KEY")
    response = upload({"provider": provider})
    assert response.status_code == 503
    assert response.json() == {"detail": "tts_not_configured"}
    assert upload_env == []


def test_clone_upload_empty_file(upload_env) -> None:
    response = upload(content=b"")
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_sample"}
    assert upload_env == []


def test_clone_upload_missing_file_part(upload_env) -> None:
    response = upload(with_file=False)
    assert response.status_code == 422
    assert upload_env == []


@pytest.mark.parametrize("content_type", ["text/plain", "application/json", "image/png", "video/webm"])
def test_clone_upload_wrong_content_type(upload_env, content_type) -> None:
    response = upload(content_type=content_type)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_sample"}
    assert upload_env == []


def test_clone_upload_exactly_max_size_is_accepted(upload_env) -> None:
    from src.runtime.voice_cloning import MAX_SAMPLE_BYTES

    assert upload(content=b"a" * MAX_SAMPLE_BYTES).status_code == 200


def test_clone_upload_oversized_is_rejected(upload_env) -> None:
    from src.runtime.voice_cloning import MAX_SAMPLE_BYTES

    response = upload(content=b"a" * (MAX_SAMPLE_BYTES + 1))
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_sample"}
    assert upload_env == []


def test_clone_upload_stream_is_cut_off_without_content_length(upload_env) -> None:
    """The cap applies while streaming, so a body with no Content-Length stops early."""
    from fastapi import HTTPException

    from src.runtime.voice_cloning import MAX_SAMPLE_BYTES

    consumed = 0

    class FakeRequest:
        headers: dict[str, str] = {}

        async def stream(self):
            nonlocal consumed
            for _ in range(1000):  # ~64 MiB if fully consumed
                consumed += 1
                yield b"a" * 65536

    async def drain():
        async for _ in api._bounded_upload_stream(FakeRequest()):
            pass

    with pytest.raises(HTTPException) as caught:
        asyncio.run(drain())
    assert caught.value.status_code == 422 and caught.value.detail == "invalid_sample"
    assert consumed * 65536 <= MAX_SAMPLE_BYTES + api._UPLOAD_OVERHEAD_BYTES + 65536


def test_clone_upload_oversized_content_length_is_rejected_before_reading(upload_env) -> None:
    """A declared Content-Length over the cap is refused without parsing the body."""
    from src.runtime.voice_cloning import MAX_SAMPLE_BYTES

    big = b"a" * (MAX_SAMPLE_BYTES + 1024 * 1024)
    response = upload(content=big)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_sample"}
    assert upload_env == []


def test_clone_upload_rejects_non_multipart_and_bad_fields(upload_env) -> None:
    with TestClient(api.app) as client:
        plain = client.post("/voices/clone-upload", json={}, headers=UPLOAD_TOKEN)
    assert plain.status_code == 422
    assert upload({"name": ""}).status_code == 422
    assert upload({"name": "x" * 201}).status_code == 422
    assert upload({"language": ""}).status_code == 422
    assert upload_env == []


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (api.ClonePlanRequired("plan"), "clone_plan_upgrade_required"),
        (api.CloneProviderError("boom"), "clone_provider_error"),
    ],
)
def test_clone_upload_provider_errors(upload_env, monkeypatch, error, detail) -> None:
    async def failing(**kwargs):
        raise error

    monkeypatch.setattr(api, "clone_voice", failing)
    response = upload()
    assert response.status_code == 502
    assert response.json() == {"detail": detail}


def test_clone_upload_end_to_end_against_mocked_elevenlabs(monkeypatch) -> None:
    import httpx

    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "upload-token")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    real = httpx.AsyncClient
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"voice_id": "el-1"})

    monkeypatch.setattr(
        "src.runtime.voice_cloning.httpx.AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    response = upload(content=b"recorded-bytes")
    assert response.status_code == 200
    assert response.json() == {"provider_voice_id": "el-1"}
    assert seen[0].headers["xi-api-key"] == "el-key"
    assert b"recorded-bytes" in seen[0].read()
