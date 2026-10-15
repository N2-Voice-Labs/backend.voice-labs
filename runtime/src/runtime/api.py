import asyncio
import hmac
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, Field

from .events_client import PlatformEventClient
from .pipeline import PipelineConfig
from .session import RuntimeSession, SessionState
from .tts_preview import TTSPreviewError, synthesize_preview

_active_sessions: dict[str, asyncio.Task[None]] = {}
_session_states: dict[str, str] = {}
_runtime_sessions: dict[str, RuntimeSession] = {}
_session_contracts: dict[str, dict[str, Any]] = {}
_terminal_session_order: OrderedDict[str, None] = OrderedDict()
_TERMINAL_SESSION_STATES = {"ended", "failed", "cancelled"}
_MAX_TERMINAL_SESSION_HISTORY = 1024


def _remember_terminal_session(session_key: str) -> None:
    """Bound terminal status retention and discard completed start contracts."""

    if _session_states.get(session_key) not in _TERMINAL_SESSION_STATES:
        return
    _session_contracts.pop(session_key, None)
    _terminal_session_order[session_key] = None
    _terminal_session_order.move_to_end(session_key)
    while len(_terminal_session_order) > _MAX_TERMINAL_SESSION_HISTORY:
        expired_key, _ = _terminal_session_order.popitem(last=False)
        _session_states.pop(expired_key, None)
        _session_contracts.pop(expired_key, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    tasks = list(_active_sessions.items())
    for key, task in tasks:
        runtime_session = _runtime_sessions.get(key)
        if runtime_session is None or not await runtime_session.request_end():
            task.cancel()
    if tasks:
        await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)


app = FastAPI(title="Voice Labs Runtime", version="0.1.0", lifespan=lifespan)


class SessionStart(BaseModel):
    session_id: UUID
    conversation_id: UUID
    agent_id: UUID
    room_name: str = Field(min_length=1, max_length=255)
    system_prompt: str = ""
    voice_id: str | None = None
    providers: dict[str, str] = Field(default_factory=dict)
    provider_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_enabled: bool = False


async def run_session(request: SessionStart) -> None:
    session_key = str(request.session_id)
    _session_states[session_key] = "starting"
    event_client: PlatformEventClient | None = None
    runtime_session: RuntimeSession | None = None
    pipeline_managed_lifecycle = False
    try:
        runtime_token = os.getenv("RUNTIME_PLATFORM_TOKEN")
        if not runtime_token:
            raise RuntimeError("RUNTIME_PLATFORM_TOKEN is required")
        event_client = PlatformEventClient(
            os.getenv("RUNTIME_PLATFORM_URL", "http://localhost:8000"),
            runtime_token,
        )
        runtime_session = RuntimeSession(
            request.session_id,
            request.conversation_id,
            request.agent_id,
            event_client,
        )
        _runtime_sessions[session_key] = runtime_session
        config = PipelineConfig.from_environment(
            {
                "room_name": request.room_name,
                "system_prompt": request.system_prompt,
                "voice_id": request.voice_id or "",
                "providers": request.providers,
                "provider_settings": request.provider_settings,
                "tools": request.tools,
                "knowledge_enabled": request.knowledge_enabled,
            }
        )
        _session_states[session_key] = "connecting"
        pipeline_managed_lifecycle = True
        await runtime_session.run_pipeline(config)
    except asyncio.CancelledError:
        event_delivery_failed = bool(
            runtime_session and runtime_session.event_delivery_error
        )
        _session_states[session_key] = (
            "failed" if event_delivery_failed else "cancelled"
        )
        if runtime_session and event_client and not pipeline_managed_lifecycle:
            await event_client.emit(
                runtime_session.event(
                    "SessionFailed" if event_delivery_failed else "SessionEnded",
                    {
                        "outcome": "failed" if event_delivery_failed else "cancelled",
                        "reason": "event_delivery_failed"
                        if event_delivery_failed
                        else "cancelled",
                    },
                )
            )
        raise
    except Exception as exc:
        _session_states[session_key] = "failed"
        if runtime_session and event_client and not pipeline_managed_lifecycle:
            await event_client.emit(
                runtime_session.event(
                    "SessionFailed",
                    {"outcome": "failed", "reason": type(exc).__name__},
                )
            )
        raise
    finally:
        try:
            if event_client:
                if (
                    not runtime_session
                    or not runtime_session.defer_event_client_close()
                ):
                    await event_client.aclose()
        finally:
            if _session_states.get(session_key) == "connecting":
                _session_states[session_key] = "ended"
            _active_sessions.pop(session_key, None)
            _runtime_sessions.pop(session_key, None)
            _remember_terminal_session(session_key)


@app.post("/sessions", status_code=status.HTTP_202_ACCEPTED)
async def start_session(
    payload: SessionStart,
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, str]:
    expected = os.getenv("RUNTIME_PLATFORM_TOKEN")
    if (
        not expected
        or not x_runtime_token
        or not hmac.compare_digest(x_runtime_token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    key = str(payload.session_id)
    session_status = _session_states.get(key)
    if session_status in _TERMINAL_SESSION_STATES:
        return {"status": "already_finished", "session_status": session_status}
    contract = payload.model_dump(mode="json")
    existing_contract = _session_contracts.get(key)
    if existing_contract is not None and existing_contract != contract:
        raise HTTPException(status_code=409, detail="session_contract_conflict")
    task = _active_sessions.get(key)
    if task and not task.done():
        return {"status": "already_running"}
    _session_contracts[key] = contract
    _session_states[key] = "starting"
    task = asyncio.create_task(run_session(payload), name=f"voice-session-{key}")
    _active_sessions[key] = task
    task.add_done_callback(_consume_session_result)
    return {"status": "accepted"}


def _consume_session_result(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()


@app.get("/sessions/{session_id}")
async def get_session_status(
    session_id: UUID,
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, Any]:
    expected = os.getenv("RUNTIME_PLATFORM_TOKEN")
    if (
        not expected
        or not x_runtime_token
        or not hmac.compare_digest(x_runtime_token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    key = str(session_id)
    runtime_session = _runtime_sessions.get(key)
    if runtime_session and runtime_session.state in {
        SessionState.CONNECTED,
        SessionState.LISTENING,
        SessionState.SPEAKING,
    }:
        session_status = "running"
    else:
        session_status = _session_states.get(key, "not_running")
    return {"session_id": session_id, "status": session_status}


@app.post("/sessions/{session_id}/end")
async def end_session(
    session_id: UUID,
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, str]:
    expected = os.getenv("RUNTIME_PLATFORM_TOKEN")
    if (
        not expected
        or not x_runtime_token
        or not hmac.compare_digest(x_runtime_token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    key = str(session_id)
    task = _active_sessions.get(key)
    if task and not task.done():
        runtime_session = _runtime_sessions.get(key)
        if runtime_session is None or not await runtime_session.request_end():
            task.cancel()
        results = await asyncio.gather(task, return_exceptions=True)
        failed = _session_states.get(key) == "failed" or any(
            isinstance(result, Exception) for result in results
        )
        _session_states[key] = "failed" if failed else "ended"
    elif _session_states.get(key) != "failed":
        _session_states[key] = "ended"
    _remember_terminal_session(key)
    return {"status": _session_states.get(key, "ended")}


class TTSPreviewRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=100)
    provider_voice_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=300)
    language: str | None = Field(default=None, max_length=50)
    model: str | None = Field(default=None, max_length=100)


@app.post("/tts/preview")
async def tts_preview(
    payload: TTSPreviewRequest,
    x_runtime_token: str | None = Header(default=None),
) -> Response:
    expected = os.getenv("RUNTIME_PLATFORM_TOKEN")
    if (
        not expected
        or not x_runtime_token
        or not hmac.compare_digest(x_runtime_token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    if payload.provider != "cartesia":
        raise HTTPException(status_code=422, detail="unsupported_tts_provider")
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="tts_not_configured")
    try:
        audio = await synthesize_preview(
            api_key=api_key,
            provider_voice_id=payload.provider_voice_id,
            text=payload.text,
            language=payload.language,
            model=payload.model,
        )
    except TTSPreviewError as exc:
        raise HTTPException(status_code=502, detail="tts_provider_error") from exc
    return Response(content=audio, media_type="audio/wav")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
