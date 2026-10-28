import asyncio
import hmac
import os
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_serializer

from .events_client import PlatformEventClient
from .participants import ParticipantDirectory, ParticipantError
from .pipeline import PipelineConfig
from . import provider_health, sip_transfer
from .lifecycle import EndReason, max_sessions, terminal_payload
from .session import RuntimeSession, SessionState
from .tts_preview import PREVIEW_PROVIDERS, TTSPreviewError, get_preview_provider
from .voice_cloning import (
    CloneProviderError,
    SampleUnavailable,
    UnsupportedSampleURL,
    UnknownVoiceProvider,
    VoiceDeleteError,
    clone_voice,
    delete_voice,
    download_sample,
    PROVIDER_VOICE_ID_PATTERN,
    supports_voice_delete,
)

_active_sessions: dict[str, asyncio.Task[None]] = {}
_session_states: dict[str, str] = {}
_runtime_sessions: dict[str, RuntimeSession] = {}
_session_contracts: dict[str, dict[str, Any]] = {}
_participant_directories: dict[str, ParticipantDirectory] = {}
_terminal_session_order: OrderedDict[str, None] = OrderedDict()
_TERMINAL_SESSION_STATES = {"ended", "failed", "cancelled"}
_MAX_TERMINAL_SESSION_HISTORY = 1024


def _remember_terminal_session(session_key: str) -> None:
    """Bound terminal status retention and discard completed start contracts."""

    if _session_states.get(session_key) not in _TERMINAL_SESSION_STATES:
        return
    _session_contracts.pop(session_key, None)
    _participant_directories.pop(session_key, None)
    _terminal_session_order[session_key] = None
    _terminal_session_order.move_to_end(session_key)
    while len(_terminal_session_order) > _MAX_TERMINAL_SESSION_HISTORY:
        expired_key, _ = _terminal_session_order.popitem(last=False)
        _session_states.pop(expired_key, None)
        _session_contracts.pop(expired_key, None)
        _participant_directories.pop(expired_key, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    sip_transfer.install_if_configured()
    health_task = asyncio.create_task(
        provider_health.refresh_forever(), name="provider-health-refresh"
    )
    yield
    health_task.cancel()
    await asyncio.gather(health_task, return_exceptions=True)
    tasks = list(_active_sessions.items())
    for key, task in tasks:
        runtime_session = _runtime_sessions.get(key)
        if runtime_session is None or not await runtime_session.request_end():
            task.cancel()
    if tasks:
        await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)


app = FastAPI(title="Voice Labs Runtime", version="0.1.0", lifespan=lifespan)


class ParticipantSpec(BaseModel):
    """One interpreter participant: spoken language and the voice their words get."""

    language: str = Field(min_length=2, max_length=16)
    voice_id: str | None = Field(default=None, max_length=200)
    display_name: str | None = Field(default=None, max_length=100)
    # Dialled into its own phone room (only with ``private_listening``); see
    # ``participants.phone_room_name``. Absent for everyone else.
    phone_room: bool = False

    @model_serializer(mode="wrap")
    def _omit_absent_phone_room(self, handler):
        data = handler(self)
        if not data.get("phone_room"):
            data.pop("phone_room", None)
        return data


class ParticipantIn(ParticipantSpec):
    identity: str = Field(min_length=1, max_length=255)


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
    mode: str = "agent"
    interpreter: dict[str, Any] | None = None
    transfer_to: str | None = Field(default=None, max_length=255)
    # Two-person interpreter sessions: LiveKit identity -> participant. Absent for
    # single-participant sessions, which behave exactly as before.
    participants: dict[str, ParticipantSpec] | None = None
    # Two-person interpreter sessions whose participants cannot subscribe in the session
    # room: their translation is published in a private room per listener. Absent
    # (false) keeps the single-room routing exactly as before.
    private_listening: bool = False
    # Agent + translator (TR-040): {working_language, caller_languages}. Absent for
    # ordinary agents, which keep the contract exactly as before.
    translation: dict[str, Any] | None = None
    # ST-13: per-session length cap in seconds; absent -> RUNTIME_MAX_SESSION_SECS.
    max_duration_secs: int | None = Field(default=None, ge=1, le=86400)

    @model_serializer(mode="wrap")
    def _omit_absent_optional_blocks(self, handler):
        # Sessions without participants (TR-020) or translation (TR-040) keep the
        # contract exactly as before those features.
        data = handler(self)
        for key in ("participants", "translation", "max_duration_secs"):
            if data.get(key) is None:
                data.pop(key, None)
        if not data.get("private_listening"):
            data.pop("private_listening", None)
        return data


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
        runtime_session.participants = _participant_directories.get(session_key)
        runtime_session.max_duration_secs = request.max_duration_secs
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
                "mode": request.mode,
                "interpreter": request.interpreter,
                "translation": request.translation,
                "private_listening": request.private_listening,
                "participants": request.participants
                and {
                    identity: spec.model_dump()
                    for identity, spec in request.participants.items()
                },
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
                    terminal_payload(
                        "failed" if event_delivery_failed else "cancelled",
                        EndReason.RUNTIME_ERROR
                        if event_delivery_failed
                        else EndReason.CANCELLED,
                        detail="event_delivery_failed" if event_delivery_failed else None,
                    ),
                )
            )
        raise
    except Exception as exc:
        _session_states[session_key] = "failed"
        if runtime_session and event_client and not pipeline_managed_lifecycle:
            await event_client.emit(
                runtime_session.event(
                    "SessionFailed",
                    terminal_payload(
                        "failed", EndReason.RUNTIME_ERROR, detail=type(exc).__name__
                    ),
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


def active_session_count() -> int:
    """Sessions whose run task is still going (the admission-control count)."""

    return sum(1 for task in _active_sessions.values() if not task.done())


@app.post("/sessions", status_code=status.HTTP_202_ACCEPTED)
async def start_session(
    payload: SessionStart,
    x_runtime_token: str | None = Header(default=None),
) -> Any:
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
    if payload.private_listening and not payload.participants:
        raise HTTPException(status_code=422, detail="private_listening_requires_participants")
    directory: ParticipantDirectory | None = None
    if payload.participants:
        directory = _build_participant_directory(payload)
    contract = payload.model_dump(mode="json")
    existing_contract = _session_contracts.get(key)
    if existing_contract is not None and existing_contract != contract:
        raise HTTPException(status_code=409, detail="session_contract_conflict")
    task = _active_sessions.get(key)
    if task and not task.done():
        return {"status": "already_running"}
    # ST-13 admission control: refuse before anything is stored or started.
    cap = max_sessions()
    if active_session_count() >= cap:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "code": "runtime_at_capacity",
                "message": f"Runtime is at capacity ({cap} active sessions); retry later.",
            },
            headers={"Retry-After": "5"},
        )
    _session_contracts[key] = contract
    if directory is not None:
        _participant_directories[key] = directory
    _session_states[key] = "starting"
    task = asyncio.create_task(run_session(payload), name=f"voice-session-{key}")
    _active_sessions[key] = task
    task.add_done_callback(_consume_session_result)
    return {"status": "accepted"}


def _engine_supports_two_way(interpreter: dict[str, Any]) -> bool:
    """Whether the engine this interpreter config resolves to serves several speakers."""

    from .interpreter import InterpreterSettings
    from .translation_engine import get_engine

    try:
        return get_engine(InterpreterSettings.from_mapping(interpreter).engine).supports_two_way
    except RuntimeError:
        return False


def _build_participant_directory(payload: SessionStart) -> ParticipantDirectory:
    """Validate and index ``participants`` (422 when they do not fit the session)."""

    interpreter = payload.interpreter or {}
    if payload.mode != "interpreter" or not _engine_supports_two_way(interpreter):
        raise HTTPException(
            status_code=422, detail="participants_require_cascaded_interpreter"
        )
    if not payload.private_listening and any(
        spec.phone_room for spec in (payload.participants or {}).values()
    ):
        raise HTTPException(status_code=422, detail="phone_room_requires_private_listening")
    try:
        return ParticipantDirectory.from_mapping(
            interpreter.get("languages") or [],
            {
                identity: spec.model_dump()
                for identity, spec in (payload.participants or {}).items()
            },
        )
    except ParticipantError as exc:
        status_code = 409 if exc.code == "session_full" else 422
        raise HTTPException(status_code=status_code, detail=exc.code) from exc


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


@app.post("/sessions/{session_id}/participants")
async def add_session_participant(
    session_id: UUID,
    payload: ParticipantIn,
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Add or update a participant of a running two-person interpreter session."""

    expected = os.getenv("RUNTIME_PLATFORM_TOKEN")
    if (
        not expected
        or not x_runtime_token
        or not hmac.compare_digest(x_runtime_token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    key = str(session_id)
    if key not in _session_contracts or _session_states.get(key) in _TERMINAL_SESSION_STATES:
        raise HTTPException(status_code=404, detail="session_not_found")
    directory = _participant_directories.get(key)
    if directory is None:
        raise HTTPException(status_code=409, detail="participants_not_enabled")
    spec = payload.model_dump(exclude={"identity"})
    if payload.phone_room and not _session_contracts[key].get("private_listening"):
        raise HTTPException(status_code=422, detail="phone_room_requires_private_listening")
    try:
        await directory.update(payload.identity, spec)
    except ParticipantError as exc:
        status_code = 409 if exc.code == "session_full" else 422
        raise HTTPException(status_code=status_code, detail=exc.code) from exc
    # Keep the stored start contract equal to what the Platform would resend on a retry.
    _session_contracts[key]["participants"] = directory.as_contract()
    return {"status": "updated", "participants": len(directory)}


@app.delete("/sessions/{session_id}/participants/{identity}")
async def remove_session_participant(
    session_id: UUID,
    identity: str,
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Withdraw a phone participant whose call did not connect (frees its slot)."""

    expected = os.getenv("RUNTIME_PLATFORM_TOKEN")
    if (
        not expected
        or not x_runtime_token
        or not hmac.compare_digest(x_runtime_token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    key = str(session_id)
    directory = _participant_directories.get(key)
    if key not in _session_contracts or directory is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    try:
        await directory.remove(identity)
    except ParticipantError as exc:
        status_code = 404 if exc.code == "participant_not_found" else 409
        raise HTTPException(status_code=status_code, detail=exc.code) from exc
    _session_contracts[key]["participants"] = directory.as_contract()
    return {"status": "removed", "participants": len(directory)}


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
    if payload.provider not in PREVIEW_PROVIDERS:
        raise HTTPException(status_code=422, detail="unsupported_tts_provider")
    preview = get_preview_provider(payload.provider)
    api_key = os.getenv(preview.api_key_env)
    if not api_key:
        raise HTTPException(status_code=503, detail="tts_not_configured")
    try:
        audio = await preview.synthesize(
            api_key=api_key,
            provider_voice_id=payload.provider_voice_id,
            text=payload.text,
            language=payload.language,
            model=payload.model,
        )
    except TTSPreviewError as exc:
        raise HTTPException(status_code=502, detail="tts_provider_error") from exc
    return Response(content=audio, media_type="audio/wav")


class VoiceCloneRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    language: str = Field(pattern=r"^[a-z]{2}$")
    sample_url: str = Field(min_length=1, max_length=1024)
    description: str | None = Field(default=None, max_length=500)


@app.post("/voices/clone")
async def voices_clone(
    payload: VoiceCloneRequest,
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, str]:
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
        sample, content_type, filename = await download_sample(payload.sample_url)
        provider_voice_id = await clone_voice(
            api_key=api_key,
            name=payload.name,
            language=payload.language,
            sample=sample,
            content_type=content_type,
            filename=filename,
            description=payload.description,
        )
    except UnsupportedSampleURL as exc:
        raise HTTPException(status_code=422, detail="unsupported_sample_url") from exc
    except SampleUnavailable as exc:
        raise HTTPException(status_code=422, detail="sample_unavailable") from exc
    except CloneProviderError as exc:
        raise HTTPException(status_code=502, detail="clone_provider_error") from exc
    return {"provider_voice_id": provider_voice_id}


_PROVIDER_API_KEY_ENV = {"cartesia": "CARTESIA_API_KEY"}
_VOICE_ID_RE = re.compile(PROVIDER_VOICE_ID_PATTERN)


@app.delete("/voices/{provider}/{provider_voice_id}")
async def voices_delete(
    provider: str,
    provider_voice_id: str,
    x_runtime_token: str | None = Header(default=None),
) -> Response:
    expected = os.getenv("RUNTIME_PLATFORM_TOKEN")
    if (
        not expected
        or not x_runtime_token
        or not hmac.compare_digest(x_runtime_token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    if not supports_voice_delete(provider):
        return JSONResponse(status_code=404, content={"code": "unknown_provider"})
    if not _VOICE_ID_RE.fullmatch(provider_voice_id):
        raise HTTPException(status_code=422, detail="invalid_provider_voice_id")
    api_key = os.getenv(_PROVIDER_API_KEY_ENV.get(provider, ""))
    if not api_key:
        raise HTTPException(status_code=503, detail="tts_not_configured")
    try:
        await delete_voice(provider, provider_voice_id, api_key=api_key)
    except UnknownVoiceProvider:
        return JSONResponse(status_code=404, content={"code": "unknown_provider"})
    except VoiceDeleteError as exc:
        return JSONResponse(
            status_code=502,
            content={"code": "voice_provider_delete_failed", "retryable": exc.retryable},
        )
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "active_sessions": active_session_count(),
        "max_sessions": max_sessions(),
    }


@app.get("/health/providers")
async def health_providers(
    refresh: bool = False,
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, Any]:
    expected = os.getenv("RUNTIME_PLATFORM_TOKEN")
    if (
        not expected
        or not x_runtime_token
        or not hmac.compare_digest(x_runtime_token, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    return await provider_health.health_report(force=refresh)
