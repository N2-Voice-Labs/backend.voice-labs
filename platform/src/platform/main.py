"""Voice Labs control-plane API."""

import hmac
import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import Agent as AgentRecord
from .models import (
    Conversation,
    Message,
    ProviderConfig,
    RuntimeEvent,
    VoiceSession,
    utcnow,
)
from .livekit import create_participant_token


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("PLATFORM_AUTO_CREATE_SCHEMA") == "1":
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Voice Labs Platform", version="0.1.0", lifespan=lifespan)
RUNTIME_PROVIDER_SETTINGS = {
    "stt": {"model", "language", "interim_results", "smart_format"},
    "llm": {"model", "temperature", "max_tokens", "top_p"},
    "tts": {"model", "language"},
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("PLATFORM_WEB_ORIGIN", "http://localhost:3000")],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["content-type", "x-runtime-token"],
)


class AgentIn(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    system_prompt: str = ""
    enabled_languages: list[str] = ["en", "ru"]
    stt_provider_id: UUID | None = None
    llm_provider_id: UUID | None = None
    tts_provider_id: UUID | None = None
    voice_id: str | None = None
    status: str = "enabled"


class Agent(AgentIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: Any
    updated_at: Any


class AgentPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    system_prompt: str | None = None
    enabled_languages: list[str] | None = None
    stt_provider_id: UUID | None = None
    llm_provider_id: UUID | None = None
    tts_provider_id: UUID | None = None
    voice_id: str | None = None
    status: str | None = None

    @field_validator("name", "enabled_languages", "status", mode="before")
    @classmethod
    def reject_null_for_required_fields(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("field cannot be null")
        return value


class ProviderIn(BaseModel):
    kind: str = Field(pattern="^(stt|llm|tts)$")
    provider: str = Field(min_length=1)
    configuration: dict[str, Any] = Field(default_factory=dict)
    secret_reference: str | None = None
    enabled: bool = True

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        secret_names = ("secret", "password", "token", "api_key", "apikey")

        def contains_secret_key(candidate: Any) -> bool:
            if isinstance(candidate, dict):
                return any(
                    any(part in key.casefold() for part in secret_names)
                    or contains_secret_key(nested)
                    for key, nested in candidate.items()
                )
            if isinstance(candidate, list):
                return any(contains_secret_key(item) for item in candidate)
            return False

        if contains_secret_key(value):
            raise ValueError(
                "provider secrets must use secret_reference, not configuration"
            )
        return value


class Provider(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    kind: str
    provider: str
    configuration: dict[str, Any]
    secret_reference: str | None
    enabled: bool


class SessionIn(BaseModel):
    agent_id: UUID


def agent_response(agent: AgentRecord) -> Agent:
    return Agent.model_validate(agent)


def provider_ids_for_agent(values: dict[str, Any], db: Session) -> dict[str, Any]:
    for field, kind in (
        ("stt_provider_id", "stt"),
        ("llm_provider_id", "llm"),
        ("tts_provider_id", "tts"),
    ):
        if field not in values:
            continue
        provider_id = values[field]
        if provider_id is None:
            continue
        provider = db.get(ProviderConfig, str(provider_id))
        if provider is None or provider.kind != kind or not provider.enabled:
            raise HTTPException(status_code=422, detail=f"invalid_{field}")
        values[field] = str(provider_id)
    return values


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agents", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentIn, db: Session = Depends(get_db)) -> Agent:
    record = AgentRecord(**provider_ids_for_agent(payload.model_dump(), db))
    db.add(record)
    db.commit()
    db.refresh(record)
    return agent_response(record)


@app.get("/agents", response_model=list[Agent])
async def list_agents(db: Session = Depends(get_db)) -> list[Agent]:
    return [agent_response(agent) for agent in db.query(AgentRecord).all()]


@app.get("/agents/{agent_id}", response_model=Agent)
async def get_agent(agent_id: UUID, db: Session = Depends(get_db)) -> Agent:
    record = db.get(AgentRecord, str(agent_id))
    if record is None:
        raise HTTPException(status_code=404, detail="agent_not_found")
    return agent_response(record)


@app.patch("/agents/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: UUID, payload: AgentPatch, db: Session = Depends(get_db)
) -> Agent:
    record = db.get(AgentRecord, str(agent_id))
    if record is None:
        raise HTTPException(status_code=404, detail="agent_not_found")
    changes = provider_ids_for_agent(payload.model_dump(exclude_unset=True), db)
    for key, value in changes.items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return agent_response(record)


@app.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: UUID, db: Session = Depends(get_db)) -> None:
    record = db.get(AgentRecord, str(agent_id))
    if record is None:
        raise HTTPException(status_code=404, detail="agent_not_found")
    record.status = "disabled"
    db.commit()


@app.post("/providers", response_model=Provider, status_code=status.HTTP_201_CREATED)
async def create_provider(
    payload: ProviderIn, db: Session = Depends(get_db)
) -> Provider:
    record = ProviderConfig(**payload.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return Provider.model_validate(record)


@app.get("/providers", response_model=list[Provider])
async def list_providers(db: Session = Depends(get_db)) -> list[Provider]:
    return [Provider.model_validate(item) for item in db.query(ProviderConfig).all()]


@app.post("/voice/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionIn, db: Session = Depends(get_db)
) -> dict[str, Any]:
    agent = db.get(AgentRecord, str(payload.agent_id))
    if agent is None or agent.status != "enabled":
        raise HTTPException(status_code=409, detail="agent_not_enabled")
    runtime_url = os.getenv("RUNTIME_API_URL")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if runtime_url and not runtime_token:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")
    providers = {}
    provider_settings = {}
    for kind in ("stt", "llm", "tts"):
        provider_id = getattr(agent, f"{kind}_provider_id")
        if provider_id:
            provider = db.get(ProviderConfig, provider_id)
            if provider is None or not provider.enabled:
                raise HTTPException(
                    status_code=409, detail=f"selected_{kind}_provider_unavailable"
                )
            providers[kind] = provider.provider
            settings = {
                key: value
                for key, value in (provider.configuration or {}).items()
                if key in RUNTIME_PROVIDER_SETTINGS[kind]
            }
            if settings:
                provider_settings[kind] = settings

    conversation = Conversation(agent_id=agent.id)
    db.add(conversation)
    db.flush()
    session_id = str(uuid4())
    session = VoiceSession(
        id=session_id,
        conversation_id=conversation.id,
        room_name=f"voice-labs-{session_id}",
    )
    configured = all(
        os.getenv(name)
        for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    )
    token = ""
    if configured:
        token = create_participant_token(session.room_name, f"browser-{session.id}")
    runtime_status = "not_configured"
    request_payload = {
        "session_id": session.id,
        "conversation_id": conversation.id,
        "agent_id": agent.id,
        "room_name": session.room_name,
        "system_prompt": agent.system_prompt,
        "voice_id": agent.voice_id,
    }
    if providers:
        request_payload["providers"] = providers
    if provider_settings:
        request_payload["provider_settings"] = provider_settings
    session.runtime_contract = request_payload
    db.add(session)
    db.commit()
    if runtime_url:
        try:
            response: httpx.Response | None = None
            async with httpx.AsyncClient(timeout=5.0) as client:
                for attempt in range(2):
                    try:
                        response = await client.post(
                            f"{runtime_url.rstrip('/')}/sessions",
                            headers={"X-Runtime-Token": runtime_token},
                            json=request_payload,
                        )
                        response.raise_for_status()
                        break
                    except httpx.TransportError:
                        if attempt == 1:
                            raise
                    except httpx.HTTPStatusError as exc:
                        retryable = (
                            exc.response.status_code in {408, 425, 429}
                            or exc.response.status_code >= 500
                        )
                        if attempt == 1 or not retryable:
                            raise
                if response is None:
                    raise ValueError("runtime handoff did not return a response")
                runtime_response = response.json()
                if not isinstance(runtime_response, dict):
                    raise ValueError("runtime response status must be accepted")
                runtime_status = runtime_response.get("status", "")
                if runtime_status not in {
                    "accepted",
                    "already_running",
                    "already_finished",
                }:
                    raise ValueError(
                        "runtime response status must acknowledge the session"
                    )
                if runtime_status == "already_finished":
                    final_status = runtime_response.get("session_status")
                    if final_status not in {"ended", "failed", "cancelled"}:
                        raise ValueError(
                            "finished runtime response must include a terminal session status"
                        )
                    session.status = (
                        "ended" if final_status == "cancelled" else final_status
                    )
                    session.ended_at = utcnow()
                    conversation.status = session.status
                    conversation.ended_at = session.ended_at
                    db.commit()
        except (httpx.HTTPError, ValueError) as exc:
            session.status = "failed"
            session.ended_at = utcnow()
            conversation.status = "failed"
            conversation.ended_at = session.ended_at
            db.commit()
            raise HTTPException(status_code=503, detail="runtime_unavailable") from exc
    elif configured:
        runtime_status = "manual_start_required"
    return {
        "session_id": session.id,
        "conversation_id": conversation.id,
        "connection": {"url": os.getenv("LIVEKIT_URL", ""), "token": token},
        "room_name": session.room_name,
        "session_status": session.status,
        "connection_status": "configured" if configured else "not_configured",
        "runtime_status": runtime_status,
    }


@app.get("/conversations")
async def list_conversations(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [conversation_dict(item) for item in db.query(Conversation).all()]


@app.get("/voice/sessions/{session_id}")
async def get_voice_session(
    session_id: UUID, db: Session = Depends(get_db)
) -> dict[str, Any]:
    session = db.get(VoiceSession, str(session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="voice_session_not_found")
    return {
        "id": session.id,
        "conversation_id": session.conversation_id,
        "status": session.status,
        "room_name": session.room_name,
        "transport": session.transport,
        "started_at": session.started_at,
        "ended_at": session.ended_at,
    }


@app.post("/voice/sessions/{session_id}/ensure-running")
async def ensure_voice_session_running(
    session_id: UUID, db: Session = Depends(get_db)
) -> dict[str, str]:
    session = db.get(VoiceSession, str(session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="voice_session_not_found")
    if session.status not in {"created", "running"}:
        return {"session_status": session.status, "runtime_status": session.status}

    runtime_url = os.getenv("RUNTIME_API_URL")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if not runtime_url:
        return {"session_status": session.status, "runtime_status": "not_configured"}
    if not runtime_token:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")

    runtime_url = runtime_url.rstrip("/")
    headers = {"X-Runtime-Token": runtime_token}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{runtime_url}/sessions/{session.id}", headers=headers
            )
            response.raise_for_status()
            status_response = response.json()
            if not isinstance(status_response, dict):
                raise ValueError("runtime status response must be an object")
            runtime_status = status_response.get("status")
            if not isinstance(runtime_status, str):
                raise ValueError("runtime status response must include a status")
            db.refresh(session)
            if session.status not in {"created", "running"}:
                return {
                    "session_status": session.status,
                    "runtime_status": session.status,
                }

            if runtime_status == "not_running":
                if session.runtime_contract is None:
                    runtime_status = "failed"
                else:
                    response = await client.post(
                        f"{runtime_url}/sessions",
                        headers=headers,
                        json=session.runtime_contract,
                    )
                    response.raise_for_status()
                    retry_response = response.json()
                    if not isinstance(retry_response, dict):
                        raise ValueError("runtime start response must be an object")
                    retry_status = retry_response.get("status")
                    if not isinstance(retry_status, str):
                        raise ValueError("runtime start response must include a status")
                    if retry_status in {"accepted", "already_running"}:
                        runtime_status = retry_status
                    elif retry_status == "already_finished":
                        final_runtime_status = retry_response.get("session_status")
                        if not isinstance(
                            final_runtime_status, str
                        ) or final_runtime_status not in {
                            "ended",
                            "failed",
                            "cancelled",
                        }:
                            raise ValueError(
                                "finished runtime response must include a terminal status"
                            )
                        runtime_status = final_runtime_status
                    else:
                        raise ValueError(
                            "runtime did not acknowledge the session retry"
                        )
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="runtime_unavailable") from exc

    if runtime_status in {"failed", "ended", "cancelled"}:
        db.refresh(session)
        if session.status in {"created", "running"}:
            session.status = (
                "ended" if runtime_status == "cancelled" else runtime_status
            )
            session.ended_at = utcnow()
            conversation = db.get(Conversation, session.conversation_id)
            if conversation and conversation.status not in {"ended", "failed"}:
                conversation.status = session.status
                conversation.ended_at = session.ended_at
            db.commit()
    return {
        "session_status": session.status,
        "runtime_status": str(runtime_status or "unknown"),
    }


@app.post("/voice/sessions/{session_id}/end")
async def end_voice_session(
    session_id: UUID, db: Session = Depends(get_db)
) -> dict[str, str]:
    session = db.get(VoiceSession, str(session_id))
    if session is None:
        raise HTTPException(status_code=404, detail="voice_session_not_found")
    if session.status in {"ended", "failed"}:
        return {"status": session.status}
    runtime_url = os.getenv("RUNTIME_API_URL")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if runtime_url and not runtime_token:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")
    if runtime_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{runtime_url.rstrip('/')}/sessions/{session.id}/end",
                    headers={"X-Runtime-Token": runtime_token},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="runtime_unavailable") from exc
    session.status = "ended"
    session.ended_at = utcnow()
    conversation = db.get(Conversation, session.conversation_id)
    if conversation:
        conversation.status = "ended"
        conversation.ended_at = session.ended_at
    db.commit()
    return {"status": "ended"}


def conversation_dict(conversation: Conversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "agent_id": conversation.agent_id,
        "status": conversation.status,
        "started_at": conversation.started_at,
        "ended_at": conversation.ended_at,
        "metadata": conversation.metadata_json,
    }


@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: UUID, db: Session = Depends(get_db)
) -> dict[str, Any]:
    record = db.get(Conversation, str(conversation_id))
    if record is None:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return conversation_dict(record)


@app.get("/conversations/{conversation_id}/runtime-state")
async def get_conversation_runtime_state(
    conversation_id: UUID, db: Session = Depends(get_db)
) -> dict[str, Any]:
    conversation = db.get(Conversation, str(conversation_id))
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    session = (
        db.query(VoiceSession)
        .filter(VoiceSession.conversation_id == str(conversation_id))
        .order_by(VoiceSession.started_at.desc())
        .first()
    )
    events = (
        db.query(RuntimeEvent)
        .filter(
            RuntimeEvent.conversation_id == str(conversation_id),
            RuntimeEvent.event_type.in_(
                {
                    "SpeechStarted",
                    "SpeechStopped",
                    "TurnCompleted",
                    "AgentSpeechStarted",
                    "UserInterrupted",
                    "AgentSpeechCancelled",
                    "RuntimeStarted",
                    "SessionEnded",
                    "SessionFailed",
                    "RuntimeFailed",
                }
            ),
        )
        .order_by(RuntimeEvent.received_at.desc())
        .limit(100)
        .all()
    )
    return {
        "session_id": session.id if session else None,
        "session_status": session.status if session else None,
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "schema_version": event.schema_version,
                "payload": event.payload,
                "received_at": event.received_at.isoformat(),
            }
            for event in reversed(events)
        ],
    }


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: UUID, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    if db.get(Conversation, str(conversation_id)) is None:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return [
        {
            "id": item.id,
            "role": item.role,
            "content": item.content,
            "primary_language": item.primary_language,
            "detected_languages": item.detected_languages,
            "created_at": item.created_at,
        }
        for item in db.query(Message)
        .filter(Message.conversation_id == str(conversation_id))
        .order_by(Message.created_at, Message.id)
        .all()
    ]


@app.post("/runtime/events", status_code=status.HTTP_202_ACCEPTED)
async def ingest_runtime_event(
    event: dict[str, Any],
    db: Session = Depends(get_db),
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, str]:
    expected = os.getenv("PLATFORM_RUNTIME_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")
    if not x_runtime_token or not hmac.compare_digest(x_runtime_token, expected):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    if event.get("event_type") not in {
        "SpeechStarted",
        "SpeechStopped",
        "TurnCompleted",
        "AgentSpeechStarted",
        "UserInterrupted",
        "AgentSpeechCancelled",
        "TranscriptFinal",
        "LatencyMeasured",
        "SessionEnded",
        "SessionFailed",
        "RuntimeFailed",
        "RuntimeStarted",
    }:
        raise HTTPException(status_code=422, detail="unsupported_runtime_event_type")
    schema_version = event.get("schema_version", 1)
    if type(schema_version) is not int or schema_version != 1:
        raise HTTPException(
            status_code=422, detail="unsupported_runtime_event_schema_version"
        )
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id or len(event_id) > 255:
        raise HTTPException(status_code=422, detail="valid_event_id_required")
    event_identity: dict[str, str] = {}
    for field in ("session_id", "conversation_id", "agent_id"):
        value = event.get(field)
        if not isinstance(value, str):
            raise HTTPException(status_code=422, detail=f"{field}_required")
        try:
            event_identity[field] = str(UUID(value))
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"{field}_must_be_uuid"
            ) from exc
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="event_payload_must_be_object")
    if event["event_type"] == "TranscriptFinal" and payload.get("content"):
        role = payload.get("role", "user")
        if not isinstance(role, str) or role not in {"user", "assistant"}:
            raise HTTPException(status_code=422, detail="invalid_transcript_role")
        if not isinstance(payload["content"], str):
            raise HTTPException(status_code=422, detail="invalid_transcript_content")
        primary_language = payload.get("primary_language")
        if primary_language is not None and not isinstance(primary_language, str):
            raise HTTPException(status_code=422, detail="invalid_primary_language")
        detected_languages = payload.get("detected_languages", [])
        if not isinstance(detected_languages, list) or not all(
            isinstance(language, str) for language in detected_languages
        ):
            raise HTTPException(status_code=422, detail="invalid_detected_languages")
    if db.query(RuntimeEvent).filter(RuntimeEvent.event_id == event_id).first():
        return {"status": "duplicate"}
    session_id = event_identity["session_id"]
    conversation_id = event_identity["conversation_id"]
    agent_id = event_identity["agent_id"]
    session = db.get(VoiceSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="voice_session_not_found")
    if conversation_id != session.conversation_id:
        raise HTTPException(status_code=409, detail="session_conversation_mismatch")
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    if agent_id != conversation.agent_id:
        raise HTTPException(status_code=409, detail="session_agent_mismatch")
    record = RuntimeEvent(
        event_id=event_id,
        event_type=event["event_type"],
        schema_version=schema_version,
        session_id=session_id,
        conversation_id=conversation_id,
        agent_id=agent_id,
        payload=payload,
    )
    db.add(record)
    if (
        event.get("event_type") == "TranscriptFinal"
        and conversation_id
        and payload.get("content")
    ):
        db.add(
            Message(
                conversation_id=conversation_id,
                role=payload.get("role", "user"),
                content=payload["content"],
                primary_language=payload.get("primary_language"),
                detected_languages=payload.get("detected_languages", []),
            )
        )
    event_type = event.get("event_type")
    if (
        session
        and event_type == "RuntimeStarted"
        and session.status not in {"ended", "failed"}
    ):
        session.status = "running"
        session.started_at = utcnow()
        if conversation:
            conversation.status = "active"
            conversation.started_at = session.started_at
    elif session and event_type in {"SessionEnded", "SessionFailed", "RuntimeFailed"}:
        reason = str(payload.get("reason", "")).strip().casefold()
        normal_reasons = {
            "cancelled",
            "canceled",
            "client_requested",
            "completed",
            "normal",
            "session ended by request",
            "livekit disconnected",
            "browser participant disconnected",
            "participant_disconnected",
            "room_disconnected",
            "user_hangup",
            "user_requested",
        }
        failed = event_type in {"SessionFailed", "RuntimeFailed"} or bool(
            payload.get("error")
            or payload.get("failed")
            or (reason and reason not in normal_reasons)
        )
        if session.status != "failed" and (session.status != "ended" or failed):
            session.status = "failed" if failed else "ended"
            session.ended_at = utcnow()
            if conversation:
                conversation.status = session.status
                conversation.ended_at = session.ended_at
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        if db.query(RuntimeEvent).filter(RuntimeEvent.event_id == event_id).first():
            return {"status": "duplicate"}
        raise
    return {"status": "accepted"}
