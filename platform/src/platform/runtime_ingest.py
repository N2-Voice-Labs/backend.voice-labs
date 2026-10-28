"""Runtime event ingestion (POST /runtime/events) and its payload validators."""

import hmac
import os
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .conversations import apply_outcome
from .models import Agent, Conversation, Message, RuntimeEvent, VoiceSession, utcnow
from .runtime_events import ACCEPTED_EVENT_TYPES
from .tenancy import bypass, get_scoped_db
from .validation import OUTCOMES

router = APIRouter()


def validate_action_event_payload(event_type: str, payload: dict[str, Any]) -> None:
    def optional_str(key: str, detail: str, max_length: int | None = None) -> None:
        value = payload.get(key)
        if value is not None and (
            not isinstance(value, str) or (max_length and len(value) > max_length)
        ):
            raise HTTPException(status_code=422, detail=detail)

    if event_type == "ToolCalled":
        name = payload.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=422, detail="invalid_tool_event")
        arguments = payload.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            raise HTTPException(status_code=422, detail="invalid_tool_event")
        optional_str("status", "invalid_tool_event")
        optional_str("error", "invalid_tool_event")
    elif event_type == "HandoffRequested":
        optional_str("reason", "invalid_handoff_event")
        optional_str("target", "invalid_handoff_event")
    elif event_type == "OutcomeRecorded":
        if payload.get("outcome") not in OUTCOMES:
            raise HTTPException(status_code=422, detail="invalid_outcome")
        optional_str("summary", "invalid_outcome_event")
        optional_str("intent", "invalid_outcome_event", 200)
        actions = payload.get("actions")
        if actions is not None and (
            not isinstance(actions, list)
            or not all(isinstance(item, dict) for item in actions)
        ):
            raise HTTPException(status_code=422, detail="invalid_outcome_event")


@router.post("/runtime/events", status_code=status.HTTP_202_ACCEPTED)
async def ingest_runtime_event(
    event: dict[str, Any],
    db: Session = Depends(get_scoped_db),
    x_runtime_token: str | None = Header(default=None),
) -> dict[str, str]:
    expected = os.getenv("PLATFORM_RUNTIME_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")
    if not x_runtime_token or not hmac.compare_digest(x_runtime_token, expected):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")
    if event.get("event_type") not in ACCEPTED_EVENT_TYPES:
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
    validate_action_event_payload(event["event_type"], payload)
    if db.query(RuntimeEvent).filter(RuntimeEvent.event_id == event_id).first():
        return {"status": "duplicate"}
    session_id = event_identity["session_id"]
    conversation_id = event_identity["conversation_id"]
    agent_id = event_identity["agent_id"]
    # The Runtime carries no tenant: the organisation is whatever the VoiceSession row
    # says, and the event's conversation and agent must belong to it. Nothing in the
    # request body is trusted for this.
    with bypass(db):
        session = db.get(VoiceSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="voice_session_not_found")
        org_id = session.organization_id
        if conversation_id != session.conversation_id:
            raise HTTPException(status_code=409, detail="session_conversation_mismatch")
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="conversation_not_found")
        if conversation.organization_id != org_id:
            raise HTTPException(status_code=409, detail="session_conversation_mismatch")
        if agent_id != conversation.agent_id:
            raise HTTPException(status_code=409, detail="session_agent_mismatch")
        agent_org = db.scalar(select(Agent.organization_id).where(Agent.id == agent_id))
        if agent_org is not None and agent_org != org_id:
            raise HTTPException(status_code=409, detail="session_agent_mismatch")
    record = RuntimeEvent(
        organization_id=org_id,
        event_id=event_id,
        event_type=event["event_type"],
        schema_version=schema_version,
        session_id=session_id,
        conversation_id=conversation_id,
        agent_id=agent_id,
        payload=payload,
    )
    db.add(record)
    session.last_event_at = utcnow()
    if (
        event.get("event_type") == "TranscriptFinal"
        and conversation_id
        and payload.get("content")
    ):
        db.add(
            Message(
                organization_id=org_id,
                conversation_id=conversation_id,
                role=payload.get("role", "user"),
                content=payload["content"],
                primary_language=payload.get("primary_language"),
                detected_languages=payload.get("detected_languages", []),
            )
        )
    event_type = event.get("event_type")
    if event_type == "HandoffRequested":
        conversation.outcome = "escalated"
        conversation.outcome_recorded_at = utcnow()
        conversation.actions = [
            *(conversation.actions or []),
            {
                "type": "handoff_requested",
                "reason": payload.get("reason"),
                "target": payload.get("target"),
            },
        ]
    elif event_type == "OutcomeRecorded":
        apply_outcome(
            conversation,
            payload["outcome"],
            payload.get("summary"),
            payload.get("intent"),
            payload.get("actions"),
        )
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
            # Runtime lifecycle.EndReason values that are not failures.
            "platform_end",
            "idle_timeout",
            "max_duration",
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
