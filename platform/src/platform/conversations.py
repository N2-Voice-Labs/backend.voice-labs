"""Conversation routes: list, detail, outcome, runtime state, and messages."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .context import AuthContext, get_ctx, require
from .models import Conversation, Message, RuntimeEvent, VoiceSession, utcnow
from .runtime_events import RUNTIME_STATE_EVENT_TYPES
from .tenancy import get_owned, get_scoped_db
from .validation import OUTCOMES

router = APIRouter()


def _org_filter(model: type, ctx: AuthContext) -> list[Any]:
    """Explicit organisation criterion (the session listener is only a backstop)."""

    return [model.organization_id == ctx.org_id] if ctx.org_id else []  # type: ignore[attr-defined]


def _owned_conversation(db: Session, ctx: AuthContext, conversation_id: UUID) -> Conversation:
    return get_owned(
        db, ctx, Conversation, str(conversation_id), not_found_detail="conversation_not_found"
    )


class OutcomeIn(BaseModel):
    outcome: str
    summary: str | None = None
    intent: str | None = Field(default=None, max_length=200)
    actions: list[dict[str, Any]] | None = None


@router.get("/conversations")
async def list_conversations(
    agent_id: UUID | None = None,
    status: str | None = None,
    outcome: str | None = None,
    channel: str | None = None,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> list[dict[str, Any]]:
    query = db.query(Conversation).filter(*_org_filter(Conversation, ctx))
    if agent_id is not None:
        query = query.filter(Conversation.agent_id == str(agent_id))
    if status is not None:
        query = query.filter(Conversation.status == status)
    if outcome is not None:
        query = query.filter(Conversation.outcome == outcome)
    if channel is not None:
        query = query.filter(Conversation.channel == channel)
    return [
        conversation_dict(item)
        for item in query.order_by(Conversation.started_at.desc()).all()
    ]


def conversation_dict(conversation: Conversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "agent_id": conversation.agent_id,
        "status": conversation.status,
        "started_at": conversation.started_at,
        "ended_at": conversation.ended_at,
        "metadata": conversation.metadata_json,
        "deployment_id": conversation.deployment_id,
        "channel": conversation.channel,
        "outcome": conversation.outcome,
        "summary": conversation.summary,
        "intent": conversation.intent,
        "actions": conversation.actions or [],
        "outcome_recorded_at": conversation.outcome_recorded_at,
    }


def apply_outcome(
    conversation: Conversation,
    outcome: str,
    summary: str | None = None,
    intent: str | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> None:
    conversation.outcome = outcome
    if summary is not None:
        conversation.summary = summary
    if intent is not None:
        conversation.intent = intent
    if actions is not None:
        conversation.actions = list(actions)
    conversation.outcome_recorded_at = utcnow()


@router.put("/conversations/{conversation_id}/outcome")
async def put_conversation_outcome(
    conversation_id: UUID,
    payload: OutcomeIn,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    if payload.outcome not in OUTCOMES:
        raise HTTPException(status_code=422, detail="invalid_outcome")
    record = _owned_conversation(db, ctx, conversation_id)
    apply_outcome(
        record, payload.outcome, payload.summary, payload.intent, payload.actions
    )
    db.commit()
    db.refresh(record)
    return conversation_dict(record)


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    ctx: AuthContext = Depends(get_ctx),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    record = _owned_conversation(db, ctx, conversation_id)
    return conversation_dict(record)


@router.get("/conversations/{conversation_id}/runtime-state")
async def get_conversation_runtime_state(
    conversation_id: UUID,
    ctx: AuthContext = Depends(get_ctx),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    _owned_conversation(db, ctx, conversation_id)
    session = (
        db.query(VoiceSession)
        .filter(VoiceSession.conversation_id == str(conversation_id))
        .filter(*_org_filter(VoiceSession, ctx))
        .order_by(VoiceSession.started_at.desc())
        .first()
    )
    events = (
        db.query(RuntimeEvent)
        .filter(
            RuntimeEvent.conversation_id == str(conversation_id),
            RuntimeEvent.event_type.in_(RUNTIME_STATE_EVENT_TYPES),
            *_org_filter(RuntimeEvent, ctx),
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


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: UUID,
    ctx: AuthContext = Depends(get_ctx),
    db: Session = Depends(get_scoped_db),
) -> list[dict[str, Any]]:
    _owned_conversation(db, ctx, conversation_id)
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
        .filter(
            Message.conversation_id == str(conversation_id),
            *_org_filter(Message, ctx),
        )
        .order_by(Message.created_at, Message.id)
        .all()
    ]
