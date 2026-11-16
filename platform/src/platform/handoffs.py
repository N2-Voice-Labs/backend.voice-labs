"""Session-token authorised requests for a translated human call.

PLATFORM_STAFF_KEY enables staff routes; PLATFORM_HANDOFF_STAFF_LANGUAGE defaults
to ko. Staff join through the existing public invite routes, which require
PLATFORM_PUBLIC_CALLS. Status reconciliation happens lazily on staff reads.
"""
import hmac
import os
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import auth, calls
from .context import AuthContext, get_ctx
from .models import Agent, Conversation, Handoff, Message, VoiceSession, utcnow
from .sessions import end_session
from .tenancy import get_owned, get_scoped_db, scoped

router = APIRouter(tags=["handoffs"])


class HandoffIn(BaseModel):
    session_id: UUID
    language: str = Field(default="en", min_length=1, max_length=20)
    staff_language: str = Field(default_factory=lambda: os.getenv("PLATFORM_HANDOFF_STAFF_LANGUAGE", "ko"), min_length=1, max_length=20)
    name: str | None = Field(default=None, max_length=100)
    reason: str | None = Field(default=None, max_length=4000)


def staff_ctx(request: Request) -> AuthContext:
    expected = os.getenv("PLATFORM_STAFF_KEY", "")
    if not expected:
        raise HTTPException(404, "staff_disabled")
    if not hmac.compare_digest(request.headers.get("x-staff-key", "").encode(), expected.encode()):
        raise HTTPException(401, "invalid_staff_key")
    return get_ctx(request)


def refresh_status(db: Session, ctx: AuthContext, item: Handoff) -> None:
    if item.status not in {"waiting", "accepted"}:
        return
    call = get_owned(db, ctx, VoiceSession, item.call_session_id)
    invite = (call.runtime_contract or {}).get("call_invite") or {}
    if call.status not in {"created", "running"}:
        item.status = "ended"
    elif not call.guest_token_hashes and utcnow().timestamp() >= float(invite.get("expires_at", 0)):
        item.status = "expired"
    else:
        return
    item.ended_at = utcnow()
    db.commit()


@router.post("/handoffs", status_code=201)
async def create(payload: HandoffIn, request: Request, _limit: None = Depends(calls.limit_create),
                 ctx: AuthContext = Depends(get_ctx), db: Session = Depends(get_scoped_db)) -> dict[str, Any]:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.startswith(auth.SESSION_TOKEN_PREFIX):
        raise HTTPException(401, "authentication_required")
    source = db.scalars(scoped(db, ctx, VoiceSession).where(VoiceSession.id == str(payload.session_id)).with_for_update()).first()
    if source is None or not source.session_token_hash or not hmac.compare_digest(auth.hash_token(token), source.session_token_hash):
        raise HTTPException(403, "forbidden")
    if db.scalars(scoped(db, ctx, Handoff).where(Handoff.from_session_id == source.id)).first():
        raise HTTPException(409, "handoff_exists")
    conversation = get_owned(db, ctx, Conversation, source.conversation_id)
    agent = get_owned(db, ctx, Agent, conversation.agent_id)
    if source.status not in {"created", "running"} or agent.mode != "agent" or agent.system:
        raise HTTPException(409, "session_not_active_agent")
    # Validate call inputs before ending the AI session.
    call_input = calls.CallIn(languages=[payload.language, payload.staff_language], name=payload.name)
    if payload.language.strip().lower() == payload.staff_language.strip().lower():
        raise HTTPException(422, "languages_must_differ")
    from .translations import TranslationIn, interpreter_config
    from .sessions import ParticipantIn
    interpreter_config(TranslationIn(languages=call_input.languages, creator=ParticipantIn(language=payload.language)))
    await end_session(db, ctx, payload.session_id)
    result = await calls.create_call(call_input, request, ctx, None, db)
    invite = await calls.create_invite(UUID(result["session_id"]), ctx, db)
    item = Handoff(organization_id=ctx.org_id, from_session_id=source.id,
                   from_conversation_id=source.conversation_id, call_session_id=result["session_id"],
                   invite_code=invite["code"], status="waiting", requester_name=payload.name,
                   requester_language=payload.language.strip().lower(), staff_language=payload.staff_language.strip().lower(), reason=payload.reason)
    db.add(item)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        await end_session(db, ctx, result["session_id"])
        raise HTTPException(409, "handoff_exists") from None
    return {**result, "handoff_id": item.id}


@router.get("/staff/handoffs")
async def list_handoffs(ctx: AuthContext = Depends(staff_ctx), db: Session = Depends(get_scoped_db)) -> list[dict[str, Any]]:
    items = db.scalars(scoped(db, ctx, Handoff).where(Handoff.status.in_(["waiting", "accepted"])).order_by(Handoff.created_at.desc())).all()
    result = []
    for item in items:
        refresh_status(db, ctx, item)
        if item.status in {"waiting", "accepted"}:
            result.append({field: getattr(item, field) for field in (
                "id", "status", "requester_name", "requester_language", "staff_language", "reason",
                "created_at", "call_session_id", "from_conversation_id")})
    return result


@router.post("/staff/handoffs/{handoff_id}/accept")
async def accept(handoff_id: UUID, ctx: AuthContext = Depends(staff_ctx), db: Session = Depends(get_scoped_db)) -> dict[str, str]:
    item = db.scalars(scoped(db, ctx, Handoff).where(Handoff.id == str(handoff_id)).with_for_update()).first()
    if item is None:
        raise HTTPException(404, "handoff_not_found")
    refresh_status(db, ctx, item)
    # Atomic claim also protects SQLite, where SELECT FOR UPDATE is ignored.
    claimed = db.execute(update(Handoff).where(Handoff.id == item.id, Handoff.status == "waiting").values(status="accepted", accepted_at=utcnow()))
    if not claimed.rowcount:
        db.rollback()
        raise HTTPException(409, "handoff_not_waiting")
    db.commit()
    return {"invite_code": item.invite_code, "call_session_id": item.call_session_id}


@router.get("/staff/handoffs/{handoff_id}/transcript")
async def transcript(handoff_id: UUID, ctx: AuthContext = Depends(staff_ctx), db: Session = Depends(get_scoped_db)) -> list[dict[str, Any]]:
    item = get_owned(db, ctx, Handoff, str(handoff_id))
    refresh_status(db, ctx, item)
    messages = db.scalars(scoped(db, ctx, Message).where(Message.conversation_id == item.from_conversation_id).order_by(Message.created_at, Message.id)).all()
    return [{field: getattr(message, field) for field in ("id", "role", "content", "primary_language", "detected_languages", "created_at")} for message in messages]
