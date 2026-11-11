"""Shareable-link calls: two people, no accounts (opt-in, ``PLATFORM_PUBLIC_CALLS=1``).

A thin front over the existing translation session: ``POST /calls`` opens the same hidden
interpreter session as ``POST /translations``, and joining seats the second person through the
same participant path as ``POST /voice/sessions/{id}/participants``. Nothing about the Runtime,
the contract or the audio routing changes.

What the browser gets, and what it never gets:

* The creator receives a LiveKit token (10 minutes) and a session token (``vls_``) for *this*
  call only; the guest receives the same for their seat. No Platform or provider key ever
  reaches a browser, and a session token cannot reach any other call or any admin route.
* The invitation is a random 128-bit secret. Only its SHA-256 digest is stored, it expires
  (``PLATFORM_CALL_INVITE_TTL_SECS``, default 30 minutes) and it seats at most one guest: the
  seat is created when the guest joins, so an unopened link holds nothing. Opening the link
  again with the same code returns the same seat with fresh credentials (a page reload or a
  reconnect), and nobody else can take the seat.
* The speech and translation providers come from the server (``PLATFORM_CALLS_PROVIDERS``), never
  from the browser.
* Creating calls and opening invitations are rate limited per client address, the number of
  simultaneous calls is capped per organisation (the existing cap), and a call nobody joins is
  closed by the stale-session reaper.

The three public routes are listed in ``auth.PUBLIC_CALL_ROUTES``. They run as the system
principal of ``PLATFORM_CALLS_ORG_ID`` (default organisation) and call only the functions below.
"""

from __future__ import annotations

import hashlib
import math
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import limits
from .context import AuthContext, get_ctx
from .models import DEFAULT_ORGANIZATION_ID, ProviderConfig, VoiceSession
from .sessions import GuestInviteIn, ParticipantIn, add_voice_session_participant, end_session
from .tenancy import get_scoped_db, scoped
from .translations import TranslationIn, create_translation
from .validation import DEFAULT_LATENCY_MODE

router = APIRouter(tags=["calls"])

DEFAULT_INVITE_TTL_SECS = 30 * 60
DEFAULT_CREATE_PER_IP = 6  # calls per client address per hour
DEFAULT_OPEN_PER_IP = 30  # invitation lookups and joins per client address per minute

_create_bucket = limits.TokenBucket()
_open_bucket = limits.TokenBucket()

Experience = Literal["ai_only", "cinematic"]


def public_calls_enabled() -> bool:
    return os.getenv("PLATFORM_PUBLIC_CALLS", "").strip().lower() in {"1", "true", "yes", "on"}


def calls_org_id() -> str:
    return os.getenv("PLATFORM_CALLS_ORG_ID", "").strip() or DEFAULT_ORGANIZATION_ID


def configured_providers() -> dict[str, str]:
    """``PLATFORM_CALLS_PROVIDERS="stt=elevenlabs,llm=google-translate,tts=elevenlabs"``: the Runtime
    providers every shareable-link call uses. A slot left out uses the Runtime's own default. The names
    are the Runtime's provider names; the browser never chooses them."""

    chosen: dict[str, str] = {}
    for item in os.getenv("PLATFORM_CALLS_PROVIDERS", "").split(","):
        kind, _, name = item.partition("=")
        kind, name = kind.strip().lower(), name.strip()
        if kind in {"stt", "llm", "tts"} and name:
            chosen[kind] = name
    return chosen


def _provider_ids(db: Session, ctx: AuthContext) -> dict[str, UUID]:
    """The calls organisation's enabled provider record for each configured slot (created on first use)."""

    ids: dict[str, UUID] = {}
    for kind, name in configured_providers().items():
        record = db.scalars(
            scoped(db, ctx, ProviderConfig).where(
                ProviderConfig.kind == kind, ProviderConfig.provider == name, ProviderConfig.enabled.is_(True)
            )
        ).first()
        if record is None:
            record = ProviderConfig(kind=kind, provider=name, configuration={}, enabled=True, organization_id=ctx.org_id)
            db.add(record)
            db.flush()
        ids[kind] = UUID(record.id)
    return ids


def invite_ttl() -> int:
    try:
        return max(int(os.getenv("PLATFORM_CALL_INVITE_TTL_SECS", "")), 30)
    except ValueError:
        return DEFAULT_INVITE_TTL_SECS


def _limit(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return max(int(raw), 0) if raw else default
    except ValueError:
        return default


def client_address(request: Request) -> str:
    """The caller's address. ``X-Forwarded-For`` counts only behind a proxy you run
    (``PLATFORM_TRUST_PROXY=1``): otherwise anyone could pick their own bucket."""

    if os.getenv("PLATFORM_TRUST_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def public_ctx(request: Request) -> AuthContext:
    """The context of the three public routes (404 while the feature is off)."""

    if not public_calls_enabled():
        raise HTTPException(status_code=404, detail="calls_disabled")
    return get_ctx(request)


def _throttle(bucket: limits.TokenBucket, key: str, per_minute: int) -> None:
    if per_minute <= 0:
        return
    wait = bucket.take(key, per_minute)
    if wait > 0:
        retry_after = max(1, math.ceil(wait))
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limited", "message": f"Too many requests. Retry in {retry_after}s."},
            headers={"Retry-After": str(retry_after)},
        )


def limit_create(request: Request) -> None:
    # The bucket refills at per_hour/60 per minute and holds per_hour tokens: a burst of
    # per_hour calls, then one every 60/per_hour minutes.
    per_hour = _limit("PLATFORM_CALLS_PER_IP_HOUR", DEFAULT_CREATE_PER_IP)
    if per_hour <= 0:
        return
    wait = _create_bucket.take(client_address(request), per_hour, now=time.monotonic() / 60.0)
    if wait > 0:
        retry_after = max(1, math.ceil(wait * 60.0))
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limited", "message": f"Too many calls from this address. Retry in {retry_after}s."},
            headers={"Retry-After": str(retry_after)},
        )


def limit_open(request: Request) -> None:
    _throttle(_open_bucket, client_address(request), _limit("PLATFORM_CALL_OPEN_PER_IP_MINUTE", DEFAULT_OPEN_PER_IP))


def reset_limits() -> None:
    _create_bucket.reset()
    _open_bucket.reset()


class CallIn(BaseModel):
    """What a person chooses on the first screen."""

    languages: list[str] = Field(min_length=2, max_length=2)
    name: str | None = Field(default=None, max_length=100)
    voice_profile_id: UUID | None = None
    latency_mode: str = DEFAULT_LATENCY_MODE
    # Audio experience for testing: the translation alone (default) or the other person's
    # real voice kept low under it (see docs/dubbing.md).
    experience: Experience = "ai_only"


class JoinIn(BaseModel):
    name: str | None = Field(default=None, max_length=100)


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _split(code: str) -> tuple[str, str]:
    session_id, _, secret = code.partition(".")
    try:
        UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="invite_not_found") from None
    if len(secret) < 20 or len(code) > 120:
        raise HTTPException(status_code=404, detail="invite_not_found")
    return session_id, secret


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _creator_seat(contract: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for identity, entry in (contract.get("participants") or {}).items():
        if identity.startswith("browser-"):
            return identity, entry
    return None


def _open_invite(db: Session, ctx: AuthContext, code: str) -> tuple[VoiceSession, dict[str, Any], dict[str, Any], list[str]]:
    """The call, its invitation record, the creator's seat and the two languages for a valid, unexpired code.

    Wrong, expired and finished all say so with a distinct 4xx, never with the call's details.
    """

    session_id, secret = _split(code)
    session = db.scalars(scoped(db, ctx, VoiceSession).where(VoiceSession.id == session_id)).first()
    contract = dict(session.runtime_contract or {}) if session is not None else {}
    invite = dict(contract.get("call_invite") or {})
    if session is None or not invite or not secrets.compare_digest(invite.get("digest", ""), _digest(secret)):
        raise HTTPException(status_code=404, detail="invite_not_found")
    if datetime.now(timezone.utc).timestamp() >= float(invite.get("expires_at", 0)):
        raise HTTPException(status_code=410, detail="invite_expired")
    if session.status not in {"created", "running"}:
        raise HTTPException(status_code=410, detail="call_ended")
    seat = _creator_seat(contract)
    if seat is None:
        raise HTTPException(status_code=404, detail="invite_not_found")
    languages = list((contract.get("interpreter") or {}).get("languages") or [])
    return session, invite, seat[1], languages


@router.post("/calls", status_code=status.HTTP_201_CREATED)
async def create_call(
    payload: CallIn,
    request: Request,
    ctx: AuthContext = Depends(public_ctx),
    _limit_ip: None = Depends(limit_create),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    """Open a call and return the creator's connection. The invitation is made separately."""

    mine, theirs = (language.strip().lower() for language in payload.languages)
    translation = TranslationIn(
        languages=[mine, theirs],
        creator=ParticipantIn(language=mine, display_name=payload.name, voice_profile_id=payload.voice_profile_id),
        latency_mode=payload.latency_mode,
        dubbing="cinematic" if payload.experience == "cinematic" else None,
        **{f"{kind}_provider_id": record for kind, record in _provider_ids(db, ctx).items()},
    )
    return await create_translation(translation, ctx, None, db)


@router.post("/calls/{session_id}/invite")
async def create_invite(
    session_id: UUID,
    ctx: AuthContext = Depends(get_ctx),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    """The creator's shareable code (a new code replaces the old one). Seats nobody yet."""

    session = db.scalars(scoped(db, ctx, VoiceSession).where(VoiceSession.id == str(session_id)).with_for_update()).first()
    if session is None:
        raise HTTPException(status_code=404, detail="call_not_found")
    if session.status not in {"created", "running"}:
        raise HTTPException(status_code=409, detail="session_not_active")
    contract = dict(session.runtime_contract or {})
    if _creator_seat(contract) is None:
        raise HTTPException(status_code=409, detail="participants_not_enabled")
    secret = secrets.token_urlsafe(16)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=invite_ttl())
    contract["call_invite"] = {"digest": _digest(secret), "expires_at": expires_at.timestamp()}
    session.runtime_contract = contract
    db.commit()
    return {"code": f"{session.id}.{secret}", "expires_at": int(expires_at.timestamp()), "ttl_seconds": invite_ttl()}


@router.post("/calls/{session_id}/end")
async def end_call(
    session_id: UUID,
    ctx: AuthContext = Depends(get_ctx),
    db: Session = Depends(get_scoped_db),
) -> dict[str, str]:
    """Hang up. Either person can: it is a two-person call, so it ends for both (idempotent).

    ``POST /voice/sessions/{id}/end`` stays creator-only for every other kind of session.
    """

    return await end_session(db, ctx, session_id)


@router.get("/calls/invites/{code}")
async def invite_info(
    code: str,
    ctx: AuthContext = Depends(public_ctx),
    _limit_ip: None = Depends(limit_open),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    """What the guest sees before joining: who invited them and the two languages."""

    _, invite, creator, languages = _open_invite(db, ctx, code)
    guest_language = next((l for l in languages if l != creator["language"]), None)
    if guest_language is None:
        raise HTTPException(status_code=409, detail="participants_not_enabled")
    return {
        "from": creator.get("display_name") or "",
        "me": guest_language,
        "them": creator["language"],
        "expires_at": int(invite["expires_at"]),
    }


@router.post("/calls/invites/{code}/join")
async def join_call(
    code: str,
    payload: JoinIn,
    ctx: AuthContext = Depends(public_ctx),
    _limit_ip: None = Depends(limit_open),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    """Seat the guest (idempotent for this code) and return their connection."""

    session, invite, creator, languages = _open_invite(db, ctx, code)
    guest_language = next((l for l in languages if l != creator["language"]), None)
    if guest_language is None:
        raise HTTPException(status_code=409, detail="participants_not_enabled")
    seated = await add_voice_session_participant(
        UUID(session.id),
        GuestInviteIn(language=guest_language, display_name=payload.name),
        ctx,
        db,
        # The code's own digest: a repeat of the same invitation is the same guest, any other
        # attempt on a taken seat is refused (409 guest_already_invited).
        f"call-{invite['digest'][:40]}",
    )
    return {
        **seated,
        "session_id": session.id,
        "conversation_id": session.conversation_id,
        "me": guest_language,
        "them": creator["language"],
        "from": creator.get("display_name") or "",
    }
