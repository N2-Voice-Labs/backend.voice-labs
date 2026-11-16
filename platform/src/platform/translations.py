"""Translation sessions: two people talking across languages, no agent to configure.

``POST /translations`` opens a two-person interpreter call. Behind it sits a hidden,
system-managed interpreter agent (``Agent.system``) per organisation and configuration, so
the existing session, contract, participant and billing paths are reused unchanged. The
hidden agent has no prompt, tools or knowledge; the Runtime interpreter pipeline never
answers questions, it only translates.
"""

import hashlib
import json
import os
from datetime import timedelta
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import health as provider_health
from . import limits
from .agents import InterpreterConfig, _load_in_org
from .context import AuthContext, require
from .models import Agent as AgentRecord
from .models import Conversation, ProviderConfig, VoiceSession, utcnow
from .sessions import (
    GuestInviteIn,
    ParticipantIn,
    add_voice_session_participant,
    member_or_session,
    open_voice_session,
)
from .tenancy import get_scoped_db, scoped
from .validation import AUTO_DIRECTION_LANGUAGES, DEFAULT_LATENCY_MODE, LATENCY_MODES

router = APIRouter(tags=["translations"])

SYSTEM_AGENT_PREFIX = "translator:"


class TranslationIn(BaseModel):
    """The two languages and the creating person's seat."""

    languages: list[str]
    creator: ParticipantIn
    speaker_language: Literal["declared", "auto"] = "declared"
    latency_mode: str = DEFAULT_LATENCY_MODE
    hear_original: bool = False
    # Cinematic dubbing: the other person's real voice under the translation (default off).
    dubbing: bool | Literal["subtle", "balanced", "cinematic", "present"] | None = None
    glossary: list[dict[str, Any]] | None = None
    # Optional provider records per slot (e.g. ElevenLabs STT, Gemini translation, ElevenLabs
    # voice); an omitted slot uses Runtime's default provider for it.
    stt_provider_id: UUID | None = None
    llm_provider_id: UUID | None = None
    tts_provider_id: UUID | None = None

    @field_validator("latency_mode")
    @classmethod
    def check_latency_mode(cls, value: str) -> str:
        if value not in LATENCY_MODES:
            raise ValueError("invalid_latency_mode")
        return value


def interpreter_config(payload: TranslationIn) -> dict[str, Any]:
    """The interpreter block of the hidden agent for ``payload``.

    Always the cascaded engine: only it takes a language and a voice per speaker, which two
    people talking to each other need (the realtime engine is one-way). For these lanes,
    ``speaker_language`` controls detection; ``direction`` only matters for validation: ``auto``
    where Deepgram can detect both languages, otherwise ``one_way`` (e.g. Korean).
    """

    languages = [lang.strip().lower() for lang in payload.languages]
    try:
        direction = (
            "auto"
            if len(languages) == 2 and all(l in AUTO_DIRECTION_LANGUAGES for l in languages)
            else "one_way"
        )
        config = InterpreterConfig(
            languages=languages,
            direction=direction,
            **(
                {"speaker_language": payload.speaker_language}
                if "speaker_language" in payload.model_fields_set else {}
            ),
            engine="cascaded",
            latency_mode=payload.latency_mode,
            hear_original=payload.hear_original,
            dubbing=payload.dubbing,
            glossary=payload.glossary,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid_interpreter_config") from exc
    data = config.model_dump()
    if not data.get("glossary"):
        data.pop("glossary", None)
    return data


def _find_system_agent(db: Session, ctx: AuthContext, key: str) -> AgentRecord | None:
    return db.scalars(
        scoped(db, ctx, AgentRecord).where(
            AgentRecord.system.is_(True), AgentRecord.system_key == key
        )
    ).first()


def system_agent_for(
    db: Session,
    ctx: AuthContext,
    interpreter: dict[str, Any],
    tts_provider_id: str | None = None,
    *,
    stt_provider_id: str | None = None,
    llm_provider_id: str | None = None,
) -> AgentRecord:
    """The org's hidden interpreter agent for this exact config (created on first use).

    Flushes only (never commits): the caller's transaction owns the row.

    Race-free: ``(organization_id, system_key)`` is unique, so a concurrent creator makes
    the insert fail with ``IntegrityError``; the loser rolls back its (still write-free)
    transaction and re-selects the winner's row. Call it before any other write.
    """

    # Hash a canonical form only (glossary entries sorted) so reordered glossaries share a
    # row; the stored and sent config keeps the caller's order.
    canonical = dict(interpreter)
    if canonical.get("glossary"):
        canonical["glossary"] = sorted(
            canonical["glossary"], key=lambda entry: json.dumps(entry, sort_keys=True)
        )
    # Only when set: an omitted provider hashes exactly as before these fields existed.
    for field, value in (
        ("stt_provider_id", stt_provider_id),
        ("llm_provider_id", llm_provider_id),
        ("tts_provider_id", tts_provider_id),
    ):
        if value:
            canonical[field] = value
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()
    ).hexdigest()[:32]
    key = f"{SYSTEM_AGENT_PREFIX}{digest}"
    agent = _find_system_agent(db, ctx, key)
    if agent is not None:
        return agent
    bound = limits.max_new_translation_configs_per_day()
    if bound > 0 and ctx.org_id is not None:
        since = utcnow() - timedelta(seconds=limits.TRANSLATION_CONFIG_WINDOW_SECONDS)
        existing_count = db.scalar(
            select(func.count()).select_from(AgentRecord).where(
                AgentRecord.organization_id == ctx.org_id,
                AgentRecord.system.is_(True),
                AgentRecord.created_at >= since,
            )
        ) or 0
        if existing_count >= bound:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "too_many_translation_configs",
                    "message": (
                        f"This organization already created {bound} new translation "
                        "configurations in the last 24 hours. Reuse an existing one "
                        "or try again later."
                    ),
                },
            )
    agent = AgentRecord(
        name=f"Interpreter {' / '.join(interpreter['languages'])}",
        description="",
        system_prompt="",
        enabled_languages=list(interpreter["languages"]),
        mode="interpreter",
        interpreter=interpreter,
        stt_provider_id=stt_provider_id,
        llm_provider_id=llm_provider_id,
        tts_provider_id=tts_provider_id,
        system=True,
        system_key=key,
        organization_id=ctx.org_id,
    )
    try:
        db.add(agent)
        db.flush()
    except IntegrityError:
        # No SAVEPOINT on purpose: pysqlite autocommits work done inside one, which would
        # break the "uncommitted until the session commits" guarantee. Instead the whole
        # transaction is rolled back, which is safe because this must be the caller's first
        # write (create_translation only reads before it), then the winner is re-selected.
        db.rollback()
        existing = _find_system_agent(db, ctx, key)
        if existing is None:
            raise
        return existing
    return agent


def _participants_view(contract: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Participants without provider voice ids."""

    return [
        {
            "identity": identity,
            "language": entry.get("language"),
            "display_name": entry.get("display_name"),
        }
        for identity, entry in ((contract or {}).get("participants") or {}).items()
    ]


def _view(session: VoiceSession) -> dict[str, Any]:
    contract = session.runtime_contract or {}
    interpreter = contract.get("interpreter") or {}
    return {
        "id": session.id,
        "conversation_id": session.conversation_id,
        "status": session.status,
        "started_at": session.started_at,
        "ended_at": session.ended_at,
        "languages": interpreter.get("languages"),
        "latency_mode": interpreter.get("latency_mode"),
        "participants": _participants_view(contract),
    }


def _translation_session(db: Session, ctx: AuthContext, session_id: UUID) -> VoiceSession | None:
    """The org's session iff its conversation belongs to a system agent (a translation)."""

    return db.scalars(
        scoped(db, ctx, VoiceSession)
        .join(Conversation, Conversation.id == VoiceSession.conversation_id)
        .join(AgentRecord, AgentRecord.id == Conversation.agent_id)
        .where(VoiceSession.id == str(session_id), AgentRecord.system.is_(True))
    ).first()


@router.post("/translations", status_code=status.HTTP_201_CREATED)
async def create_translation(
    payload: TranslationIn,
    ctx: AuthContext = Depends(require("member")),
    _rate: None = Depends(limits.limit_session_creation),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    interpreter = interpreter_config(payload)
    if payload.creator.language.strip().lower() not in interpreter["languages"]:
        raise HTTPException(status_code=422, detail="invalid_participant_language")
    # Refuse before creating anything, and only flush the hidden agent: it commits together
    # with the session in open_voice_session. Any refusal raises, and get_db closes the
    # session, rolling the uncommitted agent back (no orphan rows).
    provider_ids: dict[str, str] = {}
    provider_names: dict[str, str] = {}
    for kind in ("stt", "llm", "tts"):
        requested = getattr(payload, f"{kind}_provider_id")
        if requested is None:
            continue
        provider = _load_in_org(db, ctx, ProviderConfig, str(requested))
        if provider is None or provider.kind != kind or not provider.enabled:
            # Same answer for a foreign, missing, wrong-kind or disabled id.
            raise HTTPException(status_code=422, detail=f"invalid_{kind}_provider_id")
        provider_ids[kind] = str(requested)
        provider_names[kind] = provider.provider
    limits.check_active_session_cap(db, ctx)
    # Provider preflight is a network call: run it before system_agent_for takes the
    # write lock on the hidden agent row, so concurrent requests are not stalled behind it.
    if os.getenv("RUNTIME_API_URL"):
        await provider_health.preflight_providers(
            provider_health.required_providers(
                "interpreter", interpreter, provider_names or None
            )
        )
    agent = system_agent_for(
        db,
        ctx,
        interpreter,
        provider_ids.get("tts"),
        stt_provider_id=provider_ids.get("stt"),
        llm_provider_id=provider_ids.get("llm"),
    )
    return await open_voice_session(
        db,
        ctx,
        agent_id=agent.id,
        participant=payload.creator,
        allow_system=True,
        preflighted=True,
    )


@router.post("/translations/{session_id}/invite")
async def invite_to_translation(
    session_id: UUID,
    guest: GuestInviteIn,
    ctx: AuthContext = Depends(member_or_session),
    db: Session = Depends(get_scoped_db),
    idempotency_key: str | None = Header(default=None, min_length=8, max_length=128),
) -> dict[str, Any]:
    """Seat the second person (a browser guest); same rules as ``/voice/sessions/{id}/participants``.

    For a phone guest use ``POST /voice/sessions/{id}/phone-participants``.
    """

    if _translation_session(db, ctx, session_id) is None:
        raise HTTPException(status_code=404, detail="translation_not_found")
    return await add_voice_session_participant(
        session_id, guest, ctx, db, idempotency_key
    )


@router.get("/translations")
async def list_translations(
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
    limit: int = 50,
) -> list[dict[str, Any]]:
    statement = (
        scoped(db, ctx, VoiceSession)
        .join(Conversation, Conversation.id == VoiceSession.conversation_id)
        .join(AgentRecord, AgentRecord.id == Conversation.agent_id)
        .where(AgentRecord.system.is_(True))
        .order_by(VoiceSession.started_at.desc())
        .limit(max(1, min(limit, 200)))
    )
    return [_view(session) for session in db.scalars(statement).all()]


@router.get("/translations/{session_id}")
async def get_translation(
    session_id: UUID,
    ctx: AuthContext = Depends(member_or_session),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    session = _translation_session(db, ctx, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="translation_not_found")
    return _view(session)
