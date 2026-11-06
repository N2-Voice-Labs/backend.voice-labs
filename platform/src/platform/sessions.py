"""Voice session routes: create, participants, phone participants, runtime handoff."""

import hashlib
import os
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import update
from sqlalchemy.orm import Session

from . import audit, auth, sip
from . import health as provider_health
from . import limits
from .context import AuthContext, get_ctx, require
from .livekit import (
    create_listener_token,
    create_participant_token,
    listen_room_name,
    phone_room_name,
    private_listening_enabled,
)
from .models import Agent as AgentRecord
from .models import (
    AgentKnowledgeDocument,
    AgentTool,
    Conversation,
    Deployment,
    KnowledgeDocument,
    ProviderConfig,
    Tool as ToolRecord,
    Voice,
    VoiceSession,
    utcnow,
)
from .tenancy import get_owned, get_public_agent, get_scoped_db, scoped
from .validation import validate_e164

router = APIRouter()

RUNTIME_PROVIDER_SETTINGS = {
    "stt": {"model", "language", "interim_results", "smart_format"},
    "llm": {"model", "temperature", "max_tokens", "top_p", "translation_model"},
    "tts": {"model", "language"},
}

class ParticipantIn(BaseModel):
    """A person in a two-participant interpreter call (TR-020).

    ``language`` must be one of the agent's interpreter languages. Their translated
    words are spoken in ``voice_profile_id``'s voice (a ready, consented voice profile);
    without one the agent's voice is used.
    """

    language: str = Field(min_length=1, max_length=16)
    voice_profile_id: UUID | None = None
    display_name: str | None = Field(default=None, max_length=100)


class PhoneParticipantIn(ParticipantIn):
    """The second person of a two-person interpreter call, reached by phone (E.164)."""

    phone_number: str

    @field_validator("phone_number")
    @classmethod
    def check_phone_number(cls, value: str) -> str:
        return validate_e164(value)


class SessionIn(BaseModel):
    agent_id: UUID
    deployment_id: UUID | None = None
    # The creating browser participant of a two-person interpreter call.
    participant: ParticipantIn | None = None


def member_or_session(request: Request) -> AuthContext:
    """Session-token callers (already matched to this session by ``auth``) or a member+.

    Used on the ``SESSION_SCOPED_ROUTES`` that guests and creators must reach: a ``vls_``
    token never passes :func:`require`, but its context carries the session's own
    organisation, so every lookup below stays inside it.
    """

    ctx = get_ctx(request)
    if ctx.principal == "session":
        return ctx
    return require("member")(request)


def _owned_session(
    db: Session, ctx: AuthContext, session_id: UUID | str, *, lock: bool = False
) -> VoiceSession:
    """The org's VoiceSession (optionally row-locked); 404 when missing or foreign."""

    statement = scoped(db, ctx, VoiceSession).where(VoiceSession.id == str(session_id))
    if lock:
        statement = statement.with_for_update()
    session = db.scalars(statement).first()
    if session is None:
        raise HTTPException(status_code=404, detail="voice_session_not_found")
    return session


def runtime_tools_for_agent(
    agent_id: str, db: Session, org_id: str | None = None
) -> list[dict[str, Any]]:
    """Assigned, enabled tools in the shape the Runtime registers as LLM functions."""

    query = (
        db.query(ToolRecord)
        .join(AgentTool, AgentTool.tool_id == ToolRecord.id)
        .filter(AgentTool.agent_id == agent_id, ToolRecord.enabled.is_(True))
    )
    if org_id is not None:
        query = query.filter(ToolRecord.organization_id == org_id)
    records = query.order_by(ToolRecord.name).all()
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "tool_type": tool.tool_type,
            "input_schema": tool.input_schema or {},
            "configuration": tool.configuration or {},
        }
        for tool in records
    ]


MAX_SESSION_PARTICIPANTS = 2


def ready_voice(db: Session, ctx: AuthContext, voice_id: str) -> Voice:
    """The org's voice when it is ready and provisioned; 409 otherwise.

    Another organisation's voice is "unavailable" exactly like a pending one: a foreign
    id must not be distinguishable from a missing id.
    """

    voice = db.scalars(scoped(db, ctx, Voice).where(Voice.id == str(voice_id))).first()
    if voice is None or voice.status != "ready" or not voice.provider_voice_id:
        raise HTTPException(status_code=409, detail="selected_voice_unavailable")
    return voice


def assert_voice_matches_tts(
    db: Session, ctx: AuthContext, voice: Voice, tts_provider: str | None
) -> None:
    """409 ``voice_provider_mismatch`` when ``voice`` belongs to another TTS provider.

    When no TTS provider is chosen the Runtime speaks with its default
    (``health.DEFAULT_PROVIDERS['tts']``), which is what the voice is compared against.
    Skipped only when the voice has no ``tts_provider_id`` or its provider row cannot be
    resolved within the organisation.
    """

    tts_provider = tts_provider or provider_health.DEFAULT_PROVIDERS["tts"]
    if not voice.tts_provider_id:
        return
    owner = db.scalars(
        scoped(db, ctx, ProviderConfig).where(ProviderConfig.id == voice.tts_provider_id)
    ).first()
    if owner is not None and owner.provider != tts_provider:
        raise HTTPException(status_code=409, detail="voice_provider_mismatch")


def participant_entry(
    mode: str | None,
    interpreter: dict[str, Any] | None,
    participant: ParticipantIn,
    db: Session,
    ctx: AuthContext,
    providers: dict[str, str] | None = None,
    check_voice_provider: bool = True,
) -> dict[str, Any]:
    """Validate a participant against an interpreter and build its contract entry.

    Raises 422 ``participants_require_cascaded_interpreter`` / ``invalid_participant_language``
    and 409 ``selected_voice_unavailable``. Only the cascaded engine can speak in a
    chosen voice and take a language per speaker.
    """

    interpreter = interpreter or {}
    if mode != "interpreter" or interpreter.get("engine", "cascaded") != "cascaded":
        raise HTTPException(
            status_code=422, detail="participants_require_cascaded_interpreter"
        )
    language = participant.language.strip().lower()
    if language not in (interpreter.get("languages") or []):
        raise HTTPException(status_code=422, detail="invalid_participant_language")
    voice_id = None
    if participant.voice_profile_id is not None:
        voice = ready_voice(db, ctx, str(participant.voice_profile_id))
        if check_voice_provider:
            assert_voice_matches_tts(db, ctx, voice, (providers or {}).get("tts"))
        voice_id = voice.provider_voice_id
    display_name = (participant.display_name or "").strip() or None
    return {"language": language, "voice_id": voice_id, "display_name": display_name}


def uses_private_listening(mode: str | None, interpreter: dict[str, Any] | None) -> bool:
    """Whether participants of this interpreter hear only through their own listen room.

    True for two-person interpreter sessions unless ``hear_original`` is set (everyone
    then shares one room). When the scheme is switched off with
    ``PLATFORM_PRIVATE_LISTENING=0`` such a session is refused (409) rather than started
    in a shared room that can leak the original microphones.
    """

    if mode != "interpreter" or (interpreter or {}).get("hear_original"):
        return False
    if not private_listening_enabled():
        raise HTTPException(status_code=409, detail="private_listening_required")
    return True


def participant_connection(
    room_name: str, identity: str, *, configured: bool, private: bool
) -> dict[str, Any]:
    """LiveKit ``connection`` of one participant.

    With ``private`` the session-room token cannot subscribe (nobody's microphone can
    ever reach the participant, whatever the client does) and ``listen`` is a second,
    subscribe-only connection to a room that only the Runtime publishes to: the
    participant's translation arrives there. See ``docs/contracts.md``.
    """

    connection: dict[str, Any] = {
        "url": os.getenv("LIVEKIT_URL", ""),
        "token": (
            create_participant_token(room_name, identity, can_subscribe=not private)
            if configured
            else ""
        ),
    }
    if private and configured:
        connection["listen"] = {
            "url": connection["url"],
            "token": create_listener_token(room_name, identity),
            "room_name": listen_room_name(room_name, identity),
            "track": f"translation-for-{identity}",
        }
    return connection


def agent_has_knowledge(agent_id: str, db: Session, org_id: str | None = None) -> bool:
    query = (
        db.query(AgentKnowledgeDocument.document_id)
        .join(
            KnowledgeDocument,
            KnowledgeDocument.id == AgentKnowledgeDocument.document_id,
        )
        .filter(
            AgentKnowledgeDocument.agent_id == agent_id,
            KnowledgeDocument.enabled.is_(True),
        )
    )
    if org_id is not None:
        query = query.filter(KnowledgeDocument.organization_id == org_id)
    return query.first() is not None


def _finish_if_active(db: Session, session: VoiceSession, final_status: str) -> bool:
    """Move an active session (and its conversation) to a terminal status, race-safely.

    A conditional UPDATE (``WHERE status IN ('created','running')``) so a terminal status
    committed concurrently (Runtime ingest ``failed``, the reaper) is never overwritten
    from a stale in-memory object. Does not commit; returns whether the row changed.
    """

    finished_at = utcnow()
    result = db.execute(
        update(VoiceSession)
        .where(
            VoiceSession.id == str(session.id),
            VoiceSession.status.in_(limits.ACTIVE_SESSION_STATUSES),
        )
        .values(status=final_status, ended_at=finished_at)
        .execution_options(synchronize_session=False)
    )
    if not result.rowcount:
        return False
    db.execute(
        update(Conversation)
        .where(
            Conversation.id == session.conversation_id,
            Conversation.status.not_in(("ended", "failed")),
        )
        .values(status=final_status, ended_at=finished_at)
        .execution_options(synchronize_session=False)
    )
    return True


def _runtime_http_error(exc: Exception) -> HTTPException:
    """The API error for a failed Runtime call: 503 ``runtime_at_capacity`` or ``runtime_unavailable``."""

    if _is_runtime_at_capacity(exc):
        return HTTPException(
            status_code=503,
            detail={
                "code": "runtime_at_capacity",
                "message": "All voice workers are busy, please try again shortly.",
            },
        )
    return provider_health.runtime_unavailable()


def _is_runtime_at_capacity(exc: Exception) -> bool:
    """Whether the Runtime answered 503 ``{"code": "runtime_at_capacity"}`` (bare or in ``detail``)."""

    if not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code != 503:
        return False
    try:
        body = exc.response.json()
    except ValueError:
        return False
    if isinstance(body, dict) and isinstance(body.get("detail"), dict):
        body = body["detail"]
    return isinstance(body, dict) and body.get("code") == "runtime_at_capacity"


async def open_voice_session(
    db: Session,
    ctx: AuthContext,
    *,
    agent_id: UUID | str,
    deployment_id: UUID | str | None = None,
    participant: ParticipantIn | None = None,
    allow_system: bool = False,
    preflighted: bool = False,
) -> dict[str, Any]:
    """Validate, persist and hand off a new voice session inside ``ctx.org_id``.

    Shared by ``POST /voice/sessions`` and the in-process callers (SIP bridge, campaign
    dispatcher), which pass a system ctx instead of calling the HTTP API. The agent,
    deployment, voice and providers are loaded within the organisation: another
    organisation's agent or deployment is a 404, its voice or provider "unavailable".
    Raises ``HTTPException`` for every failure.
    """

    org_id = ctx.org_id
    limits.check_active_session_cap(db, ctx)
    # System-managed (hidden interpreter) agents are only reachable by /translations.
    agent = get_public_agent(db, ctx, agent_id, allow_system=allow_system)
    if agent.status != "enabled":
        raise HTTPException(status_code=409, detail="agent_not_enabled")
    deployment = None
    if deployment_id is not None:
        deployment = get_owned(
            db, ctx, Deployment, str(deployment_id), not_found_detail="deployment_not_found"
        )
        if deployment.status != "active":
            raise HTTPException(status_code=409, detail="deployment_not_active")
        if deployment.agent_id != agent.id:
            raise HTTPException(status_code=409, detail="deployment_agent_mismatch")
    creator_entry = None
    if participant is not None:
        creator_entry = participant_entry(
            agent.mode, agent.interpreter, participant, db, ctx,
            # Checked below against the agent's resolved providers (or the default).
            check_voice_provider=False,
        )
    runtime_url = os.getenv("RUNTIME_API_URL")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if runtime_url and not runtime_token:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")
    runtime_voice_id = agent.voice_id
    used_voices: dict[str, Voice] = {}
    if agent.voice_profile_id:
        voice = ready_voice(db, ctx, agent.voice_profile_id)
        runtime_voice_id = voice.provider_voice_id
        used_voices[voice.id] = voice
    if participant is not None and participant.voice_profile_id is not None:
        # Already validated by participant_entry; loaded again only to audit its use.
        voice = ready_voice(db, ctx, str(participant.voice_profile_id))
        used_voices[voice.id] = voice
    providers = {}
    provider_settings = {}
    for kind in ("stt", "llm", "tts"):
        provider_id = getattr(agent, f"{kind}_provider_id")
        if provider_id:
            provider = db.scalars(
                scoped(db, ctx, ProviderConfig).where(ProviderConfig.id == provider_id)
            ).first()
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
    for voice in used_voices.values():
        assert_voice_matches_tts(db, ctx, voice, providers.get("tts"))

    if runtime_url and not preflighted:
        # Before any row exists: a down provider must not leave a junk session behind.
        await provider_health.preflight_providers(
            provider_health.required_providers(agent.mode, agent.interpreter, providers)
        )

    conversation = Conversation(
        agent_id=agent.id,
        deployment_id=deployment.id if deployment else None,
        channel=deployment.channel if deployment else "web",
        organization_id=org_id,
    )
    db.add(conversation)
    db.flush()
    session_id = str(uuid4())
    session = VoiceSession(
        id=session_id,
        conversation_id=conversation.id,
        room_name=f"voice-labs-{session_id}",
        organization_id=org_id,
    )
    session_token, session.session_token_hash = auth.new_session_token()
    configured = all(
        os.getenv(name)
        for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    )
    creator_identity = f"browser-{session.id}"
    private_listening = creator_entry is not None and uses_private_listening(
        agent.mode, agent.interpreter
    )
    connection = participant_connection(
        session.room_name, creator_identity, configured=configured, private=private_listening
    )
    runtime_status = "not_configured"
    request_payload = {
        "session_id": session.id,
        "conversation_id": conversation.id,
        "agent_id": agent.id,
        # Informational: the Runtime does not use it for authorisation.
        "organization_id": org_id,
        "room_name": session.room_name,
        "system_prompt": agent.system_prompt,
        "voice_id": runtime_voice_id,
        "mode": agent.mode,
        "interpreter": agent.interpreter,
        "tools": runtime_tools_for_agent(agent.id, db, org_id),
        "knowledge_enabled": agent_has_knowledge(agent.id, db, org_id),
    }
    if agent.translation:
        # Agent + translator (TR-040); absent for ordinary agents.
        request_payload["translation"] = agent.translation
    if creator_entry is not None:
        request_payload["participants"] = {creator_identity: creator_entry}
    if private_listening:
        request_payload["private_listening"] = True
    if providers:
        request_payload["providers"] = providers
    if provider_settings:
        request_payload["provider_settings"] = provider_settings
    transfer_to = (deployment.configuration or {}).get("transfer_to") if deployment else None
    if isinstance(transfer_to, str) and transfer_to:
        request_payload["transfer_to"] = transfer_to
    session.runtime_contract = request_payload
    db.add(session)
    for voice in used_voices.values():
        if voice.source in {"clone", "design"}:
            audit.record(
                db,
                ctx,
                "voice.used",
                resource_type="voice",
                resource_id=voice.id,
                metadata={"session_id": session.id},
            )
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
                        if _is_runtime_at_capacity(exc):
                            raise  # a full Runtime will not have room a moment later
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
            raise _runtime_http_error(exc) from exc
    elif configured:
        runtime_status = "manual_start_required"
    return {
        "session_id": session.id,
        "conversation_id": conversation.id,
        "connection": connection,
        "room_name": session.room_name,
        "session_status": session.status,
        "connection_status": "configured" if configured else "not_configured",
        "runtime_status": runtime_status,
        "session_token": session_token,
    }


@router.post("/voice/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionIn,
    ctx: AuthContext = Depends(require("member")),
    _rate: None = Depends(limits.limit_session_creation),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    return await open_voice_session(
        db,
        ctx,
        agent_id=payload.agent_id,
        deployment_id=payload.deployment_id,
        participant=payload.participant,
    )


@router.get("/voice/sessions/{session_id}")
async def get_voice_session(
    session_id: UUID,
    ctx: AuthContext = Depends(member_or_session),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    session = _owned_session(db, ctx, session_id)
    return {
        "id": session.id,
        "conversation_id": session.conversation_id,
        "status": session.status,
        "room_name": session.room_name,
        "transport": session.transport,
        "started_at": session.started_at,
        "ended_at": session.ended_at,
    }


class GuestInviteIn(ParticipantIn):
    """A browser guest invite.

    ``replace`` is the deliberate way to swap out the guest already seated: it revokes
    the previous guest's session tokens, frees their Runtime slot and disconnects them
    from the LiveKit rooms before the new seat is issued. Without it an existing guest is
    never displaced.
    """

    replace: bool = False


def _key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def disconnect_livekit_participant(room_name: str, identity: str) -> None:
    """Best-effort: drop ``identity`` from its session room and private listen room."""

    if not all(os.getenv(n) for n in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")):
        return
    try:
        from livekit import api

        async with sip._new_client() as lkapi:
            for room in (room_name, listen_room_name(room_name, identity)):
                try:
                    await lkapi.room.remove_participant(
                        api.RoomParticipantIdentity(room=room, identity=identity)
                    )
                except Exception:  # noqa: BLE001 - not connected / room gone
                    continue
    except Exception:  # noqa: BLE001
        return


@router.post("/voice/sessions/{session_id}/participants")
async def add_voice_session_participant(
    session_id: UUID,
    payload: GuestInviteIn,
    ctx: AuthContext = Depends(member_or_session),
    db: Session = Depends(get_scoped_db),
    idempotency_key: str | None = Header(default=None, min_length=8, max_length=128),
) -> dict[str, Any]:
    """Add the second person to a two-participant interpreter call (TR-020).

    Returns the guest's LiveKit connection for the session's room. The participant is
    stored in the session's runtime contract (so an ensure-running restart carries it)
    and the live Runtime session is told, so the next utterance is already translated
    for them. Runtime routes each translation only to the other person and keeps the
    original microphones private unless the interpreter sets ``hear_original``.

    Retry-safe with an ``Idempotency-Key`` header: the seat is persisted before the
    Runtime is told, and a retry with the same key and language gets the same seat and
    fresh credentials while every credential already issued stays valid. An invite
    without a matching key never displaces a seated guest (409 ``guest_already_invited``);
    ``replace: true`` does so deliberately.
    """

    session = _owned_session(db, ctx, session_id, lock=True)
    if session.status not in {"created", "running"}:
        raise HTTPException(status_code=409, detail="session_not_active")
    contract = dict(session.runtime_contract or {})
    participants = dict(contract.get("participants") or {})
    if not participants:
        raise HTTPException(status_code=409, detail="participants_not_enabled")
    seat = next((i for i in participants if i.startswith("guest-")), None)
    invite = dict(contract.get("guest_invite") or {})
    key_hash = _key_digest(idempotency_key) if idempotency_key else None
    retry = False
    if seat is not None:
        if key_hash is not None and invite.get("key_hash") == key_hash:
            if participants[seat].get("language") != payload.language.strip().lower():
                raise HTTPException(status_code=409, detail="idempotency_key_conflict")
            retry = True
        elif not payload.replace:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "guest_already_invited",
                    "message": (
                        "A guest already holds this call's second seat. Retry with the "
                        "same Idempotency-Key, or send replace=true to replace them."
                    ),
                },
            )
    elif len(participants) >= MAX_SESSION_PARTICIPANTS:
        raise HTTPException(status_code=409, detail="session_full")
    replaced = seat if (seat is not None and not retry) else None
    others = {i: e for i, e in participants.items() if i != seat}
    entry = participant_entry(
        contract.get("mode"), contract.get("interpreter"), payload, db, ctx,
        contract.get("providers"),
    )
    if retry:
        entry = participants[seat]
    elif any(existing["language"] == entry["language"] for existing in others.values()):
        # Two people speaking the same language have nothing to interpret.
        raise HTTPException(status_code=422, detail="participant_languages_must_differ")
    identity = seat if retry else f"guest-{uuid4().hex[:8]}"
    configured = all(
        os.getenv(name)
        for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    )
    connection = participant_connection(
        session.room_name,
        identity,
        configured=configured,
        private=bool(contract.get("private_listening")),
    )

    # Persist the seat first, then apply it to the Runtime: if the Runtime call or the
    # response is lost, Platform's durable state still names the seat and a retry with
    # the same key reconciles it without invalidating anything already issued.
    if replaced is not None:
        participants.pop(replaced)
    participants[identity] = entry
    contract["participants"] = participants
    contract["guest_invite"] = {"identity": identity, "key_hash": key_hash}
    session.runtime_contract = contract
    guest_token, guest_hash = auth.new_session_token()
    if replaced is not None:
        session.guest_token_hashes = [guest_hash]
    else:
        session.guest_token_hashes = [*(session.guest_token_hashes or []), guest_hash]
    db.commit()
    if replaced is not None:
        await withdraw_runtime_participant(session.id, replaced)
        await disconnect_livekit_participant(session.room_name, replaced)
    await notify_runtime_participant(session.id, identity, entry)
    return {
        "participant_identity": identity,
        "connection": connection,
        "session_token": guest_token,
        "replaced_participant_identity": replaced,
    }


async def notify_runtime_participant(
    session_id: str, identity: str, entry: dict[str, Any]
) -> None:
    """Tell the live Runtime session about a new participant (503 when it cannot)."""

    runtime_url = os.getenv("RUNTIME_API_URL")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if not runtime_url:
        return
    if not runtime_token:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{runtime_url.rstrip('/')}/sessions/{session_id}/participants",
                headers={"X-Runtime-Token": runtime_token},
                json={"identity": identity, **entry},
            )
            # 404: the Runtime does not hold the session (not started yet or
            # restarted); the stored contract starts it with this participant.
            if response.status_code != 404:
                response.raise_for_status()
    except httpx.HTTPError as exc:
        raise provider_health.runtime_unavailable() from exc


async def withdraw_runtime_participant(session_id: str, identity: str) -> None:
    """Best-effort: free the Runtime slot of a participant whose call did not connect."""

    runtime_url = os.getenv("RUNTIME_API_URL")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if not runtime_url or not runtime_token:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.delete(
                f"{runtime_url.rstrip('/')}/sessions/{session_id}/participants/{identity}",
                headers={"X-Runtime-Token": runtime_token},
            )
    except httpx.HTTPError:
        return


@router.post("/voice/sessions/{session_id}/phone-participants", status_code=status.HTTP_201_CREATED)
async def add_voice_session_phone_participant(
    session_id: UUID,
    payload: PhoneParticipantIn,
    ctx: AuthContext = Depends(member_or_session),
    db: Session = Depends(get_scoped_db),
) -> dict[str, Any]:
    """Dial the second person of a two-person interpreter call by phone.

    Same rules as ``POST /voice/sessions/{id}/participants`` (the guest is a phone
    number instead of a browser). With private listening the phone is dialled into its
    own room (``phone_room_name``) that only the Runtime joins, so it can never hear the
    other person's voice; otherwise it joins the session room and gets the unsubscribe
    routing. The Runtime is told first so it is in the room when the call is answered;
    if the call does not connect the participant is withdrawn and the slot is free again.
    """

    session = _owned_session(db, ctx, session_id, lock=True)
    if session.status not in {"created", "running"}:
        raise HTTPException(status_code=409, detail="session_not_active")
    contract = dict(session.runtime_contract or {})
    participants = dict(contract.get("participants") or {})
    if not participants:
        raise HTTPException(status_code=409, detail="participants_not_enabled")
    if len(participants) >= MAX_SESSION_PARTICIPANTS:
        raise HTTPException(status_code=409, detail="session_full")
    entry = participant_entry(
        contract.get("mode"), contract.get("interpreter"), payload, db, ctx,
        contract.get("providers"),
    )
    if any(existing["language"] == entry["language"] for existing in participants.values()):
        raise HTTPException(status_code=422, detail="participant_languages_must_differ")
    if not sip.sip_configured():
        raise HTTPException(status_code=409, detail="sip_not_configured")
    identity = f"{sip.SIP_IDENTITY_PREFIX}guest-{uuid4().hex[:8]}"
    private = bool(contract.get("private_listening"))
    if private:
        entry["phone_room"] = True
    dial_room = (
        phone_room_name(session.room_name, identity) if private else session.room_name
    )

    await notify_runtime_participant(session.id, identity, entry)
    try:
        participant_id = await sip.dial(
            payload.phone_number,
            dial_room,
            identity,
            display_name=entry["display_name"],
        )
    except sip.SipError as exc:
        await withdraw_runtime_participant(session.id, identity)
        raise HTTPException(status_code=502, detail="sip_dial_failed") from exc

    participants[identity] = entry
    contract["participants"] = participants
    session.runtime_contract = contract
    db.commit()
    return {
        "participant_identity": identity,
        "participant_id": participant_id,
        "phone_number": payload.phone_number,
        "room_name": dial_room,
        "private_listening": private,
    }


@router.post("/voice/sessions/{session_id}/ensure-running")
async def ensure_voice_session_running(
    session_id: UUID,
    ctx: AuthContext = Depends(member_or_session),
    db: Session = Depends(get_scoped_db),
) -> dict[str, str]:
    session = _owned_session(db, ctx, session_id)
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
        raise _runtime_http_error(exc) from exc

    if runtime_status in {"failed", "ended", "cancelled"}:
        _finish_if_active(
            db, session, "ended" if runtime_status == "cancelled" else runtime_status
        )
        db.commit()
        db.refresh(session)
    return {
        "session_status": session.status,
        "runtime_status": str(runtime_status or "unknown"),
    }


@router.post("/voice/sessions/{session_id}/end")
async def end_voice_session(
    session_id: UUID,
    ctx: AuthContext = Depends(member_or_session),
    db: Session = Depends(get_scoped_db),
) -> dict[str, str]:
    return await end_session(db, ctx, session_id)


async def get_session_status(db: Session, ctx: AuthContext, session_id: UUID | str) -> str | None:
    """Fresh ``status`` of an org's session (None when missing); never stale."""

    session = db.scalars(
        scoped(db, ctx, VoiceSession)
        .where(VoiceSession.id == str(session_id))
        .execution_options(populate_existing=True)
    ).first()
    return session.status if session is not None else None


async def end_session(
    db: Session, ctx: AuthContext, session_id: UUID | str
) -> dict[str, str]:
    """End an org's session (Runtime first, then rows, then phone legs). Idempotent."""

    session = _owned_session(db, ctx, session_id)
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
            raise provider_health.runtime_unavailable() from exc
    # Runtime ingest may have committed a terminal status (e.g. ``failed`` from a
    # SessionFailed event) while the Runtime call was in flight; never downgrade it.
    _finish_if_active(db, session, "ended")
    db.commit()
    db.refresh(session)
    await hang_up_phone_participants(session)
    return {"status": session.status}


async def hang_up_phone_participants(session: VoiceSession) -> None:
    """Best-effort: end the calls of phone participants (they sit in rooms of their own)."""

    for identity, entry in ((session.runtime_contract or {}).get("participants") or {}).items():
        if identity.startswith(sip.SIP_IDENTITY_PREFIX):
            room = (
                phone_room_name(session.room_name, identity)
                if entry.get("phone_room")
                else session.room_name
            )
            await sip.hangup(room, identity)
