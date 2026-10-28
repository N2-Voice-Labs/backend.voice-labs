"""Voice API: library, cloned, and designed voices with a validated lifecycle."""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import quote
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
)
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from . import audit, storage
from .context import AuthContext, get_ctx, require, role_at_least
from .models import Agent as AgentRecord, ProviderConfig, Voice as VoiceRecord, utcnow
from .tenancy import bypass, get_owned, get_scoped_db, link_check, scoped

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["voices"])

# Allowed status transitions: current -> permitted next statuses.
TRANSITIONS = {
    "pending": {"processing", "failed"},
    "processing": {"ready", "failed"},
    "failed": {"pending"},
    "ready": {"disabled"},
    "disabled": {"ready"},
}
# "deleting" (provider cleanup pending) and "deleted" (tombstone) are set only by
# DELETE /voices/{id}; they have no outgoing transitions.
VoiceStatus = Literal[
    "pending", "processing", "ready", "failed", "disabled", "deleting", "deleted"
]
VoiceSource = Literal["library", "clone", "design"]
SAMPLE_REFERENCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://\S+$")


class VoiceStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy: float | None = Field(default=None, ge=0, le=1)
    warmth: float | None = Field(default=None, ge=0, le=1)
    speed: float | None = Field(default=None, ge=0, le=1)
    style: str | None = Field(default=None, max_length=50)


class VoiceConsent(BaseModel):
    confirmed: StrictBool
    subject_name: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    method: Literal["self", "authorized"]


class VoiceBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    tts_provider_id: UUID | None = None
    languages: list[str] = Field(default_factory=lambda: ["en", "ru"], min_length=1)
    style: VoiceStyle = Field(default_factory=VoiceStyle)


class VoiceIn(VoiceBase):
    tts_provider_id: UUID
    provider_voice_id: str = Field(min_length=1, max_length=200)


class VoiceCloneIn(VoiceBase):
    sample_reference: str
    # Optional at the schema level so a missing consent yields voice_consent_required.
    consent: VoiceConsent | None = None


class VoiceDesignIn(VoiceBase):
    design_prompt: str = Field(min_length=1, max_length=2000)


class VoicePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    languages: list[str] | None = Field(default=None, min_length=1)
    style: VoiceStyle | None = None

    @field_validator("name", "languages", "style", mode="before")
    @classmethod
    def reject_null_for_required_fields(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("field cannot be null")
        return value


class VoiceStatusIn(BaseModel):
    status: VoiceStatus
    provider_voice_id: str | None = Field(default=None, min_length=1, max_length=200)
    failure_reason: str | None = None


class Voice(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    source: str
    status: str
    tts_provider_id: UUID | None
    provider_voice_id: str | None
    languages: list[str]
    style: dict[str, Any]
    design_prompt: str | None
    sample_reference: str | None
    consent: dict[str, Any] | None
    failure_reason: str | None
    deleted_at: Any = None
    created_at: Any
    updated_at: Any


def validate_tts_provider(
    provider_id: UUID | None, db: Session, ctx: AuthContext
) -> str | None:
    if provider_id is None:
        return None
    # Look the provider up across organisations so another org's id is a 404
    # (provider_config_not_found) rather than indistinguishable from a bad kind.
    with bypass(db):
        provider = db.scalars(
            select(ProviderConfig).where(ProviderConfig.id == str(provider_id))
        ).first()
    link_check(ctx, provider)
    if provider is None or provider.kind != "tts" or not provider.enabled:
        raise HTTPException(status_code=422, detail="invalid_tts_provider_id")
    return provider.id


def _is_admin(ctx: AuthContext) -> bool:
    return ctx.principal in {"superadmin", "dev", "system"} or role_at_least(
        ctx.role, "admin"
    )


def get_voice_or_404(
    voice_id: UUID, db: Session, ctx: AuthContext, *, include_deleted: bool = False
) -> VoiceRecord:
    record = get_owned(
        db, ctx, VoiceRecord, str(voice_id), not_found_detail="voice_not_found"
    )
    if record.status == "deleted" and not include_deleted:
        raise HTTPException(status_code=404, detail="voice_not_found")
    return record


def require_superadmin(ctx: AuthContext = Depends(get_ctx)) -> AuthContext:
    if ctx.principal not in {"superadmin", "dev"}:
        raise HTTPException(status_code=403, detail={"code": "superadmin_required"})
    return ctx


def save_voice(
    db: Session,
    ctx: AuthContext,
    payload: VoiceBase,
    audit_metadata: dict[str, Any] | None = None,
    **fields: Any,
) -> VoiceRecord:
    record = VoiceRecord(
        name=payload.name,
        tts_provider_id=validate_tts_provider(payload.tts_provider_id, db, ctx),
        languages=payload.languages,
        style=payload.style.model_dump(exclude_none=True),
        **fields,
    )
    db.add(record)
    db.flush()
    audit.record(
        db,
        ctx,
        "voice.created",
        resource_type="voice",
        resource_id=record.id,
        metadata={"source": record.source, **(audit_metadata or {})},
    )
    db.commit()
    db.refresh(record)
    return record


def _foreign_clone_voice_exists(
    db: Session, ctx: AuthContext, provider_voice_id: str
) -> bool:
    """True when another organisation owns a cloned voice with this provider id."""

    if ctx.org_id is None:
        return False
    with bypass(db):
        return (
            db.scalars(
                select(VoiceRecord.id).where(
                    VoiceRecord.source == "clone",
                    VoiceRecord.provider_voice_id == provider_voice_id,
                    VoiceRecord.organization_id != ctx.org_id,
                )
            ).first()
            is not None
        )


@router.post("/voices", response_model=Voice, status_code=status.HTTP_201_CREATED)
async def create_voice(
    payload: VoiceIn,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> Voice:
    if _foreign_clone_voice_exists(db, ctx, payload.provider_voice_id):
        # Another organisation's cloned voice must not be reachable by raw id.
        raise HTTPException(status_code=422, detail="invalid_provider_voice_id")
    record = save_voice(
        db,
        ctx,
        payload,
        source="library",
        status="ready",
        provider_voice_id=payload.provider_voice_id,
    )
    return Voice.model_validate(record)


@router.post("/voices/clone", response_model=Voice, status_code=status.HTTP_201_CREATED)
async def clone_voice(
    payload: VoiceCloneIn,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> Voice:
    if payload.consent is None or not payload.consent.confirmed:
        raise HTTPException(status_code=422, detail="voice_consent_required")
    if len(payload.sample_reference) > 1024 or not SAMPLE_REFERENCE_RE.match(
        payload.sample_reference
    ):
        raise HTTPException(status_code=422, detail="invalid_sample_reference")
    consent = payload.consent.model_dump()
    consent["recorded_at"] = utcnow().isoformat()
    record = save_voice(
        db,
        ctx,
        payload,
        # Consent summary only: never the sample URL.
        audit_metadata={
            "consent": {
                "method": payload.consent.method,
                "subject_name": payload.consent.subject_name,
            }
        },
        source="clone",
        status="pending",
        sample_reference=payload.sample_reference,
        consent=consent,
    )
    return Voice.model_validate(record)


@router.post(
    "/voices/design", response_model=Voice, status_code=status.HTTP_201_CREATED
)
async def design_voice(
    payload: VoiceDesignIn,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> Voice:
    record = save_voice(
        db,
        ctx,
        payload,
        source="design",
        status="pending",
        design_prompt=payload.design_prompt,
    )
    return Voice.model_validate(record)


@router.get("/voices", response_model=list[Voice])
async def list_voices(
    status: VoiceStatus | None = None,
    source: VoiceSource | None = None,
    include_deleted: bool = False,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> list[Voice]:
    query = scoped(db, ctx, VoiceRecord)
    # Tombstones are visible to admins only, and only when asked for.
    if status != "deleted" and not (include_deleted and _is_admin(ctx)):
        query = query.where(VoiceRecord.status != "deleted")
    elif status == "deleted" and not _is_admin(ctx):
        return []
    if status is not None:
        query = query.where(VoiceRecord.status == status)
    if source is not None:
        query = query.where(VoiceRecord.source == source)
    return [Voice.model_validate(item) for item in db.scalars(query).all()]


@router.get("/voices/{voice_id}", response_model=Voice)
async def get_voice(
    voice_id: UUID,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> Voice:
    # Tombstones answer admins (id, status, consent, deleted_at) and 404 for members.
    record = get_voice_or_404(voice_id, db, ctx, include_deleted=_is_admin(ctx))
    return Voice.model_validate(record)


@router.patch("/voices/{voice_id}", response_model=Voice)
async def update_voice(
    voice_id: UUID,
    payload: VoicePatch,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> Voice:
    record = get_voice_or_404(voice_id, db, ctx)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    for key, value in changes.items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return Voice.model_validate(record)


@router.post("/voices/{voice_id}/status", response_model=Voice)
async def update_voice_status(
    voice_id: UUID,
    payload: VoiceStatusIn,
    ctx: AuthContext = Depends(require_superadmin),
    db: Session = Depends(get_scoped_db),
) -> Voice:
    record = get_voice_or_404(voice_id, db, ctx, include_deleted=True)
    if record.source == "clone" and payload.provider_voice_id is not None:
        # Clone ids come from provisioning only; never from a caller-supplied value.
        raise HTTPException(status_code=422, detail={"code": "provider_voice_id_managed"})
    if payload.status not in TRANSITIONS.get(record.status, set()):
        raise HTTPException(status_code=409, detail="invalid_voice_status_transition")
    if record.status == "disabled" and not record.provider_voice_id:
        # A voice disabled before provisioning finished must go back through pending.
        raise HTTPException(status_code=409, detail="invalid_voice_status_transition")
    provider_voice_id = payload.provider_voice_id or record.provider_voice_id
    if payload.status == "ready" and not provider_voice_id:
        raise HTTPException(status_code=422, detail="provider_voice_id_required")
    previous_status = record.status
    record.status = payload.status
    record.provider_voice_id = provider_voice_id
    record.failure_reason = payload.failure_reason if payload.status == "failed" else None
    audit.record(
        db,
        ctx,
        "voice.status_changed",
        resource_type="voice",
        resource_id=record.id,
        metadata={"from": previous_status, "to": payload.status},
    )
    db.commit()
    db.refresh(record)
    return Voice.model_validate(record)


def _agents_using_voice(
    db: Session, ctx: AuthContext, voice_id: str, provider_voice_id: str | None = None
) -> list[str]:
    """Ids of live (not archived/disabled) agents in the org that select this voice.

    Agents select a voice either by profile (``voice_profile_id``) or by a raw
    ``voice_id`` equal to the clone's ``provider_voice_id``; both count as in use.
    """

    selects = AgentRecord.voice_profile_id == voice_id
    if provider_voice_id:
        selects = or_(selects, AgentRecord.voice_id == provider_voice_id)
    return list(
        db.scalars(
            scoped(db, ctx, AgentRecord).with_only_columns(AgentRecord.id).where(
                selects, AgentRecord.status != "disabled"
            )
        )
    )


async def _delete_at_provider(
    runtime_url: str, runtime_token: str, provider: str, provider_voice_id: str
) -> tuple[bool, bool]:
    """Ask Runtime to delete the provider voice. Returns ``(deleted, retryable)``.

    Runtime answers 204 for success (including a provider-side 404, so retries are
    idempotent); anything else is a failure.
    """

    url = (
        f"{runtime_url.rstrip('/')}/voices/"
        f"{quote(provider, safe='')}/{quote(provider_voice_id, safe='')}"
    )
    try:
        async with httpx.AsyncClient(timeout=PROVIDER_DELETE_TIMEOUT_SECONDS) as client:
            response = await client.delete(url, headers={"X-Runtime-Token": runtime_token})
    except httpx.HTTPError:
        return False, True
    if response.status_code in {200, 202, 204}:
        return True, False
    retryable = response.status_code >= 500
    if response.status_code in {502, 503}:
        try:
            body = response.json()
            if isinstance(body, dict) and isinstance(body.get("retryable"), bool):
                retryable = body["retryable"] or response.status_code == 503
        except ValueError:
            pass
    return False, retryable


@router.delete("/voices/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice(
    voice_id: UUID,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> Response:
    """Hard delete: provider voice + stored audio are removed, consent is retained.

    The row stays as a tombstone. A failed provider delete leaves the voice in
    ``deleting`` (unusable) and the call can simply be repeated.
    """

    record = get_voice_or_404(voice_id, db, ctx, include_deleted=True)
    if record.status == "deleted":
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    agent_ids = _agents_using_voice(db, ctx, record.id, record.provider_voice_id)
    if agent_ids:
        raise HTTPException(
            status_code=409, detail={"code": "voice_in_use", "agent_ids": agent_ids}
        )

    now = utcnow()
    provider_deleted = False
    if record.provider_voice_id and record.source in {"clone", "design"}:
        runtime_url = os.getenv("RUNTIME_API_URL")
        runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
        if not runtime_url or not runtime_token:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "runtime_unavailable",
                    "message": "Runtime is not configured; the provider voice was not deleted.",
                },
            )
        provider = _voice_provider(db, ctx, record)
        provider_deleted, retryable = await _delete_at_provider(
            runtime_url,
            runtime_token,
            (provider.provider if provider else None) or "cartesia",
            record.provider_voice_id,
        )
        if not provider_deleted:
            # Unusable from now on; retrying DELETE is safe and idempotent.
            record.status = "deleting"
            audit.record(
                db,
                ctx,
                "voice.delete_failed",
                resource_type="voice",
                resource_id=record.id,
                metadata={"source": record.source, "retryable": retryable},
            )
            db.commit()
            raise HTTPException(
                status_code=502,
                detail={"code": "voice_provider_delete_failed", "retryable": retryable},
            )

    sample_deleted = False
    if record.sample_reference:
        sample_deleted = storage.delete(record.sample_reference) == "deleted"
    storage.delete_prefix(f"voices/{record.id}/previews/")

    record.sample_reference = None
    record.design_prompt = None
    record.provider_voice_id = None
    record.style = {}
    record.failure_reason = None
    record.status = "deleted"
    record.deleted_at = now
    record.deleted_by = (ctx.key_id or ctx.principal)[:64]
    if provider_deleted:
        record.provider_deleted_at = now
    if sample_deleted:
        record.sample_deleted_at = now
    audit.record(
        db,
        ctx,
        "voice.deleted",
        resource_type="voice",
        resource_id=record.id,
        metadata={
            "source": record.source,
            "provider_deleted": provider_deleted,
            "sample_deleted": sample_deleted,
        },
    )
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


PROVIDER_DELETE_TIMEOUT_SECONDS = 20.0


def _voice_provider(
    db: Session, ctx: AuthContext, record: VoiceRecord
) -> ProviderConfig | None:
    if not record.tts_provider_id:
        return None
    return db.scalars(
        scoped(db, ctx, ProviderConfig).where(ProviderConfig.id == record.tts_provider_id)
    ).first()


class VoicePreviewIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)


@router.post("/voices/{voice_id}/preview")
async def preview_voice(
    voice_id: UUID,
    payload: VoicePreviewIn,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> Response:
    record = get_voice_or_404(voice_id, db, ctx)
    if record.status != "ready" or not record.provider_voice_id:
        raise HTTPException(status_code=409, detail="voice_not_ready")
    provider = _voice_provider(db, ctx, record)
    runtime_url = os.getenv("RUNTIME_API_URL")
    if not runtime_url:
        raise HTTPException(status_code=503, detail="runtime_not_configured")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if not runtime_token:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")

    body: dict[str, Any] = {
        "provider": (provider.provider if provider else None) or "cartesia",
        "provider_voice_id": record.provider_voice_id,
        "text": payload.text,
    }
    configuration = (provider.configuration if provider else None) or {}
    for key in ("model", "language"):
        value = configuration.get(key)
        if isinstance(value, str) and value:
            body[key] = value
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{runtime_url.rstrip('/')}/tts/preview",
                headers={"X-Runtime-Token": runtime_token},
                json=body,
            )
            response.raise_for_status()
            audio = response.content
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="voice_preview_failed") from exc
    if not audio:
        raise HTTPException(status_code=502, detail="voice_preview_failed")
    audit.record(
        db,
        ctx,
        "voice.previewed",
        resource_type="voice",
        resource_id=record.id,
        metadata={"text_length": len(payload.text)},
    )
    db.commit()
    return Response(content=audio, media_type="audio/wav")


PROVISION_TIMEOUT_SECONDS = 60.0
# A voice stuck in ``processing`` this long (the request died mid-flight) may be retried.
PROVISION_STALE_AFTER = timedelta(minutes=5)
CLONE_LANGUAGE_RE = re.compile(r"^[a-z]{2}$")
# Runtime error codes that are safe to persist verbatim as failure_reason.
_RUNTIME_CLONE_ERRORS = {
    "unsupported_sample_url",
    "sample_unavailable",
    "clone_provider_error",
    "tts_not_configured",
    "unsupported_tts_provider",
    "invalid_runtime_token",
}


def _provision_failure_reason(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        if isinstance(detail, str) and detail in _RUNTIME_CLONE_ERRORS:
            return detail
        return f"runtime_error_{exc.response.status_code}"
    if isinstance(exc, httpx.HTTPError):
        return "runtime_unreachable"
    if isinstance(exc, (ValueError, AttributeError, TypeError)):
        return "runtime_invalid_response"
    return "runtime_error"


def _clone_language(languages: list[str] | None) -> str | None:
    """Primary subtag of the first language ("en-US" -> "en"), or None if not ISO 639-1."""

    candidate = ((languages or ["en"])[0] or "").strip().split("-")[0].lower()
    return candidate if CLONE_LANGUAGE_RE.match(candidate) else None


def _is_stale_processing(record: VoiceRecord) -> bool:
    if record.status != "processing" or record.updated_at is None:
        return False
    updated_at = record.updated_at
    if updated_at.tzinfo is None:  # SQLite drops the zone; values are stored as UTC
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - updated_at > PROVISION_STALE_AFTER


@router.post("/voices/{voice_id}/provision", response_model=Voice)
async def provision_voice(
    voice_id: UUID,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> Voice:
    """Clone a consented sample at the TTS provider via Runtime (Platform never touches audio)."""

    record = get_voice_or_404(voice_id, db, ctx)
    if record.source != "clone" or not (
        record.status in {"pending", "failed"} or _is_stale_processing(record)
    ):
        raise HTTPException(status_code=409, detail="invalid_voice_status_transition")
    consent = record.consent
    if not isinstance(consent, dict) or consent.get("confirmed") is not True:
        raise HTTPException(status_code=409, detail="voice_consent_required")
    sample_reference = record.sample_reference or ""
    if not sample_reference.startswith("https://"):
        raise HTTPException(status_code=422, detail="unsupported_sample_url")
    runtime_url = os.getenv("RUNTIME_API_URL")
    if not runtime_url:
        raise HTTPException(status_code=503, detail="runtime_not_configured")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if not runtime_token:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")
    provider = _voice_provider(db, ctx, record)
    language = _clone_language(record.languages)
    if language is None:
        raise HTTPException(status_code=422, detail="invalid_voice_language")
    body = {
        "provider": (provider.provider if provider else None) or "cartesia",
        "name": record.name,
        "language": language,
        "sample_url": sample_reference,
    }

    record.status = "processing"
    record.failure_reason = None
    db.commit()

    failure_reason: str | None = None
    provider_voice_id: str | None = None
    try:
        async with httpx.AsyncClient(timeout=PROVISION_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{runtime_url.rstrip('/')}/voices/clone",
                headers={"X-Runtime-Token": runtime_token},
                json=body,
            )
            response.raise_for_status()
            candidate = response.json().get("provider_voice_id")
    except Exception as exc:  # never leave the voice stuck in ``processing``
        failure_reason = _provision_failure_reason(exc)
        if failure_reason == "runtime_error":
            _logger.exception("unexpected error provisioning voice %s", voice_id)
    else:
        if isinstance(candidate, str) and 0 < len(candidate) <= 200:
            provider_voice_id = candidate
        else:
            failure_reason = "runtime_invalid_response"

    if provider_voice_id is None:
        record.status = "failed"
        record.failure_reason = failure_reason
        audit.record(
            db,
            ctx,
            "voice.provision_failed",
            resource_type="voice",
            resource_id=record.id,
            metadata={"source": record.source, "failure_reason": failure_reason},
        )
        db.commit()
        raise HTTPException(status_code=502, detail="voice_provisioning_failed")
    record.status = "ready"
    record.provider_voice_id = provider_voice_id
    record.failure_reason = None
    audit.record(
        db,
        ctx,
        "voice.provisioned",
        resource_type="voice",
        resource_id=record.id,
        metadata={"source": record.source},
    )
    db.commit()
    db.refresh(record)
    return Voice.model_validate(record)
