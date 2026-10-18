"""Voice API: library, cloned, and designed voices with a validated lifecycle."""

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
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
from sqlalchemy.orm import Session

from .db import get_db
from .models import ProviderConfig, Voice as VoiceRecord, utcnow

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
VoiceStatus = Literal["pending", "processing", "ready", "failed", "disabled"]
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
    created_at: Any
    updated_at: Any


def validate_tts_provider(provider_id: UUID | None, db: Session) -> str | None:
    if provider_id is None:
        return None
    provider = db.get(ProviderConfig, str(provider_id))
    if provider is None or provider.kind != "tts" or not provider.enabled:
        raise HTTPException(status_code=422, detail="invalid_tts_provider_id")
    return provider.id


def get_voice_or_404(voice_id: UUID, db: Session) -> VoiceRecord:
    record = db.get(VoiceRecord, str(voice_id))
    if record is None:
        raise HTTPException(status_code=404, detail="voice_not_found")
    return record


def save_voice(db: Session, payload: VoiceBase, **fields: Any) -> VoiceRecord:
    record = VoiceRecord(
        name=payload.name,
        tts_provider_id=validate_tts_provider(payload.tts_provider_id, db),
        languages=payload.languages,
        style=payload.style.model_dump(exclude_none=True),
        **fields,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post("/voices", response_model=Voice, status_code=status.HTTP_201_CREATED)
async def create_voice(payload: VoiceIn, db: Session = Depends(get_db)) -> Voice:
    record = save_voice(
        db,
        payload,
        source="library",
        status="ready",
        provider_voice_id=payload.provider_voice_id,
    )
    return Voice.model_validate(record)


@router.post("/voices/clone", response_model=Voice, status_code=status.HTTP_201_CREATED)
async def clone_voice(payload: VoiceCloneIn, db: Session = Depends(get_db)) -> Voice:
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
        payload,
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
    payload: VoiceDesignIn, db: Session = Depends(get_db)
) -> Voice:
    record = save_voice(
        db,
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
    db: Session = Depends(get_db),
) -> list[Voice]:
    query = db.query(VoiceRecord)
    if status is not None:
        query = query.filter(VoiceRecord.status == status)
    if source is not None:
        query = query.filter(VoiceRecord.source == source)
    return [Voice.model_validate(item) for item in query.all()]


@router.get("/voices/{voice_id}", response_model=Voice)
async def get_voice(voice_id: UUID, db: Session = Depends(get_db)) -> Voice:
    return Voice.model_validate(get_voice_or_404(voice_id, db))


@router.patch("/voices/{voice_id}", response_model=Voice)
async def update_voice(
    voice_id: UUID, payload: VoicePatch, db: Session = Depends(get_db)
) -> Voice:
    record = get_voice_or_404(voice_id, db)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    for key, value in changes.items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return Voice.model_validate(record)


@router.post("/voices/{voice_id}/status", response_model=Voice)
async def update_voice_status(
    voice_id: UUID, payload: VoiceStatusIn, db: Session = Depends(get_db)
) -> Voice:
    record = get_voice_or_404(voice_id, db)
    if payload.status not in TRANSITIONS.get(record.status, set()):
        raise HTTPException(status_code=409, detail="invalid_voice_status_transition")
    if record.status == "disabled" and not record.provider_voice_id:
        # A voice disabled before provisioning finished must go back through pending.
        raise HTTPException(status_code=409, detail="invalid_voice_status_transition")
    provider_voice_id = payload.provider_voice_id or record.provider_voice_id
    if payload.status == "ready" and not provider_voice_id:
        raise HTTPException(status_code=422, detail="provider_voice_id_required")
    record.status = payload.status
    record.provider_voice_id = provider_voice_id
    record.failure_reason = payload.failure_reason if payload.status == "failed" else None
    db.commit()
    db.refresh(record)
    return Voice.model_validate(record)


@router.delete("/voices/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice(voice_id: UUID, db: Session = Depends(get_db)) -> None:
    record = get_voice_or_404(voice_id, db)
    record.status = "disabled"
    db.commit()


class VoicePreviewIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)


@router.post("/voices/{voice_id}/preview")
async def preview_voice(
    voice_id: UUID, payload: VoicePreviewIn, db: Session = Depends(get_db)
) -> Response:
    record = get_voice_or_404(voice_id, db)
    if record.status != "ready" or not record.provider_voice_id:
        raise HTTPException(status_code=409, detail="voice_not_ready")
    provider = (
        db.get(ProviderConfig, record.tts_provider_id)
        if record.tts_provider_id
        else None
    )
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
async def provision_voice(voice_id: UUID, db: Session = Depends(get_db)) -> Voice:
    """Clone a consented sample at the TTS provider via Runtime (Platform never touches audio)."""

    record = get_voice_or_404(voice_id, db)
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
    provider = (
        db.get(ProviderConfig, record.tts_provider_id)
        if record.tts_provider_id
        else None
    )
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
        db.commit()
        raise HTTPException(status_code=502, detail="voice_provisioning_failed")
    record.status = "ready"
    record.provider_voice_id = provider_voice_id
    record.failure_reason = None
    db.commit()
    db.refresh(record)
    return Voice.model_validate(record)
