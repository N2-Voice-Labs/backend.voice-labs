"""Voice API: library, cloned, and designed voices with a validated lifecycle."""

import re
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
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
