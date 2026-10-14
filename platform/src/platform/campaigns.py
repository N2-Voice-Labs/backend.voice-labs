"""Campaign API.

Campaigns and their targets are durable records only. There is no dialer or call
queue: changing a campaign to ``running`` places no calls and moves no target out of
``pending``; a future worker is expected to drive that state.
"""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import get_db
from .models import Agent as AgentRecord
from .models import Campaign as CampaignRecord
from .models import CampaignTarget as TargetRecord
from .models import Deployment as DeploymentRecord
from .validation import reject_secret_configuration, validate_e164

router = APIRouter(tags=["campaigns"])

CampaignStatus = Literal[
    "draft", "scheduled", "running", "paused", "completed", "cancelled"
]
TRANSITIONS: dict[str, set[str]] = {
    "draft": {"scheduled", "cancelled"},
    "scheduled": {"draft", "running", "cancelled"},
    "running": {"paused", "completed", "cancelled"},
    "paused": {"running", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}
EDITABLE = {"draft", "scheduled"}
TARGETS_EDITABLE = {"draft", "scheduled", "paused"}


def check_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise HTTPException(status_code=422, detail="invalid_timezone") from None
    return value


def as_utc(value: datetime | None) -> datetime | None:
    """Treat naive datetimes (including those read back from SQLite) as UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def check_schedule(start: datetime | None, end: datetime | None) -> None:
    if start is not None and end is not None and end <= start:
        raise HTTPException(status_code=422, detail="invalid_schedule")


class CampaignIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    agent_id: UUID
    deployment_id: UUID
    schedule_start: datetime | None = None
    schedule_end: datetime | None = None
    timezone: str = Field(default="UTC", max_length=64)
    max_attempts: int = Field(default=1, ge=1, le=10)
    configuration: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schedule_start", "schedule_end")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        return as_utc(value)

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class Campaign(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    agent_id: UUID
    deployment_id: UUID
    status: CampaignStatus
    schedule_start: datetime | None
    schedule_end: datetime | None
    timezone: str
    max_attempts: int
    configuration: dict[str, Any]
    created_at: Any
    updated_at: Any


class CampaignDetail(Campaign):
    target_counts: dict[str, int]


class CampaignPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    schedule_start: datetime | None = None
    schedule_end: datetime | None = None
    timezone: str | None = Field(default=None, max_length=64)
    max_attempts: int | None = Field(default=None, ge=1, le=10)
    configuration: dict[str, Any] | None = None

    @field_validator("name", "timezone", "max_attempts", "configuration", mode="before")
    @classmethod
    def reject_null_for_required_fields(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("field cannot be null")
        return value

    @field_validator("schedule_start", "schedule_end")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        return as_utc(value)

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class TargetIn(BaseModel):
    phone_number: str
    name: str = Field(default="", max_length=200)
    variables: dict[str, Any] = Field(default_factory=dict)

    @field_validator("phone_number")
    @classmethod
    def check_phone_number(cls, value: str) -> str:
        return validate_e164(value)


class TargetsIn(BaseModel):
    targets: list[TargetIn] = Field(min_length=1, max_length=1000)


class TargetsAdded(BaseModel):
    added: int
    skipped_duplicates: int


class Target(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    campaign_id: UUID
    phone_number: str
    name: str
    variables: dict[str, Any]
    status: str
    attempts: int
    conversation_id: UUID | None
    created_at: Any
    updated_at: Any


class StatusIn(BaseModel):
    status: CampaignStatus


def get_campaign_or_404(campaign_id: UUID, db: Session) -> CampaignRecord:
    record = db.get(CampaignRecord, str(campaign_id))
    if record is None:
        raise HTTPException(status_code=404, detail="campaign_not_found")
    return record


def campaign_detail(record: CampaignRecord, db: Session) -> CampaignDetail:
    counts = dict(
        db.query(TargetRecord.status, func.count())
        .filter(TargetRecord.campaign_id == record.id)
        .group_by(TargetRecord.status)
        .all()
    )
    return CampaignDetail(
        **Campaign.model_validate(record).model_dump(), target_counts=counts
    )


@router.post(
    "/campaigns", response_model=CampaignDetail, status_code=status.HTTP_201_CREATED
)
async def create_campaign(
    payload: CampaignIn, db: Session = Depends(get_db)
) -> CampaignDetail:
    check_timezone(payload.timezone)
    check_schedule(payload.schedule_start, payload.schedule_end)
    if db.get(AgentRecord, str(payload.agent_id)) is None:
        raise HTTPException(status_code=422, detail="invalid_agent_id")
    deployment = db.get(DeploymentRecord, str(payload.deployment_id))
    if deployment is None:
        raise HTTPException(status_code=422, detail="invalid_deployment_id")
    if deployment.agent_id != str(payload.agent_id):
        raise HTTPException(status_code=409, detail="deployment_agent_mismatch")
    if deployment.channel != "sip":
        raise HTTPException(status_code=422, detail="campaign_requires_sip_deployment")
    record = CampaignRecord(
        **{
            **payload.model_dump(),
            "agent_id": str(payload.agent_id),
            "deployment_id": str(payload.deployment_id),
        }
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return campaign_detail(record, db)


@router.get("/campaigns", response_model=list[Campaign])
async def list_campaigns(
    agent_id: UUID | None = None,
    status: CampaignStatus | None = None,
    db: Session = Depends(get_db),
) -> list[Campaign]:
    query = db.query(CampaignRecord)
    if agent_id is not None:
        query = query.filter(CampaignRecord.agent_id == str(agent_id))
    if status is not None:
        query = query.filter(CampaignRecord.status == status)
    return [
        Campaign.model_validate(item) for item in query.order_by(CampaignRecord.created_at)
    ]


@router.get("/campaigns/{campaign_id}", response_model=CampaignDetail)
async def get_campaign(campaign_id: UUID, db: Session = Depends(get_db)) -> CampaignDetail:
    return campaign_detail(get_campaign_or_404(campaign_id, db), db)


@router.patch("/campaigns/{campaign_id}", response_model=CampaignDetail)
async def update_campaign(
    campaign_id: UUID, payload: CampaignPatch, db: Session = Depends(get_db)
) -> CampaignDetail:
    record = get_campaign_or_404(campaign_id, db)
    if record.status not in EDITABLE:
        raise HTTPException(status_code=409, detail="campaign_not_editable")
    changes = payload.model_dump(exclude_unset=True)
    if "timezone" in changes:
        check_timezone(changes["timezone"])
    check_schedule(
        as_utc(changes.get("schedule_start", record.schedule_start)),
        as_utc(changes.get("schedule_end", record.schedule_end)),
    )
    for key, value in changes.items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return campaign_detail(record, db)


@router.post("/campaigns/{campaign_id}/targets", response_model=TargetsAdded)
async def add_targets(
    campaign_id: UUID, payload: TargetsIn, db: Session = Depends(get_db)
) -> TargetsAdded:
    record = get_campaign_or_404(campaign_id, db)
    if record.status not in TARGETS_EDITABLE:
        raise HTTPException(status_code=409, detail="campaign_not_editable")
    unique: dict[str, TargetIn] = {}
    for item in payload.targets:
        unique.setdefault(item.phone_number, item)
    for _ in range(2):  # retry once if a concurrent request adds the same numbers
        existing = {
            phone
            for (phone,) in db.query(TargetRecord.phone_number).filter(
                TargetRecord.campaign_id == record.id,
                TargetRecord.phone_number.in_(unique),
            )
        }
        new = [
            TargetRecord(campaign_id=record.id, **item.model_dump())
            for phone, item in unique.items()
            if phone not in existing
        ]
        db.add_all(new)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            continue
        return TargetsAdded(
            added=len(new), skipped_duplicates=len(payload.targets) - len(new)
        )
    raise HTTPException(status_code=409, detail="target_conflict")


@router.get("/campaigns/{campaign_id}/targets", response_model=list[Target])
async def list_targets(
    campaign_id: UUID,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[Target]:
    record = get_campaign_or_404(campaign_id, db)
    query = db.query(TargetRecord).filter(TargetRecord.campaign_id == record.id)
    if status is not None:
        query = query.filter(TargetRecord.status == status)
    rows = query.order_by(TargetRecord.created_at, TargetRecord.id)
    return [Target.model_validate(item) for item in rows.offset(offset).limit(limit)]


@router.post("/campaigns/{campaign_id}/status", response_model=CampaignDetail)
async def change_campaign_status(
    campaign_id: UUID, payload: StatusIn, db: Session = Depends(get_db)
) -> CampaignDetail:
    record = get_campaign_or_404(campaign_id, db)
    if payload.status not in TRANSITIONS[record.status]:
        raise HTTPException(
            status_code=409, detail="invalid_campaign_status_transition"
        )
    if payload.status in {"scheduled", "running"}:
        deployment = db.get(DeploymentRecord, record.deployment_id)
        if deployment is None or deployment.status != "active":
            raise HTTPException(status_code=409, detail="deployment_not_active")
        has_targets = (
            db.query(TargetRecord.id)
            .filter(TargetRecord.campaign_id == record.id)
            .first()
        )
        if has_targets is None:
            raise HTTPException(status_code=409, detail="campaign_has_no_targets")
    record.status = payload.status
    db.commit()
    db.refresh(record)
    return campaign_detail(record, db)
