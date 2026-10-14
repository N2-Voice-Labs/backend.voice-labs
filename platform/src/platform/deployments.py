"""Deployment API: where an agent is reachable (web, SIP phone number, or API)."""

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from .db import get_db
from .models import Agent as AgentRecord
from .models import Deployment as DeploymentRecord
from .validation import reject_secret_configuration, validate_e164

router = APIRouter(tags=["deployments"])

Channel = Literal["web", "sip", "api"]
DeploymentStatus = Literal["active", "paused", "archived"]


class DeploymentIn(BaseModel):
    agent_id: UUID
    name: str = Field(min_length=1, max_length=200)
    channel: Channel
    phone_number: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)

    @field_validator("phone_number")
    @classmethod
    def check_phone_number(cls, value: str | None) -> str | None:
        return value if value is None else validate_e164(value)

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class Deployment(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    agent_id: UUID
    name: str
    channel: Channel
    status: DeploymentStatus
    phone_number: str | None
    configuration: dict[str, Any]
    created_at: Any
    updated_at: Any


class DeploymentPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    status: DeploymentStatus | None = None
    phone_number: str | None = None
    configuration: dict[str, Any] | None = None

    @field_validator("name", "status", "configuration", mode="before")
    @classmethod
    def reject_null_for_required_fields(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("field cannot be null")
        return value

    @field_validator("phone_number")
    @classmethod
    def check_phone_number(cls, value: str | None) -> str | None:
        return value if value is None else validate_e164(value)

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


def check_phone_invariant(channel: str, phone_number: str | None) -> None:
    """SIP deployments need a phone number; other channels must not have one."""
    if channel == "sip" and phone_number is None:
        raise HTTPException(status_code=422, detail="phone_number_required")
    if channel != "sip" and phone_number is not None:
        raise HTTPException(status_code=422, detail="phone_number_only_for_sip")


def get_deployment_or_404(deployment_id: UUID, db: Session) -> DeploymentRecord:
    record = db.get(DeploymentRecord, str(deployment_id))
    if record is None:
        raise HTTPException(status_code=404, detail="deployment_not_found")
    return record


@router.post(
    "/deployments", response_model=Deployment, status_code=status.HTTP_201_CREATED
)
async def create_deployment(
    payload: DeploymentIn, db: Session = Depends(get_db)
) -> Deployment:
    agent = db.get(AgentRecord, str(payload.agent_id))
    if agent is None or agent.status == "disabled":
        raise HTTPException(status_code=422, detail="invalid_agent_id")
    check_phone_invariant(payload.channel, payload.phone_number)
    record = DeploymentRecord(
        **{**payload.model_dump(), "agent_id": str(payload.agent_id)}
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return Deployment.model_validate(record)


@router.get("/deployments", response_model=list[Deployment])
async def list_deployments(
    agent_id: UUID | None = None,
    channel: Channel | None = None,
    status: DeploymentStatus | None = None,
    db: Session = Depends(get_db),
) -> list[Deployment]:
    query = db.query(DeploymentRecord)
    if agent_id is not None:
        query = query.filter(DeploymentRecord.agent_id == str(agent_id))
    if channel is not None:
        query = query.filter(DeploymentRecord.channel == channel)
    if status is not None:
        query = query.filter(DeploymentRecord.status == status)
    return [
        Deployment.model_validate(item)
        for item in query.order_by(DeploymentRecord.created_at)
    ]


@router.get("/deployments/{deployment_id}", response_model=Deployment)
async def get_deployment(
    deployment_id: UUID, db: Session = Depends(get_db)
) -> Deployment:
    return Deployment.model_validate(get_deployment_or_404(deployment_id, db))


@router.patch("/deployments/{deployment_id}", response_model=Deployment)
async def update_deployment(
    deployment_id: UUID, payload: DeploymentPatch, db: Session = Depends(get_db)
) -> Deployment:
    record = get_deployment_or_404(deployment_id, db)
    if record.status == "archived":
        raise HTTPException(status_code=409, detail="deployment_archived")
    changes = payload.model_dump(exclude_unset=True)
    check_phone_invariant(
        record.channel, changes.get("phone_number", record.phone_number)
    )
    for key, value in changes.items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return Deployment.model_validate(record)


@router.delete(
    "/deployments/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def archive_deployment(
    deployment_id: UUID, db: Session = Depends(get_db)
) -> None:
    record = get_deployment_or_404(deployment_id, db)
    record.status = "archived"
    db.commit()
