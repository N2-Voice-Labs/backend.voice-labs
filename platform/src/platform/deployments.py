"""Deployment API: where an agent is reachable (web, SIP phone number, or API)."""

import asyncio
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from . import sip
from .db import get_db
from .models import Agent as AgentRecord
from .models import Deployment as DeploymentRecord
from .validation import validate_deployment_configuration, validate_e164

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
        return validate_deployment_configuration(value)


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
        return validate_deployment_configuration(value)


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


class BridgeCallIn(BaseModel):
    from_number: str
    to_number: str

    @field_validator("from_number", "to_number")
    @classmethod
    def check_number(cls, value: str) -> str:
        return validate_e164(value)


class BridgeParticipant(BaseModel):
    role: Literal["from", "to"]
    phone_number: str
    identity: str
    participant_id: str


class BridgeCall(BaseModel):
    session_id: str
    conversation_id: str
    room_name: str
    participants: list[BridgeParticipant]


@router.post(
    "/deployments/{deployment_id}/bridge-call",
    response_model=BridgeCall,
    status_code=status.HTTP_201_CREATED,
)
async def bridge_call(
    deployment_id: UUID, payload: BridgeCallIn, db: Session = Depends(get_db)
) -> BridgeCall:
    """Phone interpreter: dial both parties into one room with the interpreter agent.

    Both legs join the same LiveKit room as the interpreter runtime, which hears them
    and speaks translations into the room. The session is created without TR-020
    participants (each leg's SIP identity is only known once the session exists), so
    nobody is mapped to a language; the interpreter's own detection of the spoken
    language must pick the direction. That needs the cascaded engine with ``auto``
    direction: ``one_way`` or the one-direction ``realtime`` engine would translate
    only one party, so they are rejected with ``bridge_requires_two_way_interpreter``.
    The ``from`` party is dialed first, then ``to``; if ``to`` does not connect, the
    session is ended and ``from`` is hung up.
    """

    record = get_deployment_or_404(deployment_id, db)
    agent = db.get(AgentRecord, record.agent_id)
    if record.channel != "sip":
        raise HTTPException(status_code=422, detail="deployment_not_sip")
    if agent is None or agent.mode != "interpreter":
        raise HTTPException(status_code=422, detail="agent_not_interpreter")
    interpreter = agent.interpreter or {}
    if (
        interpreter.get("engine", "cascaded") != "cascaded"
        or interpreter.get("direction", "auto") != "auto"
    ):
        raise HTTPException(status_code=422, detail="bridge_requires_two_way_interpreter")
    if record.status != "active":
        raise HTTPException(status_code=409, detail="deployment_not_active")
    if payload.from_number == payload.to_number:
        raise HTTPException(status_code=422, detail="bridge_numbers_must_differ")
    if not sip.sip_configured():
        raise HTTPException(status_code=409, detail="sip_not_configured")
    try:
        session = await sip.create_voice_session(record.agent_id, record.id)
    except sip.SessionCreateError as exc:
        raise HTTPException(status_code=502, detail="session_create_failed") from exc
    session_id = str(session["session_id"])
    room_name = str(session["room_name"])
    participants: list[BridgeParticipant] = []
    for role, number in (("from", payload.from_number), ("to", payload.to_number)):
        identity = f"{sip.SIP_IDENTITY_PREFIX}{role}-{session_id}"
        try:
            participant_id = await sip.dial(number, room_name, identity)
        except sip.SipError as exc:
            await asyncio.gather(
                sip.end_voice_session(session_id),
                *(sip.hangup(room_name, item.identity) for item in participants),
            )
            raise HTTPException(
                status_code=502,
                detail="sip_dial_failed",
            ) from exc
        participants.append(
            BridgeParticipant(
                role=role,
                phone_number=number,
                identity=identity,
                participant_id=participant_id,
            )
        )
    return BridgeCall(
        session_id=session_id,
        conversation_id=str(session["conversation_id"]),
        room_name=room_name,
        participants=participants,
    )
