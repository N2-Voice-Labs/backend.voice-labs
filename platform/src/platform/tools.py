"""Tool API."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .context import AuthContext, require
from .models import Agent as AgentRecord
from .models import AgentTool
from .models import Tool as ToolRecord
from .tenancy import get_owned, get_public_agent, get_scoped_db, scoped
from .validation import reject_secret_configuration

router = APIRouter(tags=["tools"])
NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
TOOL_TYPE_PATTERN = "^(http|builtin)$"


def validate_input_schema(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("type") != "object":
        raise ValueError("input_schema must be a JSON Schema object")
    if "properties" in value and not isinstance(value["properties"], dict):
        raise ValueError("input_schema properties must be an object")
    return value


class ToolIn(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str = Field(min_length=1, max_length=2000)
    tool_type: str = Field(default="http", pattern=TOOL_TYPE_PATTERN)
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("input_schema")
    @classmethod
    def check_input_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_input_schema(value)

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class ToolPatch(BaseModel):
    name: str | None = Field(default=None, pattern=NAME_PATTERN)
    description: str | None = Field(default=None, min_length=1, max_length=2000)
    tool_type: str | None = Field(default=None, pattern=TOOL_TYPE_PATTERN)
    input_schema: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    enabled: bool | None = None

    @field_validator("*", mode="before")
    @classmethod
    def reject_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("field cannot be null")
        return value

    @field_validator("input_schema")
    @classmethod
    def check_input_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_input_schema(value)

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class Tool(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    description: str
    tool_type: str
    input_schema: dict[str, Any]
    configuration: dict[str, Any]
    enabled: bool


class AgentToolsIn(BaseModel):
    tool_ids: list[UUID]


def get_tool_or_404(tool_id: UUID, db: Session, ctx: AuthContext) -> ToolRecord:
    return get_owned(
        db, ctx, ToolRecord, str(tool_id), not_found_detail="tool_not_found"
    )


def commit_or_conflict(db: Session) -> None:
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="tool_name_conflict")


@router.post("/tools", response_model=Tool, status_code=status.HTTP_201_CREATED)
async def create_tool(
    payload: ToolIn,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> ToolRecord:
    record = ToolRecord(**payload.model_dump())
    db.add(record)
    commit_or_conflict(db)
    db.refresh(record)
    return record


@router.get("/tools", response_model=list[Tool])
async def list_tools(
    enabled: bool | None = None,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> list[ToolRecord]:
    query = db.query(ToolRecord)
    if enabled is not None:
        query = query.filter(ToolRecord.enabled == enabled)
    return query.order_by(ToolRecord.name).all()


@router.get("/tools/{tool_id}", response_model=Tool)
async def get_tool(
    tool_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> ToolRecord:
    return get_tool_or_404(tool_id, db, ctx)


@router.patch("/tools/{tool_id}", response_model=Tool)
async def patch_tool(
    tool_id: UUID,
    payload: ToolPatch,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> ToolRecord:
    record = get_tool_or_404(tool_id, db, ctx)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(record, key, value)
    commit_or_conflict(db)
    db.refresh(record)
    return record


@router.delete("/tools/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tool(
    tool_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> Response:
    get_tool_or_404(tool_id, db, ctx).enabled = False
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_agent_or_404(agent_id: UUID, db: Session, ctx: AuthContext) -> AgentRecord:
    return get_public_agent(db, ctx, agent_id)


@router.put("/agents/{agent_id}/tools", response_model=list[Tool])
async def set_agent_tools(
    agent_id: UUID,
    payload: AgentToolsIn,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> list[ToolRecord]:
    get_agent_or_404(agent_id, db, ctx)
    ids = list(dict.fromkeys(str(tool_id) for tool_id in payload.tool_ids))
    tools: list[ToolRecord] = []
    if ids:
        # Org-scoped: another organisation's tool looks exactly like a missing one (422).
        tools = list(
            db.scalars(scoped(db, ctx, ToolRecord).where(ToolRecord.id.in_(ids)))
        )
    if len(tools) != len(ids) or not all(tool.enabled for tool in tools):
        raise HTTPException(status_code=422, detail="invalid_tool_id")
    db.query(AgentTool).filter(AgentTool.agent_id == str(agent_id)).delete()
    db.add_all(AgentTool(agent_id=str(agent_id), tool_id=tool.id) for tool in tools)
    db.commit()
    return sorted(tools, key=lambda tool: tool.name)


@router.get("/agents/{agent_id}/tools", response_model=list[Tool])
async def list_agent_tools(
    agent_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> list[ToolRecord]:
    get_agent_or_404(agent_id, db, ctx)
    return (
        db.query(ToolRecord)
        .join(AgentTool, AgentTool.tool_id == ToolRecord.id)
        .filter(AgentTool.agent_id == str(agent_id))
        .order_by(ToolRecord.name)
        .all()
    )
