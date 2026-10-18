"""Durable control-plane entities."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Agent(Base):
    __tablename__ = "agents"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    enabled_languages: Mapped[list[str]] = mapped_column(
        JSON, default=lambda: ["en", "ru"]
    )
    stt_provider_id: Mapped[str | None] = mapped_column(
        ForeignKey("provider_configs.id"), nullable=True
    )
    llm_provider_id: Mapped[str | None] = mapped_column(
        ForeignKey("provider_configs.id"), nullable=True
    )
    tts_provider_id: Mapped[str | None] = mapped_column(
        ForeignKey("provider_configs.id"), nullable=True
    )
    voice_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    voice_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("voices.id"), nullable=True
    )
    mode: Mapped[str] = mapped_column(String(20), default="agent", server_default="agent")
    interpreter: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Agent + translator (TR-040): {working_language, caller_languages}.
    translation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="enabled")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ProviderConfig(Base):
    __tablename__ = "provider_configs"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    kind: Mapped[str] = mapped_column(String(20))
    provider: Mapped[str] = mapped_column(String(100))
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    secret_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_agent_started_at", "agent_id", "started_at"),
    )
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    status: Mapped[str] = mapped_column(String(30), default="active")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("deployments.id"), nullable=True
    )
    channel: Mapped[str] = mapped_column(String(20), default="web")
    outcome: Mapped[str | None] = mapped_column(String(40), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent: Mapped[str | None] = mapped_column(String(200), nullable=True)
    actions: Mapped[list] = mapped_column(JSON, default=list)
    outcome_recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index(
            "ix_messages_conversation_created_at", "conversation_id", "created_at", "id"
        ),
    )
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    primary_language: Mapped[str | None] = mapped_column(String(20), nullable=True)
    detected_languages: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class VoiceSession(Base):
    __tablename__ = "voice_sessions"
    __table_args__ = (
        Index(
            "ix_voice_sessions_conversation_started_at", "conversation_id", "started_at"
        ),
    )
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    status: Mapped[str] = mapped_column(String(30), default="created")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    transport: Mapped[str] = mapped_column(String(50), default="livekit")
    room_name: Mapped[str] = mapped_column(String(255), unique=True)
    runtime_contract: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class RuntimeEvent(Base):
    __tablename__ = "runtime_events"
    __table_args__ = (
        UniqueConstraint("event_id"),
        Index(
            "ix_runtime_events_conversation_received_at",
            "conversation_id",
            "received_at",
        ),
    )
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(String(80))
    schema_version: Mapped[int] = mapped_column(default=1)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Voice(Base):
    __tablename__ = "voices"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(20))  # library|clone|design
    status: Mapped[str] = mapped_column(String(20), default="pending")
    tts_provider_id: Mapped[str | None] = mapped_column(
        ForeignKey("provider_configs.id"), nullable=True
    )
    provider_voice_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    languages: Mapped[list[str]] = mapped_column(JSON, default=list)
    style: Mapped[dict] = mapped_column(JSON, default=dict)
    design_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    sample_reference: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    consent: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    name: Mapped[str] = mapped_column(String(200))
    channel: Mapped[str] = mapped_column(String(20))  # web|sip|api
    status: Mapped[str] = mapped_column(String(20), default="active")
    phone_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Campaign(Base):
    __tablename__ = "campaigns"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(200))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    deployment_id: Mapped[str] = mapped_column(ForeignKey("deployments.id"))
    status: Mapped[str] = mapped_column(String(20), default="draft")
    schedule_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    schedule_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    max_attempts: Mapped[int] = mapped_column(Integer, default=1)
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class CampaignTarget(Base):
    __tablename__ = "campaign_targets"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id", "phone_number", name="uq_campaign_targets_campaign_phone"
        ),
        Index("ix_campaign_targets_campaign_status", "campaign_id", "status"),
    )
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    campaign_id: Mapped[str] = mapped_column(ForeignKey("campaigns.id"))
    phone_number: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200), default="")
    variables: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Tool(Base):
    __tablename__ = "tools"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str] = mapped_column(Text)
    tool_type: Mapped[str] = mapped_column(String(50), default="http")
    input_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class AgentTool(Base):
    __tablename__ = "agent_tools"
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), primary_key=True)
    tool_id: Mapped[str] = mapped_column(ForeignKey("tools.id"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    title: Mapped[str] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(1024), default="")
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class AgentKnowledgeDocument(Base):
    __tablename__ = "agent_knowledge_documents"
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
