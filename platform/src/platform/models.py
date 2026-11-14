"""Durable control-plane entities."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


DEFAULT_ORGANIZATION_ID = "00000000-0000-0000-0000-000000000001"


class TenantMixin:
    """Row belongs to exactly one organization (see tenancy.py for write-time stamping)."""

    organization_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organizations.id"), nullable=False, index=True
    )


class Agent(TenantMixin, Base):
    __tablename__ = "agents"
    __table_args__ = (
        UniqueConstraint("organization_id", "system_key", name="uq_agents_org_system_key"),
    )
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
    # System-managed agents (the interpreter presets behind ``/translations``) are hidden
    # from ``GET /agents``; users never create or edit them.
    system: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False
    )
    # Idempotency key of a system agent (hash of its config), unique per organisation so
    # concurrent get-or-create cannot make duplicates. NULL for ordinary agents.
    system_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="enabled")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ProviderConfig(TenantMixin, Base):
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


class Conversation(TenantMixin, Base):
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


class Message(TenantMixin, Base):
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


class VoiceSession(TenantMixin, Base):
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
    # SHA-256 hex digests of session tokens (RM-004); the tokens themselves are never stored.
    session_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    guest_token_hashes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Time of the last accepted Runtime event; the reaper's liveness signal (ST-13).
    last_event_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )


class RuntimeEvent(TenantMixin, Base):
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


class Voice(TenantMixin, Base):
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
    # Hard-delete tombstone (status "deleting" while provider cleanup is pending).
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deleted_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sample_deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Deployment(TenantMixin, Base):
    __tablename__ = "deployments"
    __table_args__ = (
        # Phone numbers are globally unique among active deployments (inbound routing).
        Index(
            "uq_deployments_active_phone_number",
            "phone_number",
            unique=True,
            postgresql_where=text("status = 'active' AND phone_number IS NOT NULL"),
            sqlite_where=text("status = 'active' AND phone_number IS NOT NULL"),
        ),
    )
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


class Campaign(TenantMixin, Base):
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


class CampaignTarget(TenantMixin, Base):
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


class Tool(TenantMixin, Base):
    __tablename__ = "tools"
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_tools_organization_id_name"),)
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(100))
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


class KnowledgeDocument(TenantMixin, Base):
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


class KnowledgeChunk(TenantMixin, Base):
    __tablename__ = "knowledge_chunks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("knowledge_documents.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding_model: Mapped[str] = mapped_column(String(100))
    embedding: Mapped[list[float] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AgentKnowledgeDocument(Base):
    __tablename__ = "agent_knowledge_documents"
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


# -- Organisations, API keys, audit log -------------------------------------------------

ORGANIZATION_ROLES = ("owner", "admin", "member")


class Organization(Base):
    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(63), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_organization_id_revoked_at", "organization_id", "revoked_at"),
    )
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    key_prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    role: Mapped[str] = mapped_column(String(20))
    created_by_key_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AuditLog(Base):
    """Append-only; deliberately no foreign keys so rows outlive what they describe."""

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_audit_log_resource_type_resource_id", "resource_type", "resource_id"),
    )
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    organization_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    actor_type: Mapped[str] = mapped_column(String(20))
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    action: Mapped[str] = mapped_column(String(80))
    resource_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class ProblemReport(TenantMixin, Base):
    """A "report a problem" note on a conversation (ST-02b); filed by staff or a guest."""

    __tablename__ = "problem_reports"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id"), index=True
    )
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    note: Mapped[str] = mapped_column(Text)
    # Who filed it: "api_key:<id>", "session:<id>", or the principal name.
    reporter: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Handoff(TenantMixin, Base):
    __tablename__ = "handoffs"
    __table_args__ = (
        UniqueConstraint("from_session_id", name="uq_handoffs_from_session"),
        CheckConstraint("status IN ('waiting', 'accepted', 'ended', 'expired')", name="ck_handoffs_status"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    from_session_id: Mapped[str | None] = mapped_column(ForeignKey("voice_sessions.id"), nullable=True)
    from_conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    call_session_id: Mapped[str] = mapped_column(ForeignKey("voice_sessions.id"))
    invite_code: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="waiting")
    requester_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    requester_language: Mapped[str] = mapped_column(String(20))
    staff_language: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


from . import tenancy  # noqa: E402,F401  (registers the before_flush stamping hook)
