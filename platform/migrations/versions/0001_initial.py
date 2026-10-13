"""Create initial Voice Labs control-plane tables."""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("provider_configs", sa.Column("id", sa.String(36), primary_key=True), sa.Column("kind", sa.String(20), nullable=False), sa.Column("provider", sa.String(100), nullable=False), sa.Column("configuration", sa.JSON(), nullable=False), sa.Column("secret_reference", sa.String(255)), sa.Column("enabled", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("agents", sa.Column("id", sa.String(36), primary_key=True), sa.Column("name", sa.String(200), nullable=False), sa.Column("description", sa.Text(), nullable=False), sa.Column("system_prompt", sa.Text(), nullable=False), sa.Column("enabled_languages", sa.JSON(), nullable=False), sa.Column("stt_provider_id", sa.String(36), sa.ForeignKey("provider_configs.id")), sa.Column("llm_provider_id", sa.String(36), sa.ForeignKey("provider_configs.id")), sa.Column("tts_provider_id", sa.String(36), sa.ForeignKey("provider_configs.id")), sa.Column("voice_id", sa.String(200)), sa.Column("status", sa.String(30), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("conversations", sa.Column("id", sa.String(36), primary_key=True), sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), nullable=False), sa.Column("status", sa.String(30), nullable=False), sa.Column("started_at", sa.DateTime(timezone=True), nullable=False), sa.Column("ended_at", sa.DateTime(timezone=True)), sa.Column("metadata", sa.JSON(), nullable=False))
    op.create_table("messages", sa.Column("id", sa.String(36), primary_key=True), sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), nullable=False), sa.Column("role", sa.String(20), nullable=False), sa.Column("content", sa.Text(), nullable=False), sa.Column("primary_language", sa.String(20)), sa.Column("detected_languages", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("voice_sessions", sa.Column("id", sa.String(36), primary_key=True), sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), nullable=False), sa.Column("status", sa.String(30), nullable=False), sa.Column("started_at", sa.DateTime(timezone=True), nullable=False), sa.Column("ended_at", sa.DateTime(timezone=True)), sa.Column("transport", sa.String(50), nullable=False), sa.Column("room_name", sa.String(255), nullable=False, unique=True))
    op.create_table("runtime_events", sa.Column("id", sa.String(36), primary_key=True), sa.Column("event_id", sa.String(255), nullable=False, unique=True), sa.Column("event_type", sa.String(80), nullable=False), sa.Column("schema_version", sa.Integer(), nullable=False), sa.Column("session_id", sa.String(36)), sa.Column("conversation_id", sa.String(36)), sa.Column("agent_id", sa.String(36)), sa.Column("payload", sa.JSON(), nullable=False), sa.Column("received_at", sa.DateTime(timezone=True), nullable=False))


def downgrade() -> None:
    for table in ("runtime_events", "voice_sessions", "messages", "conversations", "agents", "provider_configs"):
        op.drop_table(table)
