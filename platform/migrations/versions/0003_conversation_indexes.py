"""Index transcript and runtime-state conversation queries."""

from alembic import op

revision = "0003_conversation_indexes"
down_revision = "0002_runtime_contract"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_messages_conversation_created_at",
        "messages",
        ["conversation_id", "created_at", "id"],
    )
    op.create_index(
        "ix_voice_sessions_conversation_started_at",
        "voice_sessions",
        ["conversation_id", "started_at"],
    )
    op.create_index(
        "ix_runtime_events_conversation_received_at",
        "runtime_events",
        ["conversation_id", "received_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_runtime_events_conversation_received_at", table_name="runtime_events"
    )
    op.drop_index(
        "ix_voice_sessions_conversation_started_at", table_name="voice_sessions"
    )
    op.drop_index("ix_messages_conversation_created_at", table_name="messages")
