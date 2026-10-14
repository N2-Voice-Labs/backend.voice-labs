"""Add voices, deployments, campaigns, tools, knowledge, and call outcomes."""

from alembic import op
import sqlalchemy as sa

revision = "0004_voice_studio"
down_revision = "0003_conversation_indexes"
branch_labels = None
depends_on = None


def _ts(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False)


def upgrade() -> None:
    op.create_table(
        "voices",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("tts_provider_id", sa.String(36), sa.ForeignKey("provider_configs.id")),
        sa.Column("provider_voice_id", sa.String(200)),
        sa.Column("languages", sa.JSON(), nullable=False),
        sa.Column("style", sa.JSON(), nullable=False),
        sa.Column("design_prompt", sa.Text()),
        sa.Column("sample_reference", sa.String(1024)),
        sa.Column("consent", sa.JSON()),
        sa.Column("failure_reason", sa.Text()),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_table(
        "deployments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("phone_number", sa.String(32)),
        sa.Column("configuration", sa.JSON(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_table(
        "campaigns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), nullable=False),
        sa.Column("deployment_id", sa.String(36), sa.ForeignKey("deployments.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("schedule_start", sa.DateTime(timezone=True)),
        sa.Column("schedule_end", sa.DateTime(timezone=True)),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_table(
        "campaign_targets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("campaign_id", sa.String(36), sa.ForeignKey("campaigns.id"), nullable=False),
        sa.Column("phone_number", sa.String(32), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("variables", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id")),
        _ts("created_at"),
        _ts("updated_at"),
        sa.UniqueConstraint(
            "campaign_id", "phone_number", name="uq_campaign_targets_campaign_phone"
        ),
    )
    op.create_index(
        "ix_campaign_targets_campaign_status",
        "campaign_targets",
        ["campaign_id", "status"],
    )
    op.create_table(
        "tools",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("tool_type", sa.String(50), nullable=False),
        sa.Column("input_schema", sa.JSON(), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_table(
        "agent_tools",
        sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), primary_key=True),
        sa.Column("tool_id", sa.String(36), sa.ForeignKey("tools.id"), primary_key=True),
        _ts("created_at"),
    )
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("source", sa.String(1024), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_table(
        "agent_knowledge_documents",
        sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("knowledge_documents.id"),
            primary_key=True,
        ),
        _ts("created_at"),
    )

    # batch mode emits plain ALTER on PostgreSQL and table rebuilds on SQLite,
    # which cannot add a foreign-key constraint to an existing table.
    with op.batch_alter_table("agents") as batch:
        batch.add_column(sa.Column("voice_profile_id", sa.String(36)))
        batch.create_foreign_key(
            "fk_agents_voice_profile_id_voices", "voices", ["voice_profile_id"], ["id"]
        )
    with op.batch_alter_table("conversations") as batch:
        batch.add_column(sa.Column("deployment_id", sa.String(36)))
        batch.add_column(
            sa.Column("channel", sa.String(20), nullable=False, server_default="web")
        )
        batch.add_column(sa.Column("outcome", sa.String(40)))
        batch.add_column(sa.Column("summary", sa.Text()))
        batch.add_column(sa.Column("intent", sa.String(200)))
        batch.add_column(
            sa.Column("actions", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(sa.Column("outcome_recorded_at", sa.DateTime(timezone=True)))
        batch.create_foreign_key(
            "fk_conversations_deployment_id_deployments",
            "deployments",
            ["deployment_id"],
            ["id"],
        )
    op.create_index(
        "ix_conversations_agent_started_at",
        "conversations",
        ["agent_id", "started_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_conversations_agent_started_at", table_name="conversations")
    with op.batch_alter_table("conversations") as batch:
        batch.drop_constraint(
            "fk_conversations_deployment_id_deployments", type_="foreignkey"
        )
        for column in (
            "outcome_recorded_at",
            "actions",
            "intent",
            "summary",
            "outcome",
            "channel",
            "deployment_id",
        ):
            batch.drop_column(column)
    with op.batch_alter_table("agents") as batch:
        batch.drop_constraint("fk_agents_voice_profile_id_voices", type_="foreignkey")
        batch.drop_column("voice_profile_id")

    op.drop_table("agent_knowledge_documents")
    op.drop_table("knowledge_documents")
    op.drop_table("agent_tools")
    op.drop_table("tools")
    op.drop_index("ix_campaign_targets_campaign_status", table_name="campaign_targets")
    op.drop_table("campaign_targets")
    op.drop_table("campaigns")
    op.drop_table("deployments")
    op.drop_table("voices")
