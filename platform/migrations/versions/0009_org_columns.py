"""Add nullable organization_id to tenant tables and backfill the default organization."""

from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa

revision = "0009_org_columns"
down_revision = "0008_organizations"
branch_labels = None
depends_on = None

DEFAULT_ORGANIZATION_ID = "00000000-0000-0000-0000-000000000001"

TENANT_TABLES = (
    "agents",
    "provider_configs",
    "conversations",
    "messages",
    "voice_sessions",
    "runtime_events",
    "voices",
    "deployments",
    "campaigns",
    "campaign_targets",
    "tools",
    "knowledge_documents",
)


def _ensure_default_org() -> None:
    bind = op.get_bind()
    exists = bind.execute(
        sa.text("SELECT 1 FROM organizations WHERE id = :id"),
        {"id": DEFAULT_ORGANIZATION_ID},
    ).first()
    if exists:
        return
    now = datetime.now(timezone.utc)
    bind.execute(
        sa.text(
            "INSERT INTO organizations (id, name, slug, status, created_at, updated_at) "
            "VALUES (:id, 'Default', 'default', 'active', :now, :now)"
        ),
        {"id": DEFAULT_ORGANIZATION_ID, "now": now},
    )


def upgrade() -> None:
    for table in TENANT_TABLES:
        op.add_column(table, sa.Column("organization_id", sa.String(36), nullable=True))
        op.create_index(f"ix_{table}_organization_id", table, ["organization_id"])
    _ensure_default_org()
    # NOTE: on very large tables, backfill in id-range batches instead of one UPDATE.
    for table in TENANT_TABLES:
        op.execute(
            sa.text(
                f"UPDATE {table} SET organization_id = :org WHERE organization_id IS NULL"
            ).bindparams(org=DEFAULT_ORGANIZATION_ID)
        )


def downgrade() -> None:
    for table in reversed(TENANT_TABLES):
        op.drop_index(f"ix_{table}_organization_id", table_name=table)
        with op.batch_alter_table(table) as batch:
            batch.drop_column("organization_id")
