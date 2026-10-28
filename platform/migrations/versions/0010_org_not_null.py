"""Make organization_id NOT NULL with FKs; org-scoped tool names; unique active phone numbers."""

from alembic import op
import sqlalchemy as sa

revision = "0010_org_not_null"
down_revision = "0009_org_columns"
branch_labels = None
depends_on = None

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

ACTIVE_PHONE_WHERE = "status = 'active' AND phone_number IS NOT NULL"
# 0004 created tools.name with an unnamed UNIQUE; PostgreSQL names it tools_name_key.
# The naming convention lets SQLite batch mode address the reflected unnamed constraint.
_UQ_CONVENTION = {"uq": "uq_%(table_name)s_%(column_0_name)s"}


def _fk(table: str) -> str:
    return f"fk_{table}_organization_id_organizations"


def _assert_no_duplicate_active_phone_numbers() -> None:
    """Fail clearly (not with an opaque unique-index error) on existing duplicates."""

    rows = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT phone_number, count(*) FROM deployments "
                f"WHERE {ACTIVE_PHONE_WHERE} "
                "GROUP BY phone_number HAVING count(*) > 1"
            )
        )
        .all()
    )
    if rows:
        listing = ", ".join(f"{number} (x{count})" for number, count in rows)
        raise RuntimeError(
            "Cannot create unique index uq_deployments_active_phone_number: duplicate "
            f"active deployment phone numbers exist: {listing}. Disable or renumber the "
            "duplicate deployments, then re-run the migration."
        )


def upgrade() -> None:
    # Check first so a failure leaves the schema untouched (PostgreSQL DDL is transactional;
    # SQLite is not).
    _assert_no_duplicate_active_phone_numbers()
    # Batch mode emits plain ALTER on PostgreSQL and table rebuilds on SQLite.
    # On large PostgreSQL tables prefer ADD CONSTRAINT ... NOT VALID, then VALIDATE
    # CONSTRAINT, and SET NOT NULL after a validated CHECK (organization_id IS NOT NULL).
    for table in TENANT_TABLES:
        kwargs = {"naming_convention": _UQ_CONVENTION} if table == "tools" else {}
        with op.batch_alter_table(table, **kwargs) as batch:
            batch.alter_column(
                "organization_id", existing_type=sa.String(36), nullable=False
            )
            batch.create_foreign_key(
                _fk(table), "organizations", ["organization_id"], ["id"]
            )
            if table == "tools":
                bind = op.get_bind()
                if bind.dialect.name == "postgresql":
                    batch.drop_constraint("tools_name_key", type_="unique")
                else:
                    batch.drop_constraint("uq_tools_name", type_="unique")
                batch.create_unique_constraint(
                    "uq_tools_organization_id_name", ["organization_id", "name"]
                )
    op.create_index(
        "uq_deployments_active_phone_number",
        "deployments",
        ["phone_number"],
        unique=True,
        postgresql_where=sa.text(ACTIVE_PHONE_WHERE),
        sqlite_where=sa.text(ACTIVE_PHONE_WHERE),
    )


def downgrade() -> None:
    op.drop_index("uq_deployments_active_phone_number", table_name="deployments")
    for table in reversed(TENANT_TABLES):
        with op.batch_alter_table(table) as batch:
            if table == "tools":
                batch.drop_constraint("uq_tools_organization_id_name", type_="unique")
                batch.create_unique_constraint("uq_tools_name", ["name"])
            batch.drop_constraint(_fk(table), type_="foreignkey")
            batch.alter_column(
                "organization_id", existing_type=sa.String(36), nullable=True
            )
