"""Add organizations, API keys, and the audit log; seed the default organization."""

from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa

revision = "0008_organizations"
down_revision = "0007_session_tokens"
branch_labels = None
depends_on = None

DEFAULT_ORGANIZATION_ID = "00000000-0000-0000-0000-000000000001"


def _ts(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    organizations = op.create_table(
        "organizations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(63), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        _ts("created_at"),
        _ts("updated_at"),
        sa.UniqueConstraint("slug", name="uq_organizations_slug"),
    )
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "organization_id",
            sa.String(36),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("key_prefix", sa.String(16), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("created_by_key_id", sa.String(36), nullable=True),
        _ts("created_at"),
        _ts("last_used_at", nullable=True),
        _ts("expires_at", nullable=True),
        _ts("revoked_at", nullable=True),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    op.create_index("ix_api_keys_organization_id", "api_keys", ["organization_id"])
    op.create_index(
        "ix_api_keys_organization_id_revoked_at",
        "api_keys",
        ["organization_id", "revoked_at"],
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("organization_id", sa.String(36), nullable=True),
        sa.Column("actor_type", sa.String(20), nullable=False),
        sa.Column("actor_id", sa.String(36), nullable=True),
        sa.Column("action", sa.String(80), nullable=False),
        sa.Column("resource_type", sa.String(40), nullable=True),
        sa.Column("resource_id", sa.String(36), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=True),
        _ts("created_at"),
    )
    op.create_index(
        "ix_audit_log_organization_id_created_at",
        "audit_log",
        ["organization_id", "created_at"],
    )
    op.create_index(
        "ix_audit_log_resource_type_resource_id",
        "audit_log",
        ["resource_type", "resource_id"],
    )

    now = datetime.now(timezone.utc)
    op.bulk_insert(
        organizations,
        [
            {
                "id": DEFAULT_ORGANIZATION_ID,
                "name": "Default",
                "slug": "default",
                "status": "active",
                "created_at": now,
                "updated_at": now,
            }
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_log_resource_type_resource_id", table_name="audit_log")
    op.drop_index("ix_audit_log_organization_id_created_at", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_api_keys_organization_id_revoked_at", table_name="api_keys")
    op.drop_index("ix_api_keys_organization_id", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_table("organizations")
