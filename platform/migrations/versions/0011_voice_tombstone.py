"""Voice hard-delete tombstone columns.

A deleted voice keeps its row (id, consent, timestamps) but loses provider id, sample
reference and design prompt. ``status`` gains the values ``deleting`` and ``deleted``.
"""

from alembic import op
import sqlalchemy as sa

revision = "0011_voice_tombstone"
down_revision = "0010_org_not_null"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("voices") as batch:
        batch.add_column(sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("deleted_by", sa.String(64), nullable=True))
        batch.add_column(
            sa.Column("provider_deleted_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(
            sa.Column("sample_deleted_at", sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("voices") as batch:
        batch.drop_column("sample_deleted_at")
        batch.drop_column("provider_deleted_at")
        batch.drop_column("deleted_by")
        batch.drop_column("deleted_at")
