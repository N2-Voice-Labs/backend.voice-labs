"""Persist the exact contract used to start a runtime session."""

from alembic import op
import sqlalchemy as sa

revision = "0002_runtime_contract"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "voice_sessions",
        sa.Column("runtime_contract", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("voice_sessions", "runtime_contract")
