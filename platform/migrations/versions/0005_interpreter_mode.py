"""Add interpreter mode columns to agents."""

from alembic import op
import sqlalchemy as sa

revision = "0005_interpreter_mode"
down_revision = "0004_voice_studio"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("agents") as batch:
        batch.add_column(
            sa.Column("mode", sa.String(20), nullable=False, server_default="agent")
        )
        batch.add_column(sa.Column("interpreter", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("agents") as batch:
        batch.drop_column("interpreter")
        batch.drop_column("mode")
