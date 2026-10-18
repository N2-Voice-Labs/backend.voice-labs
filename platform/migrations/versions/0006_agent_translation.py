"""Add agent translation config (agent + translator, TR-040)."""

from alembic import op
import sqlalchemy as sa

revision = "0006_agent_translation"
down_revision = "0005_interpreter_mode"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("agents") as batch:
        batch.add_column(sa.Column("translation", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("agents") as batch:
        batch.drop_column("translation")
