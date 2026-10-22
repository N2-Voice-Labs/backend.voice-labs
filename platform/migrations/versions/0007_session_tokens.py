"""Add hashed session tokens to voice sessions (RM-004)."""

from alembic import op
import sqlalchemy as sa

revision = "0007_session_tokens"
down_revision = "0006_agent_translation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("voice_sessions") as batch:
        batch.add_column(sa.Column("session_token_hash", sa.String(64), nullable=True))
        batch.add_column(sa.Column("guest_token_hashes", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("voice_sessions") as batch:
        batch.drop_column("guest_token_hashes")
        batch.drop_column("session_token_hash")
