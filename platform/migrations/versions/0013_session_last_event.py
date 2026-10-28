"""Voice session liveness: ``last_event_at`` (ST-13).

Set on every accepted Runtime event; the stale-session reaper compares it (or the
session's start time when no event ever arrived) against a timeout.
"""

from alembic import op
import sqlalchemy as sa

revision = "0013_session_last_event"
down_revision = "0012_problem_reports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("voice_sessions") as batch:
        batch.add_column(
            sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.create_index("ix_voice_sessions_last_event_at", ["last_event_at"])


def downgrade() -> None:
    with op.batch_alter_table("voice_sessions") as batch:
        batch.drop_index("ix_voice_sessions_last_event_at")
        batch.drop_column("last_event_at")
