"""Human handoffs from agent sessions to translated calls."""
from alembic import op
import sqlalchemy as sa

revision = "0015_handoffs"
down_revision = "0014_system_agents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "handoffs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("organization_id", sa.String(36), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("from_session_id", sa.String(36), sa.ForeignKey("voice_sessions.id"), nullable=True),
        sa.Column("from_conversation_id", sa.String(36), nullable=True),
        sa.Column("call_session_id", sa.String(36), sa.ForeignKey("voice_sessions.id"), nullable=False),
        sa.Column("invite_code", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("requester_name", sa.String(100), nullable=True),
        sa.Column("requester_language", sa.String(20), nullable=False),
        sa.Column("staff_language", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("from_session_id", name="uq_handoffs_from_session"),
        sa.CheckConstraint("status IN ('waiting', 'accepted', 'ended', 'expired')", name="ck_handoffs_status"),
    )
    op.create_index("ix_handoffs_organization_id", "handoffs", ["organization_id"])


def downgrade() -> None:
    op.drop_table("handoffs")
