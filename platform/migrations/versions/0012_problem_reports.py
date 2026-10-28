"""Problem reports (ST-02b): notes filed against a conversation from the debug view."""

from alembic import op
import sqlalchemy as sa

revision = "0012_problem_reports"
down_revision = "0011_voice_tombstone"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "problem_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "organization_id",
            sa.String(36),
            sa.ForeignKey("organizations.id"),
            nullable=False,
        ),
        sa.Column(
            "conversation_id",
            sa.String(36),
            sa.ForeignKey("conversations.id"),
            nullable=False,
        ),
        sa.Column("session_id", sa.String(36), nullable=True),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("reporter", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_problem_reports_organization_id", "problem_reports", ["organization_id"]
    )
    op.create_index(
        "ix_problem_reports_conversation_id", "problem_reports", ["conversation_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_problem_reports_conversation_id", table_name="problem_reports")
    op.drop_index("ix_problem_reports_organization_id", table_name="problem_reports")
    op.drop_table("problem_reports")
