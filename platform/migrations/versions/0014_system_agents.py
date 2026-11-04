"""System-managed agents: ``agents.system`` hides the interpreter presets behind /translations.

``agents.system_key`` + a unique (organization_id, system_key) make their creation race-free.
"""

from alembic import op
import sqlalchemy as sa

revision = "0014_system_agents"
down_revision = "0013_session_last_event"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("agents") as batch:
        batch.add_column(
            sa.Column(
                "system", sa.Boolean(), nullable=False, server_default=sa.false()
            )
        )
        batch.add_column(sa.Column("system_key", sa.String(64), nullable=True))
        batch.create_unique_constraint(
            "uq_agents_org_system_key", ["organization_id", "system_key"]
        )


def downgrade() -> None:
    with op.batch_alter_table("agents") as batch:
        batch.drop_constraint("uq_agents_org_system_key", type_="unique")
        batch.drop_column("system_key")
        batch.drop_column("system")
