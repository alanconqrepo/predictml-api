"""add task_runs table for ARQ job tracking

Revision ID: a1b2c3d4
Revises: fab2c3d4
Create Date: 2026-05-23

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4"
down_revision: Union[str, None] = "fab2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "task_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("task_type", sa.String(50), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("new_version", sa.String(50), nullable=True),
        sa.Column("triggered_by", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column(
            "enqueued_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("logs", sa.Text(), nullable=True),
    )
    op.create_index("ix_task_runs_id", "task_runs", ["id"])
    op.create_index("ix_task_runs_task_type", "task_runs", ["task_type"])
    op.create_index("ix_task_runs_model_name", "task_runs", ["model_name"])
    op.create_index("ix_task_runs_status", "task_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_task_runs_status", table_name="task_runs")
    op.drop_index("ix_task_runs_model_name", table_name="task_runs")
    op.drop_index("ix_task_runs_task_type", table_name="task_runs")
    op.drop_index("ix_task_runs_id", table_name="task_runs")
    op.drop_table("task_runs")
