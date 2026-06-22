"""add alert_check_logs table for alerting history

Revision ID: e1f2g3h4
Revises: d4e5f6a7b8c9
Create Date: 2026-06-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e1f2g3h4"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "alert_check_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("checked_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("check_type", sa.String(50), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("result", sa.String(30), nullable=False),
        sa.Column("alert_sent", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("webhook_sent", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("new_predictions_count", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_alert_check_logs_id", "alert_check_logs", ["id"])
    op.create_index("ix_alert_check_logs_checked_at", "alert_check_logs", ["checked_at"])
    op.create_index("ix_alert_check_logs_check_type", "alert_check_logs", ["check_type"])
    op.create_index("ix_alert_check_logs_model_name", "alert_check_logs", ["model_name"])


def downgrade() -> None:
    op.drop_index("ix_alert_check_logs_model_name", table_name="alert_check_logs")
    op.drop_index("ix_alert_check_logs_check_type", table_name="alert_check_logs")
    op.drop_index("ix_alert_check_logs_checked_at", table_name="alert_check_logs")
    op.drop_index("ix_alert_check_logs_id", table_name="alert_check_logs")
    op.drop_table("alert_check_logs")
