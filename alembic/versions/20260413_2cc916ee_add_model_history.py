"""add model history

Revision ID: 2cc916ee
Revises: 51c8f3e0ee1f
Create Date: 2026-04-13

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2cc916ee"
down_revision: Union[str, None] = "51c8f3e0ee1f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("changed_by_user_id", sa.Integer(), nullable=True),
        sa.Column("changed_by_username", sa.String(50), nullable=True),
        sa.Column(
            "action",
            sa.String(50),  # native_enum=False → stored as varchar
            nullable=False,
        ),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("changed_fields", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["changed_by_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_history_id", "model_history", ["id"])
    op.create_index("ix_model_history_model_name", "model_history", ["model_name"])
    op.create_index("ix_model_history_model_version", "model_history", ["model_version"])
    op.create_index("ix_model_history_action", "model_history", ["action"])
    op.create_index("ix_model_history_timestamp", "model_history", ["timestamp"])
    op.create_index(
        "ix_model_history_changed_by_user_id", "model_history", ["changed_by_user_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_model_history_changed_by_user_id", table_name="model_history")
    op.drop_index("ix_model_history_timestamp", table_name="model_history")
    op.drop_index("ix_model_history_action", table_name="model_history")
    op.drop_index("ix_model_history_model_version", table_name="model_history")
    op.drop_index("ix_model_history_model_name", table_name="model_history")
    op.drop_index("ix_model_history_id", table_name="model_history")
    op.drop_table("model_history")
