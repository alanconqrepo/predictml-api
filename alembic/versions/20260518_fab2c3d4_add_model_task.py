"""add model_task to model_metadata

Revision ID: fab2c3d4
Revises: efa8c9d0
Create Date: 2026-05-18

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "fab2c3d4"
down_revision: Union[str, None] = "efa8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("model_task", sa.String(50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "model_task")
