"""add auc column to model_metadata

Revision ID: b2c3d4e5
Revises: a1b2c3d4
Create Date: 2026-05-23

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5"
down_revision: Union[str, None] = "a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("auc", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "auc")
