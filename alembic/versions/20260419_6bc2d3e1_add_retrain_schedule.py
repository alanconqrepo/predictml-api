"""add retrain_schedule to model_metadata

Revision ID: 6bc2d3e1
Revises: 5ab8c1f0
Create Date: 2026-04-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6bc2d3e1"
down_revision: Union[str, None] = "5ab8c1f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("retrain_schedule", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "retrain_schedule")
