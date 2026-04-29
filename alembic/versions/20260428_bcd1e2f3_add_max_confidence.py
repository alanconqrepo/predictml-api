"""add max_confidence to predictions

Revision ID: bcd1e2f3
Revises: aaa1b2c3
Create Date: 2026-04-28

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bcd1e2f3"
down_revision: Union[str, None] = "aaa1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("max_confidence", sa.Float(), nullable=True),
    )
    op.create_index(
        "ix_predictions_max_confidence",
        "predictions",
        ["max_confidence"],
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_max_confidence", table_name="predictions")
    op.drop_column("predictions", "max_confidence")
