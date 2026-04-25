"""add training_stats to model_metadata

Revision ID: 9fa7b8c4
Revises: 8ef5a6b3
Create Date: 2026-04-25

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9fa7b8c4"
down_revision: Union[str, None] = "8ef5a6b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("training_stats", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "training_stats")
