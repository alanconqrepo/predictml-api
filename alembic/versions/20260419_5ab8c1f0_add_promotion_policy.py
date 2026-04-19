"""add promotion_policy to model_metadata

Revision ID: 5ab8c1f0
Revises: 4de8a1b2
Create Date: 2026-04-19

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5ab8c1f0"
down_revision: Union[str, None] = "4de8a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("promotion_policy", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "promotion_policy")
