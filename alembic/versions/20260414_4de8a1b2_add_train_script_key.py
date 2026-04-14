"""add train_script_object_key to model_metadata

Revision ID: 4de8a1b2
Revises: 3ab7f2d1
Create Date: 2026-04-14

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4de8a1b2"
down_revision: Union[str, None] = "3ab7f2d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("train_script_object_key", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "train_script_object_key")
