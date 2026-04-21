"""add parent_version to model_metadata

Revision ID: 7cd4e5f2
Revises: 6bc2d3e1
Create Date: 2026-04-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7cd4e5f2"
down_revision: Union[str, None] = "6bc2d3e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("parent_version", sa.String(50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "parent_version")
