"""add status to model_metadata

Revision ID: aaa1b2c3
Revises: 9fa7b8c4
Create Date: 2026-04-26

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "aaa1b2c3"
down_revision: Union[str, None] = "9fa7b8c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "status")
