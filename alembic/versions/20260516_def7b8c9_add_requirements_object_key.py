"""add requirements_object_key to model_metadata

Revision ID: def7b8c9
Revises: cef6a7b8
Create Date: 2026-05-16

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "def7b8c9"
down_revision: Union[str, None] = "cef6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("requirements_object_key", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "requirements_object_key")
