"""add alert_thresholds to model_metadata

Revision ID: 8ef5a6b3
Revises: 7cd4e5f2
Create Date: 2026-04-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8ef5a6b3"
down_revision: Union[str, None] = "7cd4e5f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("alert_thresholds", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "alert_thresholds")
