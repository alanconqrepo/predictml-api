"""add pkl_hmac_signature to model_metadata

Revision ID: dfa3b1c2
Revises: cde3f4a5
Create Date: 2026-05-07

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dfa3b1c2"
down_revision: Union[str, None] = "cde3f4a5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("pkl_hmac_signature", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "pkl_hmac_signature")
