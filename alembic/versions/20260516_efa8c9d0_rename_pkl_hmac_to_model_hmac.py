"""rename pkl_hmac_signature to model_hmac_signature

Revision ID: efa8c9d0
Revises: def7b8c9
Create Date: 2026-05-16

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "efa8c9d0"
down_revision: Union[str, None] = "def7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "model_metadata",
        "pkl_hmac_signature",
        new_column_name="model_hmac_signature",
    )


def downgrade() -> None:
    op.alter_column(
        "model_metadata",
        "model_hmac_signature",
        new_column_name="pkl_hmac_signature",
    )
