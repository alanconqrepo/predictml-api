"""add tags and webhook_url to model_metadata

Revision ID: 51c8f3e0ee1f
Revises: 50080c7f3635
Create Date: 2026-04-12

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "51c8f3e0ee1f"
down_revision: Union[str, Sequence[str], None] = "50080c7f3635"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Ajoute les colonnes tags (JSON) et webhook_url (VARCHAR) à model_metadata."""
    op.add_column("model_metadata", sa.Column("tags", sa.JSON(), nullable=True))
    op.add_column("model_metadata", sa.Column("webhook_url", sa.String(500), nullable=True))


def downgrade() -> None:
    """Supprime les colonnes tags et webhook_url de model_metadata."""
    op.drop_column("model_metadata", "webhook_url")
    op.drop_column("model_metadata", "tags")
