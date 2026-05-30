"""add feature_importances column

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6
Create Date: 2026-05-30
"""

from alembic import op
import sqlalchemy as sa

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("feature_importances", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "feature_importances")
