"""add categorical_baseline to model_metadata

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5
Create Date: 2026-05-30

Adds a JSON column `categorical_baseline` to model_metadata.
Structure: {feature_name: {category_value: frequency}} e.g.
  {"pclass": {"1st": 0.24, "2nd": 0.21, "3rd": 0.55},
   "sex":    {"male": 0.65, "female": 0.35}}

Used to detect drift on categorical input features via PSI
on the frequency distribution observed in production.
"""

from alembic import op
import sqlalchemy as sa

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_metadata",
        sa.Column("categorical_baseline", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_metadata", "categorical_baseline")
