"""add ab testing and shadow deployment fields

Revision ID: 3ab7f2d1
Revises: 2cc916ee
Create Date: 2026-04-13

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3ab7f2d1"
down_revision: Union[str, None] = "2cc916ee"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # model_metadata : champs A/B Testing
    op.add_column(
        "model_metadata",
        sa.Column("traffic_weight", sa.Float(), nullable=True),
    )
    op.add_column(
        "model_metadata",
        sa.Column("deployment_mode", sa.String(20), nullable=True),
    )

    # predictions : flag shadow
    op.add_column(
        "predictions",
        sa.Column(
            "is_shadow",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # Index pour filtrage rapide des prédictions shadow
    op.create_index("ix_predictions_is_shadow", "predictions", ["is_shadow"])
    # Index composé pour les requêtes de comparaison A/B
    op.create_index(
        "ix_predictions_model_shadow",
        "predictions",
        ["model_name", "model_version", "is_shadow"],
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_model_shadow", table_name="predictions")
    op.drop_index("ix_predictions_is_shadow", table_name="predictions")
    op.drop_column("predictions", "is_shadow")
    op.drop_column("model_metadata", "deployment_mode")
    op.drop_column("model_metadata", "traffic_weight")
