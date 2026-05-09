"""initial_schema

Revision ID: 50080c7f3635
Revises:
Create Date: 2026-04-06 17:44:00.408485

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM as PgEnum

# revision identifiers, used by Alembic.
revision: str = "50080c7f3635"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Crée le schéma initial : tables users, model_metadata, predictions, observed_results."""

    # Enum PostgreSQL pour les rôles utilisateur
    userrole_enum = sa.Enum("admin", "user", "readonly", name="userrole")
    userrole_enum.create(op.get_bind(), checkfirst=True)

    # Table users
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(100), nullable=False),
        sa.Column("api_token", sa.String(255), nullable=False),
        sa.Column(
            "role",
            PgEnum("admin", "user", "readonly", name="userrole", create_type=False),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("rate_limit_per_day", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("last_login", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_api_token", "users", ["api_token"], unique=True)

    # Table model_metadata
    op.create_table(
        "model_metadata",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("minio_bucket", sa.String(100), nullable=True),
        sa.Column("minio_object_key", sa.String(255), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("file_hash", sa.String(64), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("algorithm", sa.String(100), nullable=True),
        sa.Column("features_count", sa.Integer(), nullable=True),
        sa.Column("classes", sa.JSON(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("training_metrics", sa.JSON(), nullable=True),
        sa.Column("confidence_threshold", sa.Float(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(255), nullable=True),
        sa.Column("user_id_creator", sa.Integer(), nullable=True),
        sa.Column("trained_by", sa.String(100), nullable=True),
        sa.Column("training_date", sa.DateTime(), nullable=True),
        sa.Column("training_dataset", sa.String(255), nullable=True),
        sa.Column("training_params", sa.JSON(), nullable=True),
        sa.Column("feature_baseline", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_production", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("deprecated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id_creator"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_metadata_id", "model_metadata", ["id"])
    op.create_index("ix_model_metadata_name", "model_metadata", ["name"])
    op.create_index("ix_model_metadata_version", "model_metadata", ["version"])
    op.create_index("ix_model_metadata_mlflow_run_id", "model_metadata", ["mlflow_run_id"])
    op.create_index("ix_model_metadata_user_id_creator", "model_metadata", ["user_id_creator"])
    op.create_index("ix_model_metadata_is_active", "model_metadata", ["is_active"])

    # Table predictions
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("id_obs", sa.String(255), nullable=True),
        sa.Column("input_features", sa.JSON(), nullable=False),
        sa.Column("prediction_result", sa.JSON(), nullable=False),
        sa.Column("probabilities", sa.JSON(), nullable=True),
        sa.Column("response_time_ms", sa.Float(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("client_ip", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_id", "predictions", ["id"])
    op.create_index("ix_predictions_user_id", "predictions", ["user_id"])
    op.create_index("ix_predictions_model_name", "predictions", ["model_name"])
    op.create_index("ix_predictions_id_obs", "predictions", ["id_obs"])
    op.create_index("ix_predictions_timestamp", "predictions", ["timestamp"])

    # Table observed_results
    op.create_table(
        "observed_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("id_obs", sa.String(255), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("observed_result", sa.JSON(), nullable=False),
        sa.Column("date_time", sa.DateTime(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id_obs", "model_name", name="uq_observed_result_obs_model"),
    )
    op.create_index("ix_observed_results_id", "observed_results", ["id"])
    op.create_index("ix_observed_results_id_obs", "observed_results", ["id_obs"])
    op.create_index("ix_observed_results_model_name", "observed_results", ["model_name"])
    op.create_index("ix_observed_results_date_time", "observed_results", ["date_time"])


def downgrade() -> None:
    """Supprime toutes les tables et l'enum userrole."""
    op.drop_table("observed_results")
    op.drop_table("predictions")
    op.drop_table("model_metadata")
    op.drop_table("users")

    sa.Enum(name="userrole").drop(op.get_bind(), checkfirst=True)
