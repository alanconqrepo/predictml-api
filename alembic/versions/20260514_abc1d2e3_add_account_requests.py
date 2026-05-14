"""add account_requests table

Revision ID: abc1d2e3
Revises: eff1a2b3
Create Date: 2026-05-14

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM

revision: str = "abc1d2e3"
down_revision: Union[str, None] = "eff1a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "DO $$ BEGIN CREATE TYPE accountrequeststatus AS ENUM ('pending', 'approved', 'rejected');"
        " EXCEPTION WHEN duplicate_object THEN NULL; END $$"
    )

    op.create_table(
        "account_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(100), nullable=False),
        sa.Column("message", sa.String(500), nullable=True),
        sa.Column("role_requested", sa.String(20), nullable=False, server_default="user"),
        sa.Column(
            "status",
            PG_ENUM("pending", "approved", "rejected", name="accountrequeststatus", create_type=False),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("rejection_reason", sa.String(300), nullable=True),
        sa.Column("requested_at", sa.DateTime(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "reviewer_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_account_requests_id", "account_requests", ["id"])
    op.create_index("ix_account_requests_email", "account_requests", ["email"])
    op.create_index("ix_account_requests_status", "account_requests", ["status"])


def downgrade() -> None:
    op.drop_table("account_requests")
    sa.Enum(name="accountrequeststatus").drop(op.get_bind(), checkfirst=True)
