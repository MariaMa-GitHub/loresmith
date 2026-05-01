"""Add embedding_input_kind column to passages

Revision ID: 009
Revises: 008
Create Date: 2026-04-30

NOTE: Run `alembic upgrade head` before re-ingesting with the new asymmetric
embedder. Existing passages retain the default value 'document'.
"""

import sqlalchemy as sa

from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # server_default is set only to backfill existing rows, then immediately
    # removed so the application layer controls the value going forward.
    op.add_column(
        "passages",
        sa.Column(
            "embedding_input_kind",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'document'"),
        ),
    )
    op.alter_column("passages", "embedding_input_kind", server_default=None)


def downgrade() -> None:
    op.drop_column("passages", "embedding_input_kind")
