"""Add unique constraint to passages.content_hash

Revision ID: 002
Revises: 001
Create Date: 2026-04-16
"""
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_passages_content_hash", table_name="passages")
    op.create_unique_constraint(
        "uq_passages_content_hash", "passages", ["content_hash"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_passages_content_hash", "passages", type_="unique")
    op.create_index(
        op.f("ix_passages_content_hash"), "passages", ["content_hash"], unique=False
    )
