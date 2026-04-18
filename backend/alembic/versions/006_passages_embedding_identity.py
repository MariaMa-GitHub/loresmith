"""Track embedding backend/model on passages

Revision ID: 006
Revises: 005
Create Date: 2026-04-17
"""
from alembic import op
import sqlalchemy as sa


revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "passages",
        sa.Column("embedding_backend", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "passages",
        sa.Column("embedding_model", sa.String(length=256), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("passages", "embedding_model")
    op.drop_column("passages", "embedding_backend")
