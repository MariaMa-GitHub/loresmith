"""Persist assistant response metadata on chat_messages

Revision ID: 008
Revises: 007
Create Date: 2026-04-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column(
            "response_meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.alter_column("chat_messages", "response_meta", server_default=None)


def downgrade() -> None:
    op.drop_column("chat_messages", "response_meta")
