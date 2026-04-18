"""Add anonymous-owner token to chat sessions

Revision ID: 005
Revises: 004
Create Date: 2026-04-17
"""
from alembic import op
import sqlalchemy as sa


revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("owner_token", sa.String(length=36), nullable=True),
    )
    op.create_index(
        op.f("ix_chat_sessions_owner_token"),
        "chat_sessions",
        ["owner_token"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_chat_sessions_owner_token"), table_name="chat_sessions")
    op.drop_column("chat_sessions", "owner_token")
