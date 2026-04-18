"""Scope passage content-hash uniqueness per game

Revision ID: 003
Revises: 002
Create Date: 2026-04-17
"""
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_passages_content_hash", "passages", type_="unique")
    op.create_unique_constraint(
        "uq_passages_game_content_hash",
        "passages",
        ["game_slug", "content_hash"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_passages_game_content_hash", "passages", type_="unique")
    op.create_unique_constraint(
        "uq_passages_content_hash",
        "passages",
        ["content_hash"],
    )
