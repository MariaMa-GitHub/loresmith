"""Scope passage uniqueness to source pages and add an HNSW vector index

Revision ID: 004
Revises: 003
Create Date: 2026-04-17
"""
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_passages_game_content_hash", "passages", type_="unique")
    op.create_unique_constraint(
        "uq_passages_game_source_content_hash",
        "passages",
        ["game_slug", "source_url", "content_hash"],
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_passages_embedding_hnsw
        ON passages
        USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_passages_embedding_hnsw")
    op.drop_constraint(
        "uq_passages_game_source_content_hash",
        "passages",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_passages_game_content_hash",
        "passages",
        ["game_slug", "content_hash"],
    )
