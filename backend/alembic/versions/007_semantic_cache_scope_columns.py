"""Add scope columns to semantic_cache

Revision ID: 007
Revises: 006
Create Date: 2026-04-18
"""
from alembic import op
import sqlalchemy as sa


revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM semantic_cache")
    op.add_column("semantic_cache", sa.Column("corpus_revision", sa.String(length=64), nullable=False, server_default=""))
    op.add_column("semantic_cache", sa.Column("max_spoiler_tier", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("semantic_cache", sa.Column("embedding_backend", sa.String(length=32), nullable=False, server_default=""))
    op.add_column("semantic_cache", sa.Column("embedding_model", sa.String(length=256), nullable=False, server_default=""))
    op.alter_column("semantic_cache", "corpus_revision", server_default=None)
    op.alter_column("semantic_cache", "max_spoiler_tier", server_default=None)
    op.alter_column("semantic_cache", "embedding_backend", server_default=None)
    op.alter_column("semantic_cache", "embedding_model", server_default=None)
    op.create_index("ix_semantic_cache_scope", "semantic_cache", ["game_slug", "corpus_revision", "max_spoiler_tier", "embedding_backend", "embedding_model"])


def downgrade() -> None:
    op.drop_index("ix_semantic_cache_scope", table_name="semantic_cache")
    op.drop_column("semantic_cache", "embedding_model")
    op.drop_column("semantic_cache", "embedding_backend")
    op.drop_column("semantic_cache", "max_spoiler_tier")
    op.drop_column("semantic_cache", "corpus_revision")
