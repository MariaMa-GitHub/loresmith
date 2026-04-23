from sqlalchemy import UniqueConstraint

from app.db.models import (
    ChatMessage,
    ChatSession,
    Entity,
    EvalRun,
    Passage,
    QueryLog,
    SemanticCache,
    SharedThread,
    UserFeedback,
)


def test_all_models_have_tablename():
    for model in [Passage, Entity, ChatSession, ChatMessage,
                  SharedThread, SemanticCache, EvalRun, UserFeedback, QueryLog]:
        assert hasattr(model, "__tablename__")


def test_passage_has_embedding_column():
    cols = {c.key for c in Passage.__table__.columns}
    assert "embedding" in cols
    assert "embedding_backend" in cols
    assert "embedding_model" in cols
    assert "game_slug" in cols
    assert "content_hash" in cols
    assert "spoiler_tier" in cols


def test_passage_uses_per_source_content_hash_uniqueness():
    uniques = {
        tuple(col.name for col in constraint.columns)
        for constraint in Passage.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("game_slug", "source_url", "content_hash") in uniques


def test_semantic_cache_has_embedding_column():
    cols = {c.key for c in SemanticCache.__table__.columns}
    assert "query_embedding" in cols
    assert "game_slug" in cols
    assert "corpus_revision" in cols
    assert "max_spoiler_tier" in cols
    assert "embedding_backend" in cols
    assert "embedding_model" in cols


def test_chat_message_references_chat_session():
    fks = {fk.column.table.name for fk in ChatMessage.__table__.foreign_keys}
    assert "chat_sessions" in fks


def test_shared_thread_has_slug_pk():
    pk_cols = [c.key for c in SharedThread.__table__.primary_key.columns]
    assert "slug" in pk_cols


def test_chat_session_tracks_owner_token():
    cols = {c.key for c in ChatSession.__table__.columns}
    assert "owner_token" in cols
