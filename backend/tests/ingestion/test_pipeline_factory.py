"""Tests for make_embedder — the dispatch between local and Gemini backends."""
from unittest.mock import patch

import pytest

from app.config import Settings
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.local_embedder import LocalEmbedder
from app.ingestion.pipeline import make_embedder


def _settings(**overrides) -> Settings:
    base = {
        "database_url": "postgresql+asyncpg://u:p@h/db",
        "gemini_api_key": "fake-key",
        "embedding_backend": "local",
    }
    base.update(overrides)
    return Settings(**base)


def test_make_embedder_defaults_to_local():
    assert _settings().embedding_backend == "local"
    embedder = make_embedder(_settings())
    assert isinstance(embedder, LocalEmbedder)


def test_make_embedder_local_explicit():
    embedder = make_embedder(_settings(embedding_backend="local"))
    assert isinstance(embedder, LocalEmbedder)


def test_make_embedder_gemini_with_key():
    with patch("app.ingestion.embedder.genai.Client"):
        embedder = make_embedder(_settings(embedding_backend="gemini"))
    assert isinstance(embedder, GeminiEmbedder)


def test_make_embedder_gemini_without_key_raises():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_embedder(_settings(embedding_backend="gemini", gemini_api_key=""))


def test_make_embedder_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown embedding_backend"):
        make_embedder(_settings(embedding_backend="openai"))


def test_make_embedder_case_insensitive():
    embedder = make_embedder(_settings(embedding_backend="LOCAL"))
    assert isinstance(embedder, LocalEmbedder)
