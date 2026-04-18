import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_requires_database_url(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)  # ensure no .env file is found
    with pytest.raises((ValidationError, Exception)):
        Settings(_env_file=None)


def test_settings_defaults():
    s = Settings(
        database_url="postgresql+asyncpg://u:p@host/db",
        gemini_api_key="test-key",
        _env_file=None,
    )
    assert s.llm_backend == "gemini"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_strong_model == "qwen2.5:7b"
    assert s.ollama_fast_model == "qwen2.5:3b"
    assert s.retrieval_top_k_per_method == 10
    assert s.retrieval_top_k_final == 5
    assert s.anon_session_cookie_name == "loresmith_anon_session"
    assert s.anon_session_cookie_samesite == "lax"


def test_settings_ingest_token_has_default():
    s = Settings(
        database_url="postgresql+asyncpg://u:p@host/db",
        gemini_api_key="test-key",
        _env_file=None,
    )
    assert s.ingest_token != ""
