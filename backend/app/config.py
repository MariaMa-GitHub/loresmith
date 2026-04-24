from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str

    # LLM
    gemini_api_key: str = ""
    llm_backend: str = "gemini"  # gemini | ollama | auto

    # Embeddings
    embedding_backend: str = "local"  # local | gemini
    local_embedding_model: str = "BAAI/bge-base-en-v1.5"  # 768d, matches EMBEDDING_DIM
    retrieval_top_k_per_method: int = Field(default=10, ge=1)
    retrieval_top_k_final: int = Field(default=5, ge=1)

    # Reranker
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = Field(default=20, ge=1)

    # Semantic cache
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    semantic_cache_lookup_limit: int = Field(default=3, ge=1)

    # Verifier
    verifier_enabled: bool = True

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_strong_model: str = "qwen2.5:7b"
    ollama_fast_model: str = "qwen2.5:3b"

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    sentry_dsn: str = ""

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "https://loresmith.vercel.app",
    ]

    # Security
    ingest_token: str = "change-me"
    anon_session_cookie_name: str = "loresmith_anon_session"
    anon_session_cookie_max_age_seconds: int = Field(default=60 * 60 * 24 * 90, ge=1)
    anon_session_cookie_secure: bool = False
    anon_session_cookie_samesite: Literal["lax", "none", "strict"] = "lax"


@lru_cache
def get_settings() -> Settings:
    return Settings()
