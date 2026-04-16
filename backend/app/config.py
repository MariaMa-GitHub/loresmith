from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str

    # LLM
    gemini_api_key: str = ""
    llm_backend: str = "gemini"  # gemini | ollama | auto

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_strong_model: str = "qwen2.5:7b"
    ollama_fast_model: str = "qwen2.5:3b"

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    sentry_dsn: str = ""

    # Security
    ingest_token: str = "change-me"


@lru_cache
def get_settings() -> Settings:
    return Settings()
