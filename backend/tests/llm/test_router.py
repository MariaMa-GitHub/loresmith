from unittest.mock import AsyncMock, patch

import pytest

from app.config import Settings
from app.llm.base import TaskType
from app.llm.router import LLMRouter, build_llm_router


class FakeProvider:
    def __init__(self, name: str):
        self.model_name = name
        self.complete = AsyncMock(return_value="ok")

        async def stream_gen(messages, system=None):
            yield "chunk"

        self.stream = stream_gen


def test_router_returns_strong_for_answer():
    strong = FakeProvider("gemini-2.5-flash")
    fast = FakeProvider("gemini-2.5-flash-lite")
    router = LLMRouter(strong=strong, fast=fast)
    assert router.for_task(TaskType.ANSWER) is strong


def test_router_returns_fast_for_rewrite():
    strong = FakeProvider("gemini-2.5-flash")
    fast = FakeProvider("gemini-2.5-flash-lite")
    router = LLMRouter(strong=strong, fast=fast)
    assert router.for_task(TaskType.REWRITE) is fast


def test_router_returns_fast_for_tag():
    strong = FakeProvider("gemini-2.5-flash")
    fast = FakeProvider("gemini-2.5-flash-lite")
    router = LLMRouter(strong=strong, fast=fast)
    assert router.for_task(TaskType.TAG) is fast


def test_router_returns_fast_for_verify():
    strong = FakeProvider("gemini-2.5-flash")
    fast = FakeProvider("gemini-2.5-flash-lite")
    router = LLMRouter(strong=strong, fast=fast)
    assert router.for_task(TaskType.VERIFY) is fast


def test_router_returns_fast_for_moderate():
    strong = FakeProvider("gemini-2.5-flash")
    fast = FakeProvider("gemini-2.5-flash-lite")
    router = LLMRouter(strong=strong, fast=fast)
    assert router.for_task(TaskType.MODERATE) is fast


def test_router_returns_fast_for_extract():
    strong = FakeProvider("gemini-2.5-flash")
    fast = FakeProvider("gemini-2.5-flash-lite")
    router = LLMRouter(strong=strong, fast=fast)
    assert router.for_task(TaskType.EXTRACT) is fast


@pytest.mark.asyncio
async def test_router_for_task_result_is_callable():
    strong = FakeProvider("strong")
    fast = FakeProvider("fast")
    router = LLMRouter(strong=strong, fast=fast)
    provider = router.for_task(TaskType.ANSWER)
    result = await provider.complete([{"role": "user", "content": "test"}])
    assert result == "ok"


def _settings(**overrides) -> Settings:
    base = {
        "database_url": "postgresql+asyncpg://u:p@host/db",
        "gemini_api_key": "test-key",
        "llm_backend": "gemini",
        "ollama_base_url": "http://localhost:11434",
        "ollama_strong_model": "qwen2.5:7b",
        "ollama_fast_model": "qwen2.5:3b",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_build_llm_router_uses_gemini_backend():
    strong = FakeProvider("strong")
    fast = FakeProvider("fast")
    with patch("app.llm.router.GeminiProvider", side_effect=[strong, fast]) as mock_gemini:
        router = build_llm_router(_settings(llm_backend="gemini"))
    assert mock_gemini.call_count == 2
    assert router.for_task(TaskType.ANSWER) is strong
    assert router.for_task(TaskType.REWRITE) is fast


def test_build_llm_router_uses_ollama_backend():
    strong = FakeProvider("strong")
    fast = FakeProvider("fast")
    with patch("app.llm.router.OllamaProvider", side_effect=[strong, fast]) as mock_ollama:
        router = build_llm_router(_settings(llm_backend="ollama", gemini_api_key=""))
    assert mock_ollama.call_count == 2
    assert router.for_task(TaskType.ANSWER) is strong
    assert router.for_task(TaskType.TAG) is fast


def test_build_llm_router_auto_falls_back_to_ollama_without_gemini_key():
    strong = FakeProvider("strong")
    fast = FakeProvider("fast")
    with patch("app.llm.router.OllamaProvider", side_effect=[strong, fast]) as mock_ollama:
        router = build_llm_router(_settings(llm_backend="auto", gemini_api_key=""))
    assert mock_ollama.call_count == 2
    assert router.for_task(TaskType.ANSWER) is strong


def test_build_llm_router_requires_gemini_key_for_gemini_backend():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        build_llm_router(_settings(llm_backend="gemini", gemini_api_key=""))
