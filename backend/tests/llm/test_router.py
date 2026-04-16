import pytest
from unittest.mock import AsyncMock
from app.llm.base import TaskType
from app.llm.router import LLMRouter


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


@pytest.mark.asyncio
async def test_router_for_task_result_is_callable():
    strong = FakeProvider("strong")
    fast = FakeProvider("fast")
    router = LLMRouter(strong=strong, fast=fast)
    provider = router.for_task(TaskType.ANSWER)
    result = await provider.complete([{"role": "user", "content": "test"}])
    assert result == "ok"
