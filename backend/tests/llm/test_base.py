import pytest
from app.llm.base import LLMProvider, TaskType


def test_task_type_values():
    assert TaskType.ANSWER == "answer"
    assert TaskType.REWRITE == "rewrite"
    assert TaskType.VERIFY == "verify"
    assert TaskType.MODERATE == "moderate"


def test_llm_provider_is_protocol():
    """Verify LLMProvider is a runtime-checkable Protocol."""
    assert hasattr(LLMProvider, "__protocol_attrs__") or (
        hasattr(LLMProvider, "_is_protocol") and LLMProvider._is_protocol
    )


class FakeProvider:
    model_name = "fake-v1"

    async def complete(self, messages, system=None):
        return "answer"

    async def stream(self, messages, system=None):
        yield "chunk"


def test_fake_provider_satisfies_protocol():
    assert isinstance(FakeProvider(), LLMProvider)


@pytest.mark.asyncio
async def test_fake_provider_complete():
    provider = FakeProvider()
    result = await provider.complete([{"role": "user", "content": "hello"}])
    assert result == "answer"


@pytest.mark.asyncio
async def test_fake_provider_stream():
    provider = FakeProvider()
    chunks = []
    async for chunk in provider.stream([{"role": "user", "content": "hello"}]):
        chunks.append(chunk)
    assert chunks == ["chunk"]
