from enum import StrEnum
from typing import AsyncIterator, Protocol, runtime_checkable


class TaskType(StrEnum):
    ANSWER = "answer"      # strong model
    REWRITE = "rewrite"    # fast/cheap model
    VERIFY = "verify"      # fast/cheap model
    MODERATE = "moderate"  # fast/cheap model


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol all LLM provider adapters must satisfy."""

    model_name: str

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> str:
        """Non-streaming completion. Returns the full response text."""
        ...

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Streaming completion. Yields text chunks as they arrive."""
        ...
