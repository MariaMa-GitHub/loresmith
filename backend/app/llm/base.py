from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class TaskType(StrEnum):
    ANSWER = "answer"      # strong model
    REWRITE = "rewrite"    # fast/cheap model
    TAG = "tag"            # fast/cheap model
    VERIFY = "verify"      # fast/cheap model
    JUDGE = "judge"        # strong model — verifier / grounded judge
    MODERATE = "moderate"  # fast/cheap model
    EXTRACT = "extract"    # fast/cheap model — structured entity extraction


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

    async def complete_json(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        system: str | None = None,
    ) -> BaseModel:
        """Structured JSON completion validated against the given Pydantic schema.

        Implementations should request constrained JSON output from the
        underlying model (e.g. Gemini ``response_schema``, Ollama ``format``)
        and validate the response with ``schema.model_validate_json``. Callers
        treat ``ValidationError`` and provider exceptions as ``abstain`` —
        do not silently coerce malformed output into a default verdict.
        """
        ...
