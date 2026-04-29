from app.config import Settings
from app.llm.base import LLMProvider, TaskType
from app.llm.gemini import GeminiProvider
from app.llm.ollama import OllamaProvider

_STRONG_TASKS = {TaskType.ANSWER}


class LLMRouter:
    """Dispatches LLM calls to the appropriate provider based on task type."""

    def __init__(self, strong: LLMProvider, fast: LLMProvider) -> None:
        self._strong = strong
        self._fast = fast

    def for_task(self, task: TaskType) -> LLMProvider:
        """Return the provider best suited for the given task."""
        if task in _STRONG_TASKS:
            return self._strong
        return self._fast


def _build_gemini_pair(settings: Settings) -> tuple[LLMProvider, LLMProvider]:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "llm_backend=gemini requires GEMINI_API_KEY to be set."
        )
    return (
        GeminiProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_strong_model,
            min_call_interval=settings.gemini_min_call_interval,
        ),
        GeminiProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_fast_model,
            min_call_interval=settings.gemini_min_call_interval,
        ),
    )


def _build_ollama_pair(settings: Settings) -> tuple[LLMProvider, LLMProvider]:
    return (
        OllamaProvider(
            model_name=settings.ollama_strong_model,
            base_url=settings.ollama_base_url,
        ),
        OllamaProvider(
            model_name=settings.ollama_fast_model,
            base_url=settings.ollama_base_url,
        ),
    )


def build_llm_router(settings: Settings) -> LLMRouter:
    """Build the configured strong/fast provider pair for the app."""
    backend = settings.llm_backend.lower()
    if backend == "gemini":
        strong, fast = _build_gemini_pair(settings)
    elif backend == "ollama":
        strong, fast = _build_ollama_pair(settings)
    elif backend == "auto":
        if settings.gemini_api_key:
            strong, fast = _build_gemini_pair(settings)
        else:
            strong, fast = _build_ollama_pair(settings)
    else:
        raise ValueError(
            f"Unknown llm_backend={backend!r}; expected 'gemini', 'ollama', or 'auto'."
        )
    return LLMRouter(strong=strong, fast=fast)
