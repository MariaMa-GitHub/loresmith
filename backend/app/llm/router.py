from app.llm.base import LLMProvider, TaskType

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
