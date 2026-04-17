from app.llm.base import LLMProvider

_REWRITE_SYSTEM = (
    "You are a query rewriting assistant. "
    "Rewrite the user's latest question to be fully self-contained "
    "given the prior conversation context. "
    "Output only the rewritten question — no explanation, no preamble."
)

_MAX_HISTORY_TURNS = 3


class QueryRewriter:
    """Rewrites a follow-up question into a standalone query using LLM."""

    def __init__(self, llm: LLMProvider, max_history_turns: int = _MAX_HISTORY_TURNS) -> None:
        self._llm = llm
        self._max_history_turns = max_history_turns

    async def rewrite(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question

        turns = history[-(self._max_history_turns * 2):]

        context_lines = []
        for msg in turns:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{role}: {msg['content']}")

        context_block = "\n".join(context_lines)
        prompt = (
            f"Conversation so far:\n{context_block}\n\n"
            f"New question: {question}\n\n"
            "Rewrite the new question to be fully self-contained:"
        )

        rewritten = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=_REWRITE_SYSTEM,
        )
        return rewritten.strip()
