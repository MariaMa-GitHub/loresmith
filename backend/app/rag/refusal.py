from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.verifier import VerifierVerdict

_DEFAULT_MESSAGE = (
    "I don't have enough evidence in the retrieved passages to answer that confidently."
)

_HEURISTIC_TEMPLATES = (
    "Try asking about a specific character or entity by name.",
    "Narrow your question to a single topic or game mechanic.",
    "Add the game's title to disambiguate (e.g. 'in Hades II').",
)


@dataclass(frozen=True)
class RefusalPayload:
    message: str
    unsupported_claims: list[str] = field(default_factory=list)
    rewrite_suggestions: list[str] = field(default_factory=list)
    question: str = ""


def build_refusal(
    *,
    question: str,
    verdict: VerifierVerdict,
    passages: list[dict],
) -> RefusalPayload:
    suggestions = list(verdict.rewrite_suggestions)[:3]
    if not suggestions:
        suggestions = list(_HEURISTIC_TEMPLATES[: 2 if passages else 3])

    return RefusalPayload(
        message=_DEFAULT_MESSAGE,
        unsupported_claims=list(verdict.unsupported_claims),
        rewrite_suggestions=suggestions,
        question=question,
    )
