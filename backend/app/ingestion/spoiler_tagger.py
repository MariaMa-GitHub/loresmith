from __future__ import annotations

from app.llm.base import LLMProvider

_TIER_KEYWORDS: list[tuple[int, list[str]]] = [
    (3, [
        "ten escapes", "10 escapes", "true ending", "final ending",
        "reconcile", "reconciliation", "persephone returns",
        "hades forgives", "epilogue",
    ]),
    (2, [
        "persephone is zagreus", "zagreus's mother", "zagreus's true mother",
        "true mother", "cannot survive", "dies on the surface",
        "real mother", "mother's identity",
    ]),
    (1, [
        "first escape", "reaches the surface", "reaches the mortal world",
        "achilles and patroclus reunite", "patroclus reunite",
        "sisyphus's punishment lightened",
    ]),
]

_LLM_SYSTEM = """You are a spoiler classifier for the video game Hades.
Assign a spoiler tier to the passage:
0 = safe (mechanics, early characters, no plot reveals)
1 = minor (mid-game character arcs, first escape)
2 = major (Persephone identity, surface survival reveal)
3 = endgame (true ending, post-10-escape story)

Reply with a single digit: 0, 1, 2, or 3. Nothing else."""

_AMBIGUOUS_KEYWORDS = [
    "origins", "mother", "family", "secret", "reveals", "truth",
    "surface", "escape", "ending",
]


def _is_ambiguous(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _AMBIGUOUS_KEYWORDS)


class SpoilerTagger:
    """Assigns spoiler_tier 0–3 to a passage.

    Uses heuristic keyword rules first; calls the LLM only for ambiguous
    passages to minimize cost.
    """

    def __init__(self, llm: LLMProvider | None) -> None:
        self._llm = llm

    def heuristic_tier(self, text: str, game_slug: str) -> int:
        # game_slug reserved for per-game keyword lists; Hades-specific for now.
        lower = text.lower()
        for tier, keywords in _TIER_KEYWORDS:
            if any(kw in lower for kw in keywords):
                return tier
        return 0

    async def tag_async(self, text: str, game_slug: str) -> int:
        tier = self.heuristic_tier(text, game_slug)
        if tier > 0:
            return tier

        if self._llm is None or not _is_ambiguous(text):
            return 0

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": f"Passage:\n{text}"}],
                system=_LLM_SYSTEM,
            )
            digit = response.strip()[:1]
            if digit in {"0", "1", "2", "3"}:
                return int(digit)
        except Exception:
            pass

        return 0
