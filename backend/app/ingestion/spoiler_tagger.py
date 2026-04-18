from __future__ import annotations

import logging

from app.adapters.base import DEFAULT_SPOILER_PROFILE, SpoilerProfile
from app.llm.base import LLMProvider
from app.tracing.langfuse import noop_tracer

logger = logging.getLogger(__name__)


def _profile_for_game(game_slug: str) -> SpoilerProfile:
    # Imported lazily so the tagger stays importable before the adapter
    # registry is resolved (tests construct SpoilerTagger without a live app).
    from app.games import ADAPTERS

    adapter = ADAPTERS.get(game_slug)
    if adapter is None:
        logger.warning(
            "No adapter registered for game %r; using conservative default "
            "spoiler profile",
            game_slug,
        )
        return DEFAULT_SPOILER_PROFILE
    return adapter.spoiler_profile


def _is_ambiguous(text: str, ambiguous_keywords: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ambiguous_keywords)


class SpoilerTagger:
    """Assigns spoiler_tier 0–3 to a passage.

    Uses heuristic keyword rules first; calls the LLM only for ambiguous
    passages to minimize cost. Per-game rules live on the adapter's
    ``spoiler_profile`` attribute.
    """

    def __init__(self, llm: LLMProvider | None, tracer=None) -> None:
        self._llm = llm
        self._tracer = tracer or noop_tracer()

    def heuristic_tier(self, text: str, game_slug: str) -> int:
        profile = _profile_for_game(game_slug)
        lower = text.lower()
        for tier, keywords in profile.tier_keywords:
            if any(kw in lower for kw in keywords):
                return tier
        return 0

    async def tag_async(self, text: str, game_slug: str) -> int:
        profile = _profile_for_game(game_slug)
        tier = self.heuristic_tier(text, game_slug)
        if tier > 0:
            return tier

        if not _is_ambiguous(text, profile.ambiguous_keywords):
            return 0

        fallback_tier = profile.fallback_ambiguous_tier
        if self._llm is None:
            logger.warning(
                "No LLM available for ambiguous spoiler classification; "
                "using conservative fallback tier %d for %s",
                fallback_tier,
                game_slug,
            )
            return fallback_tier

        try:
            with self._tracer.trace(
                "ingestion.spoiler_tag",
                metadata={"game": game_slug, "fallback_tier": fallback_tier},
            ) as span:
                response = await self._llm.complete(
                    messages=[{"role": "user", "content": f"Passage:\n{text}"}],
                    system=profile.system_prompt,
                )
                span.set_output(response)
            digit = response.strip()[:1]
            if digit in {"0", "1", "2", "3"}:
                return int(digit)
            logger.warning(
                "Malformed spoiler classifier output %r; using conservative "
                "fallback tier %d for %s",
                response,
                fallback_tier,
                game_slug,
            )
        except Exception as exc:
            logger.warning(
                "Spoiler classification failed; using conservative fallback "
                "tier %d for %s: %s",
                fallback_tier,
                game_slug,
                exc,
            )

        return fallback_tier
