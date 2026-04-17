from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ingestion.spoiler_tagger import SpoilerTagger


def test_heuristic_tier_0_for_mechanics():
    tagger = SpoilerTagger(llm=None)
    tier = tagger.heuristic_tier(
        "The Mirror of Night allows Zagreus to spend Darkness on permanent upgrades.",
        game_slug="hades",
    )
    assert tier == 0


def test_heuristic_tier_2_for_persephone():
    tagger = SpoilerTagger(llm=None)
    tier = tagger.heuristic_tier(
        "Persephone is revealed to be Zagreus's true mother.",
        game_slug="hades",
    )
    assert tier == 2


def test_heuristic_tier_3_for_true_ending():
    tagger = SpoilerTagger(llm=None)
    tier = tagger.heuristic_tier(
        "After ten escapes, Hades and Persephone reconcile and she returns to the Underworld.",
        game_slug="hades",
    )
    assert tier == 3


def test_heuristic_tier_1_for_first_escape():
    tagger = SpoilerTagger(llm=None)
    tier = tagger.heuristic_tier(
        "Zagreus attempts his first escape and reaches the surface.",
        game_slug="hades",
    )
    assert tier == 1


@pytest.mark.asyncio
async def test_llm_tagger_called_for_ambiguous():
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="1")

    tagger = SpoilerTagger(llm=mock_llm)
    tier = await tagger.tag_async(
        "Zagreus has a conversation with Nyx about his origins.",
        game_slug="hades",
    )

    assert tier == 1
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_llm_tagger_not_called_for_clear_tier_0():
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="0")

    tagger = SpoilerTagger(llm=mock_llm)
    tier = await tagger.tag_async(
        "Boons are power-ups from the Olympian gods.",
        game_slug="hades",
    )

    assert tier == 0
    mock_llm.complete.assert_not_called()
