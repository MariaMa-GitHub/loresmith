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


def test_hades2_heuristic_tier_1_for_chronos():
    tagger = SpoilerTagger(llm=None)
    tier = tagger.heuristic_tier(
        "Chronos is the Titan of Time and the main threat facing Melinoe.",
        game_slug="hades2",
    )
    assert tier == 1


def test_hades2_heuristic_tier_0_for_early_crossroads_context():
    tagger = SpoilerTagger(llm=None)
    tier = tagger.heuristic_tier(
        "Melinoe trains with Hecate at the Crossroads and prepares her incantations.",
        game_slug="hades2",
    )
    assert tier == 0


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


@pytest.mark.asyncio
async def test_hades2_ambiguous_passage_uses_hades2_prompt():
    captured_system = {}

    async def complete(messages, system=None):
        captured_system["system"] = system
        return "1"

    mock_llm = MagicMock()
    mock_llm.complete = complete

    tagger = SpoilerTagger(llm=mock_llm)
    tier = await tagger.tag_async(
        "The reveal changes what Melinoe understands about her mission.",
        game_slug="hades2",
    )

    assert tier == 1
    assert "Hades II" in captured_system["system"]


@pytest.mark.asyncio
async def test_ambiguous_passage_uses_conservative_fallback_when_llm_missing(caplog):
    tagger = SpoilerTagger(llm=None)

    with caplog.at_level("WARNING"):
        tier = await tagger.tag_async(
            "Zagreus learns a troubling truth about his family.",
            game_slug="hades",
        )

    assert tier == 2
    assert "conservative fallback tier 2" in caplog.text


@pytest.mark.asyncio
async def test_ambiguous_passage_uses_conservative_fallback_when_llm_fails(caplog):
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("llm unavailable"))
    tagger = SpoilerTagger(llm=mock_llm)

    with caplog.at_level("WARNING"):
        tier = await tagger.tag_async(
            "The reveal changes what Zagreus understands about his mother.",
            game_slug="hades",
        )

    assert tier == 2
    assert "Spoiler classification failed" in caplog.text


@pytest.mark.asyncio
async def test_spoiler_tagger_emits_trace_for_llm_path():
    trace_names = []

    class _RecordingSpan:
        def set_output(self, output): pass
        def set_metadata(self, metadata): pass

    class _RecordingTracer:
        def trace(self, name, metadata=None, **kwargs):
            trace_names.append(name)
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield _RecordingSpan()

            return _cm()

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="1")

    tagger = SpoilerTagger(llm=mock_llm, tracer=_RecordingTracer())
    tier = await tagger.tag_async(
        "Zagreus has a conversation with Nyx about his origins.",
        game_slug="hades",
    )

    assert tier == 1
    assert trace_names == ["ingestion.spoiler_tag"]
