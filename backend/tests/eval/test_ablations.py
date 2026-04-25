from unittest.mock import MagicMock

import pytest

from app.eval.ablations import (
    AblationConfig,
    _reset_cache_scope,
    default_matrix,
    provider_supports_tools,
    render_markdown_report,
)


def test_default_matrix_includes_expected_configs():
    ids = {cfg.config_id for cfg in default_matrix()}
    required = {
        "baseline", "+rewriter", "+rerank", "+verifier", "+cache", "+tools",
        "full", "hybrid-no-dense", "hybrid-no-bm25",
    }
    assert required.issubset(ids)


def test_render_markdown_report_emits_table_header():
    rows = [
        {
            "config_id": "baseline",
            "metrics": {
                "faithfulness_rate": 0.80,
                "retrieval_recall_at_5_mean": 0.77,
                "citation_validity_rate": 0.93,
                "answer_correctness_rate": 0.70,
                "avg_latency_ms": 1234.5,
            },
        },
    ]
    md = render_markdown_report(game_slug="hades", rows=rows)
    assert "| config" in md
    assert "baseline" in md
    assert "0.80" in md


def test_render_markdown_report_marks_skipped_configs():
    md = render_markdown_report(
        game_slug="hades",
        rows=[
            {
                "config_id": "+tools",
                "metrics": {},
                "skipped_reason": "answer provider lacks complete_with_tools support",
            }
        ],
    )
    assert "skipped" in md
    assert "complete_with_tools" in md


def test_ablation_config_retrieval_mode_validates():
    with pytest.raises(ValueError):
        AblationConfig(config_id="bad", retrieval="trigram_only")


def test_provider_supports_tools_true_when_method_present():
    provider = MagicMock(spec=["complete_with_tools"])
    assert provider_supports_tools(provider) is True


def test_provider_supports_tools_false_when_method_absent():
    provider = MagicMock(spec=["complete"])
    assert provider_supports_tools(provider) is False


@pytest.mark.asyncio
async def test_reset_cache_scope_deletes_matching_rows_only(monkeypatch):
    deleted_wheres = []

    class FakeQuery:
        def where(self, clause):
            deleted_wheres.append(str(clause))
            return self

    class FakeSession:
        async def execute(self, stmt):
            return None
        async def commit(self):
            pass

    monkeypatch.setattr(
        "app.eval.ablations.delete",
        lambda model: FakeQuery(),
    )

    await _reset_cache_scope(
        FakeSession(), game_slug="hades", corpus_revision="312:312:2026-04-19"
    )
    # Both WHERE clauses were applied (game_slug and corpus_revision).
    assert len(deleted_wheres) == 2
