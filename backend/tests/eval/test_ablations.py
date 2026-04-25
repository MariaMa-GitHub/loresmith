import json
from unittest.mock import MagicMock

import pytest

from app.eval.ablations import (
    AblationConfig,
    _load_existing_results,
    _reset_cache_scope,
    default_matrix,
    merge_report_rows,
    provider_supports_tools,
    render_markdown_report,
)


def test_default_matrix_includes_expected_configs():
    ids = {cfg.config_id for cfg in default_matrix()}
    required = {
        "baseline", "+rewriter", "+rerank", "+verifier", "+cache", "+tools",
        "full", "full-no-tools", "hybrid-no-dense", "hybrid-no-bm25",
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


def test_merge_report_rows_returns_only_completed_configs(tmp_path):
    configs = [
        AblationConfig("baseline"),
        AblationConfig("+rewriter", rewriter=True),
        AblationConfig("+rerank", rerank=True),
    ]
    # Write JSON for two of the three configs.
    (tmp_path / "hades-ablation-baseline.json").write_text(
        json.dumps({"config_id": "baseline", "metrics": {"faithfulness_rate": 0.80}})
    )
    (tmp_path / "hades-ablation-+rewriter.json").write_text(
        json.dumps({"config_id": "+rewriter", "metrics": {"faithfulness_rate": 0.85}})
    )

    rows = merge_report_rows(tmp_path, "hades", configs)
    assert len(rows) == 2
    assert rows[0]["config_id"] == "baseline"
    assert rows[1]["config_id"] == "+rewriter"


def test_merge_report_rows_preserves_matrix_order(tmp_path):
    configs = default_matrix()
    # Write a few out of matrix order.
    for cid in ["+rerank", "baseline", "+rewriter"]:
        (tmp_path / f"hades-ablation-{cid}.json").write_text(
            json.dumps({"config_id": cid, "metrics": {}})
        )

    rows = merge_report_rows(tmp_path, "hades", configs)
    ids = [r["config_id"] for r in rows]
    # Order must follow the canonical matrix, not insertion order.
    matrix_ids = [c.config_id for c in configs]
    assert ids == [mid for mid in matrix_ids if mid in ids]


def test_load_existing_results_skips_corrupt_json(tmp_path):
    configs = [AblationConfig("baseline")]
    (tmp_path / "hades-ablation-baseline.json").write_text("NOT JSON {{{")
    results = _load_existing_results(tmp_path, "hades", configs)
    assert results == []


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
