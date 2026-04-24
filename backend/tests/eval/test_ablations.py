import pytest

from app.eval.ablations import (
    AblationConfig,
    default_matrix,
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
