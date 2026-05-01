from app.eval.stratum_metrics import aggregate_by_stratum


def test_returns_entry_per_stratum():
    results = [
        {"stratum": "factual", "retrieval_recall_at_5": 1.0, "faithful": True,
         "citation_f1": 1.0, "answer_correct": True, "refusal_appropriate": None},
        {"stratum": "factual", "retrieval_recall_at_5": 0.5, "faithful": False,
         "citation_f1": 0.5, "answer_correct": False, "refusal_appropriate": None},
        {"stratum": "adversarial", "retrieval_recall_at_5": None, "faithful": False,
         "citation_f1": None, "answer_correct": False, "refusal_appropriate": True},
    ]
    out = aggregate_by_stratum(results)
    assert set(out.keys()) == {"factual", "adversarial"}
    assert out["factual"]["dataset_size"] == 2
    assert out["factual"]["faithfulness_rate"] == 0.5
    assert out["factual"]["retrieval_recall_at_5_mean"] == 0.75
    assert out["adversarial"]["faithfulness_rate"] == 0.0
    assert out["adversarial"]["refusal_appropriateness_rate"] == 1.0


def test_empty_stratum_skips_nones():
    results = [{"stratum": "ambiguous", "retrieval_recall_at_5": None,
                "faithful": None, "citation_f1": None, "answer_correct": None,
                "refusal_appropriate": None}]
    out = aggregate_by_stratum(results)
    assert out["ambiguous"]["faithfulness_rate"] is None
    assert out["ambiguous"]["retrieval_recall_at_5_mean"] is None


def test_render_markdown_includes_stratum_section():
    from app.eval.ablations import render_markdown_report
    rows = [
        {
            "config_id": "baseline",
            "metrics": {
                "faithfulness_rate": 0.7,
            },
            "skipped_reason": None,
            "metrics_by_stratum": {
                "factual": {
                    "dataset_size": 10,
                    "faithfulness_rate": 0.7,
                    "retrieval_recall_at_5_mean": 0.85,
                    "citation_f1_mean": 0.6,
                    "answer_correctness_rate": 0.5,
                    "refusal_appropriateness_rate": None,
                }
            },
        }
    ]
    md = render_markdown_report(game_slug="hades", rows=rows)
    assert "## Per-stratum breakdown" in md
    assert "factual" in md
