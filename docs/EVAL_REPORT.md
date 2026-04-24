# Eval report — hades (placeholder)

> **Note:** This report is a placeholder. Run the ablation harness with a live database and
> LLM backend to populate real metrics:
>
> ```bash
> cd backend && source .env
> python -m app.eval.ablations --game hades
> ```
>
> Ship gates (full configuration): faithfulness ≥ 0.85, recall@5 ≥ 0.80, citation validity ≥ 0.95.

| config | faithfulness | recall@5 | citation_valid | correctness | avg_latency_ms |
| --- | --- | --- | --- | --- | --- |
| baseline | - | - | - | - | - |
| +rewriter | - | - | - | - | - |
| +rerank | - | - | - | - | - |
| +verifier | - | - | - | - | - |
| +cache | - | - | - | - | - |
| +tools | - | - | - | - | - |
| full | - | - | - | - | - |
| hybrid-no-dense | - | - | - | - | - |
| hybrid-no-bm25 | - | - | - | - | - |
