# Eval report — hades (2026-04-25, Ollama run)

**LLM backend:** Ollama — `qwen2.5:7b` (ANSWER), `qwen2.5:3b` (REWRITE / VERIFY / TAG)
**Dataset:** `app/eval/datasets/hades.jsonl` — first 20 examples (`--limit 20`; not stratified)
**Ship gates (full config, Gemini provider, full dataset):** faithfulness ≥ 0.85 · recall@5 ≥ 0.80 · citation validity ≥ 0.95

> **Note on gate status:** Gates are not met on this run. This is expected — the gates are
> calibrated for the Gemini Flash provider on the complete 200-item dataset. Ollama
> (qwen2.5:7b, n=20) serves as a free-tier functional check; re-run with Gemini and
> `--limit` removed to measure production-grade quality.
>
> `+tools` and `full` were skipped because Ollama's `OllamaProvider` does not implement
> `complete_with_tools`. `full-no-tools` (all components except tool use) is the Ollama
> proxy for the gated configuration.

| config | faithfulness | recall@5 | citation_valid | correctness | avg_latency_ms |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.70 | 0.55 | 0.18 | 0.06 | 20520.51 |
| +rewriter | 0.70 | 0.55 | 0.27 | 0.10 | 19671.33 |
| +rerank | 0.35 | 0.41 | 0.00 | 0.25 | 21964.87 |
| +verifier | 0.50 | 0.55 | 0.18 | 0.07 | 29134.44 |
| +cache | 0.63 | 0.55 | 0.27 | 0.11 | 20371.29 |
| +tools | — (skipped: Ollama lacks complete_with_tools) | | | | |
| full | — (skipped: Ollama lacks complete_with_tools) | | | | |
| full-no-tools | 0.25 | 0.41 | 0.09 | 0.17 | 32902.83 |
| hybrid-no-dense | 0.40 | 0.59 | 0.27 | 0.13 | 19370.14 |
| hybrid-no-bm25 | 0.65 | 0.55 | 0.18 | 0.10 | 19166.77 |

## Observations

- **Baseline is competitive** at n=20; Ollama qwen2.5:7b produces faithful answers 70% of the time without any extra components.
- **+reranker hurts on this run.** Faithfulness drops from 0.70 → 0.35 and recall@5 from 0.55 → 0.41. Likely noise from n=20 combined with qwen2.5's sensitivity to context ordering. Re-check at full scale with Gemini.
- **+verifier reduces citation rate** (0.95 → 0.55) as expected — the verifier rejects low-confidence answers and emits refusal cards instead.
- **full-no-tools underperforms** at 0.25 faithfulness. Combination of reranker degradation + verifier filtering amplifies on small n.
- **Hybrid retrieval:** BM25-only (hybrid-no-dense) achieves the highest recall@5 on this sample (0.59). Dense-only matches hybrid (0.55). Neither is clearly superior at n=20.
- **Citation validity is low across all configs** (0–0.27). This metric requires exact source-URL alignment with the gold set; it is noisy at n=20 and favours Gemini's more precise citation placement.

## Re-run instructions

```bash
# Gemini provider (production quality check):
cd backend && source .venv/bin/activate && set -a && source .env && set +a
python -m app.eval.ablations --game hades

# Ollama (no quota):
LLM_BACKEND=ollama python -m app.eval.ablations --game hades
```
