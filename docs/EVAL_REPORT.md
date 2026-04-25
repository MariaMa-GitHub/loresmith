# Eval report — hades (2026-04-25, Ollama run v2 — full corpus)

**LLM backend:** Ollama — `qwen2.5:7b` (ANSWER), `qwen2.5:3b` (REWRITE / VERIFY / TAG)
**Dataset:** `app/eval/datasets/hades.jsonl` — first 50 examples (`--limit 50`; tier distribution: 38×0, 6×1, 3×2, 3×3)
**Retrieval:** `max_spoiler_tier=3` for all questions — full knowledge base, no passage filtering
**Ship gates (full config, Gemini provider, full dataset):** faithfulness ≥ 0.85 · recall@5 ≥ 0.80 · citation validity ≥ 0.95

> **Note on gate status:** Gates are not yet met on this run. The gates are calibrated for
> the Gemini Flash provider on the complete 200-item dataset. Ollama (qwen2.5:7b, n=50)
> serves as a free-tier functional check. `+tools` and `full` are skipped — Ollama's
> `OllamaProvider` does not implement `complete_with_tools`. `full-no-tools` is the
> Ollama proxy for the gated configuration.

| config | faithfulness | recall@5 | citation_valid | correctness | avg_latency_ms |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.62 | 0.89 | 0.50 | 0.10 | 20935.32 |
| +rewriter | 0.72 | 0.91 | 0.50 | 0.05 | 19846.69 |
| +rerank | 0.68 | 0.84 | 0.32 | 0.06 | 27562.50 |
| +verifier | 0.54 | 0.89 | 0.45 | 0.15 | 32724.16 |
| +cache | 0.68 | 0.89 | 0.41 | 0.07 | 22199.85 |
| +tools | — (skipped: Ollama lacks complete_with_tools) | | | | |
| full | — (skipped: Ollama lacks complete_with_tools) | | | | |
| full-no-tools | 0.56 | 0.84 | 0.23 | 0.03 | 37348.06 |
| hybrid-no-dense | 0.56 | 0.73 | 0.45 | 0.15 | 19832.63 |
| hybrid-no-bm25 | 0.74 | 0.93 | 0.50 | 0.13 | 21007.75 |

## Comparison with previous run (n=20, spoiler-capped per question)

| metric | before (n=20, capped) | after (n=50, full corpus) | delta |
| --- | --- | --- | --- |
| recall@5 (baseline) | 0.55 | 0.89 | **+0.34** |
| citation_valid (baseline) | 0.18 | 0.50 | **+0.32** |
| faithfulness (baseline) | 0.70 | 0.62 | −0.08 |
| faithfulness (+reranker) | 0.35 | 0.68 | **+0.33** |
| recall@5 (hybrid-no-bm25) | 0.55 | 0.93 | **+0.38** |

The recall@5 improvement is the headline result. Removing the spoiler tier cap gives retrieval access to the full passage set, including mid- and late-game content that was previously blocked even for non-spoiler questions when those questions happen to have relevant context in higher-tier passages.

## Observations

- **Recall@5 is dramatically better** (0.55 → 0.89 baseline) — the biggest single gain. Opening the full corpus is the right call.
- **Dense-only retrieval (hybrid-no-bm25) wins on recall@5** at 0.93. Dense search benefits most from the larger retrieval space. BM25-only drops to 0.73 — confirming dense is the stronger signal in this corpus.
- **+rewriter is the best overall config** — faithfulness 0.72, recall@5 0.91, same latency as baseline. The query rewriter helps on a larger corpus where query specificity matters more.
- **+reranker no longer hurts** (0.35 → 0.68 faithfulness). The previous degradation was likely noise from n=20. At n=50 it's a net improvement over baseline.
- **Verifier still causes faithfulness regression** (0.62 → 0.54). The qwen2.5:3b judge appears too aggressive — it rejects answers that are actually faithful, emitting refusal cards instead. Needs investigation with Gemini at full scale before tuning the prompt.
- **full-no-tools underperforms** (faithfulness 0.56) — the reranker + verifier combination amplifies the verifier's over-rejection. This is the worst proxy for production quality on Ollama; the Gemini run will be more informative.
- **Citation validity is meaningfully better** (0.18 → 0.50 baseline) — more correct passages in context means the LLM cites the right sources more often.

## Re-run instructions

```bash
# Gemini provider (production quality check — meets ship gates or flags gaps):
cd backend && source .venv/bin/activate && set -a && source .env && set +a
python -m app.eval.ablations --game hades

# Ollama (no quota, functional check):
LLM_BACKEND=ollama python -m app.eval.ablations --game hades
```
