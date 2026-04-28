# Eval report — hades (2026-04-28, Ollama run — full corpus, n=200)

**LLM backend:** Ollama — `llama3.1:8b` (ANSWER), `llama3.2:3b` (REWRITE / VERIFY / JUDGE)
**Dataset:** `app/eval/datasets/hades.jsonl` — 200 examples (tier distribution: 152×0, 26×1, 16×2, 6×3; stratum distribution: factual:60, multi_hop:60, ambiguous:40, adversarial:40)
**Retrieval:** `max_spoiler_tier=3` for all questions — full knowledge base, no passage filtering
**Ship gates (full config, Gemini provider, full dataset):** faithfulness ≥ 0.85 · recall@5 ≥ 0.80 · citation validity ≥ 0.95

> **Note on gate status:** The recall@5 gate (≥ 0.80) is met across all hybrid retrieval configs.
> Faithfulness and citation validity gates are not met. The gates are calibrated for the Gemini Flash
> provider; these results use a local 8B model whose answer quality is substantially lower than
> Gemini Flash. `+tools` and `full` are skipped — `OllamaProvider` does not implement
> `complete_with_tools`.

| config | faithfulness | recall@5 | citation_valid | correctness | avg_latency_ms |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.56 | 0.86 | 0.49 | 0.48 | 23143 |
| +rewriter | 0.53 | 0.89 | 0.58 | 0.52 | 23662 |
| +rerank | 0.54 | 0.87 | 0.58 | 0.52 | 26847 |
| +verifier | 0.51 | 0.86 | 0.49 | 0.60 | 33336 |
| +cache | 0.55 | 0.86 | 0.47 | 0.54 | 22481 |
| +tools | — (skipped: OllamaProvider lacks complete_with_tools) | | | | |
| full | — (skipped: OllamaProvider lacks complete_with_tools) | | | | |
| full-no-tools | 0.55 | 0.87 | 0.51 | 0.60 | 39042 |
| hybrid-no-dense | 0.47 | 0.71 | 0.42 | 0.55 | 22196 |
| hybrid-no-bm25 | 0.57 | 0.88 | 0.63 | 0.55 | 22191 |

## Observations

- **Recall@5 meets the ship gate across all hybrid configs (0.86–0.89).** BM25-only (`hybrid-no-dense`) drops to 0.71, confirming that dense retrieval is the stronger signal on this corpus. The hybrid approach is the correct default.
- **`+rewriter` achieves the best recall@5 (0.89)** at negligible latency cost (+519 ms vs baseline). Query rewriting improves retrieval on a large, varied corpus where the original question may not phrase well for keyword or vector search.
- **Dense-only (`hybrid-no-bm25`) has the best citation validity (0.63)** — dense retrieval returns more precisely relevant passages, which the model cites correctly more often.
- **Faithfulness is 0.51–0.57 across all configs**, well below the 0.85 gate. The answer model introduces unsupported claims or paraphrases beyond the retrieved context at high frequency. This reflects a known limitation of 8B parameter models on grounded generation tasks; the gate is calibrated for Gemini Flash.
- **The verifier improves correctness (0.48 → 0.60) but reduces faithfulness (0.56 → 0.51).** The `llama3.2:3b` judge accepts some unfaithful answers and rejects some faithful ones, producing a net faithfulness regression while filtering out more incorrect answers.
- **The reranker has no meaningful effect** on faithfulness (0.56 → 0.54) or recall@5 (0.86 → 0.87). The cross-encoder adds latency (+3704 ms) without a clear quality gain on this corpus at n=200.
- **`full-no-tools` matches `+verifier` on correctness (0.60)** — combining all components does not degrade below the verifier alone, but adds significant latency (39042 ms, the slowest config).
- **`hybrid-no-dense` is the weakest config overall** — recall@5 drops to 0.71, faithfulness to 0.47, and citation validity to 0.42. Dense retrieval is necessary for acceptable performance on this corpus.
- **Latency baseline is ~23 s per question** on CPU-only hardware. The verifier adds ~10 s per question (judge call). Production latency with Gemini Flash on hosted hardware would be substantially lower.

## Re-run instructions

```bash
# Ollama (no API quota — current results):
cd backend && source .venv/bin/activate && set -a && source .env && set +a
LLM_BACKEND=ollama OLLAMA_STRONG_MODEL=llama3.1:8b OLLAMA_FAST_MODEL=llama3.2:3b python -m app.eval.ablations --game hades

# Resume a partial run without re-running completed configs:
LLM_BACKEND=ollama OLLAMA_STRONG_MODEL=llama3.1:8b OLLAMA_FAST_MODEL=llama3.2:3b python -m app.eval.ablations --game hades --resume

# Gemini provider (ship gate check — requires paid API key):
python -m app.eval.ablations --game hades
```
