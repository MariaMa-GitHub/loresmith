# Eval report — hades (placeholder)

> **Note:** This report is a placeholder. The ablation harness is fully implemented but the free-tier
> Gemini API quota (20 requests/day for Flash-Lite, intermittent 503 on Flash) is too low to complete
> a multi-config run in a single session.
>
> **To populate real metrics, use one of the following:**
>
> **Option A — Ollama (recommended, no quota):**
> ```bash
> ollama pull qwen2.5:7b && ollama pull qwen2.5:3b
> ollama serve   # in a separate terminal
>
> cd backend && source .venv/bin/activate && set -a && source .env && set +a
> LLM_BACKEND=ollama python -m app.eval.ablations --game hades
> ```
>
> **Option B — Gemini free tier (multi-day, quota-safe):**
> Run one or two configs per day to stay within 20 requests/day for Flash-Lite.
> A rate-limiting flag is wired in to prevent per-minute 429s:
> ```bash
> cd backend && source .venv/bin/activate && set -a && source .env && set +a
> # Run 3 baseline configs today (~15 Flash-Lite calls)
> GEMINI_MIN_CALL_INTERVAL=7 python -m app.eval.ablations --game hades --limit 25
> ```
>
> **Option C — Paid Gemini API key:**
> ```bash
> cd backend && source .venv/bin/activate && set -a && source .env && set +a
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
