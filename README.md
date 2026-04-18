# Loresmith

A free-to-use, open-source RAG (Retrieval-Augmented Generation) platform for video-game lore. Ask natural-language questions about a game and get grounded answers with inline citations, spoiler-tier filtering, and multi-turn chat — all backed by wiki content retrieved at query time. Currently supports **Hades** and **Hades II**, with a pluggable adapter interface for adding more games.

---

## Features

- **Hybrid retrieval** — BM25 + dense vector search (pgvector) fused via Reciprocal Rank Fusion
- **Multi-turn chat** — follow-up questions are rewritten into standalone queries before retrieval
- **Spoiler control** — passages are tagged tier 0–3 at ingestion; retrieval enforces a configurable max tier per request
- **Streaming answers** — server-sent events with inline `[N]` citations linked to source passages
- **Multi-game** — pluggable `GameAdapter` interface; currently supports Hades and Hades II
- **Eval harness** — CLI runner with JSON report output over 150 hand-labeled questions across four strata (factual, multi-hop, ambiguous, adversarial)

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12 + FastAPI |
| Frontend | Next.js 16 + Tailwind + shadcn/ui |
| Database | Postgres + pgvector |
| LLM (answers) | Gemini 2.5 Flash |
| LLM (rewriter / tagger) | Gemini 2.5 Flash-Lite |
| Embeddings | `bge-base-en-v1.5` (local, 768d); optional `gemini-embedding-001` via `EMBEDDING_BACKEND=gemini` |
| Retrieval | `rank_bm25` + pgvector cosine ANN + RRF |
| Observability | Langfuse |
| Scraping | `httpx` + `selectolax`, robots-aware, Fandom MediaWiki API, conservative Fandom robots fallback |

---

## Prerequisites

- Python `3.12`
- Node `24.15.0`
- Postgres with the `pgvector` extension (e.g. [Neon](https://neon.tech) free tier)
- A [Google AI Studio](https://aistudio.google.com) API key (`GEMINI_API_KEY`)

---

## Setup

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set DATABASE_URL and GEMINI_API_KEY
set -a && source .env && set +a && alembic upgrade head   # apply migrations
uvicorn app.main:app --reload   # starts on :8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev   # starts on :3000
```

**Run tests:**
```bash
cd backend && pytest -v
```

---

## Ingestion

Ingest wiki content into the database before querying. The pipeline scrapes, chunks, embeds, and tags each passage with a spoiler tier.

```bash
cd backend && source .venv/bin/activate && set -a; source .env; set +a

# Dry run (no DB writes — useful for testing)
python -m app.ingestion.pipeline --game hades --dry-run
python -m app.ingestion.pipeline --game hades2 --dry-run

# Full ingest
python -m app.ingestion.pipeline --game hades
python -m app.ingestion.pipeline --game hades2
```

On Fandom wikis, the scraper uses the live `robots.txt` whenever it can fetch it directly. If Cloudflare serves `robots.txt` behind a browser challenge, Loresmith falls back to a conservative Fandom policy that only permits article and `api.php?action=...` fetches and still blocks the obvious non-content namespaces.

The embedder defaults to the local `bge-base-en-v1.5` model (downloaded on first run to `~/.cache/huggingface/`). To use Gemini embeddings instead, set `EMBEDDING_BACKEND=gemini`. **Do not mix backends across ingest and query — the two embedding spaces are incompatible.**

---

## Adding a game

1. Create `backend/app/adapters/<game>.py` implementing the `GameAdapter` protocol (`app/adapters/base.py`).
2. Add the adapter class to `_ADAPTER_CLASSES` in `backend/app/games.py`.
3. Run ingestion with `--game <slug>`.

---

## Project structure

```
backend/
  app/
    adapters/     GameAdapter protocol + per-game adapters
    ingestion/    scraper, chunker, embedder, spoiler_tagger, pipeline CLI
    retrieval/    bm25, dense, hybrid (RRF)
    rag/          query rewriter, pipeline, Jinja prompts
    llm/          LLMProvider protocol + Gemini / Ollama adapters
    eval/         labeled dataset (datasets/hades.jsonl, 150 questions)
    db/           SQLAlchemy models, session factory, Alembic migrations
    tracing/      Langfuse wrapper
    main.py       FastAPI entrypoint
  tests/
frontend/
  src/
    app/          game picker + chat pages
    components/   GamePicker, ChatView, MessageBubble, HistorySidebar
    lib/api.ts    fetch helpers + SSE reader
```

---

## License

MIT
