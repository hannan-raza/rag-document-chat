# CLAUDE.md — AskDocs Backend

Project context for Claude Code. Read this first. It describes the architecture,
the conventions to follow, what already exists, and what to build next.

**Do not put secrets in this file or any committed file.** All config comes from
environment variables (loaded from a gitignored `.env` locally, and from the ECS
task definition in production). Reference env var *names*, never values.

---

## 1. What this project is

AskDocs is a multi-tenant "chat with your data" platform with two capabilities:

- **Module 1 — RAG (chat with PDFs):** upload PDFs → chunk → embed → store in
  pgvector → answer questions with hybrid search + rerank + grounded generation.
  **Status: DONE and deployed.**
- **Module 2 — text-to-SQL (chat with CSV/Excel):** upload a CSV → parse → load
  into its own Postgres table → answer analytical questions by generating and
  running SQL. **Status: core backend DONE; unified query + frontend REMAINING.**

Everything is scoped per user via the Auth0 `sub` (the `user_id`). A user only
ever sees and queries their own data.

## 2. Stack

- **API:** FastAPI, served by uvicorn (ASGI). Entry point `app.main:app`.
- **DB:** Postgres (AWS RDS) with the **pgvector** extension.
- **LLM/embeddings:** OpenAI by default (`text-embedding-3-small`, 1536-dim
  embeddings; `gpt-4o-mini` for generation), behind a **provider abstraction**
  (`app/providers.py`) that can switch to AWS Bedrock via the `LLM_PROVIDER` env var.
- **Auth:** Auth0, RS256 JWTs validated against Auth0's JWKS. No local user table.
- **Async ingestion (Module 1 only):** S3 (file store) + SQS (job queue) + a
  separate worker service. CSV ingestion (Module 2) is **synchronous** (no worker).
- **Deploy:** Docker image on AWS Fargate (ECS). Two services from one image:
  `rag-api-svc` (behind an ALB) and `rag-worker-svc` (no load balancer, polls SQS).
- **CSV parsing:** pandas.
- **Frontend:** separate repo (Next.js on Vercel). Calls this API through a
  Next.js rewrite proxy at `/backend/*` to avoid HTTPS→HTTP mixed content.

## 3. File map (`app/`)

- `main.py` — FastAPI app + all endpoints. Endpoints are **thin**: they validate,
  apply auth, and delegate to the modules below. Keep business logic OUT of main.py.
- `db.py` — `connect()` returns a psycopg connection with pgvector registered.
- `providers.py` — `embed(text)` and `generate(prompt)`; provider-agnostic.
- `rag.py` — Module 1 retrieval + generation: `vector_search`, `keyword_search`,
  `hybrid_search` (RRF fusion), `rerank` (LLM judge), `retrieve`, `rewrite_query`
  (history-aware), `answer` (grounded). All searches filter by `user_id`.
- `worker.py` — Module 1 async ingestion consumer (polls SQS, embeds, stores).
- `auth.py` — `get_current_user` dependency: validates the JWT, returns the `sub`.
- `datasets.py` — Module 2: CSV ingestion + text-to-SQL. `create_dataset`,
  `make_table_name`, `infer_pg_type`, `safe_col`, plus the query path
  (`get_dataset`, `generate_sql`, `_is_safe_select`, `run_select`, `phrase_answer`,
  `query_dataset`). `delete_dataset` currently lives in `main.py`.
- `scripts/` — one-off migrations (e.g. `add_user_id.py`).

## 4. Data model

- **`documents`** (Module 1): `id, content, embedding vector(1536), source, user_id`.
  One row per PDF chunk. All chunks for all PDFs live here, filtered by `user_id`.
- **`datasets`** (Module 2 registry): `id, user_id, source (original filename),
  table_name (the real ds_ table), columns (JSONB schema: [{name,type},...]),
  row_count, created_at`. **One row per uploaded CSV file.** This is a catalog —
  it does NOT hold the CSV data.
- **`ds_<slug>_<hash>`** (Module 2 data): one real Postgres table PER uploaded CSV,
  with the CSV's actual columns and rows. The `datasets.table_name` points to it.

## 5. Conventions — FOLLOW THESE

1. **Endpoints are thin.** Validate + auth + delegate. Logic goes in modules.
2. **Per-user isolation is mandatory and non-negotiable.** Every query that touches
   `documents`, `datasets`, or any `ds_` table MUST filter by `user_id`. For
   Module 2, a user's dataset is resolved from the `datasets` registry scoped by
   `user_id` — never let generated SQL choose or reach another user's table.
3. **Parameterized SQL always.** User input goes in as query parameters (`%s`),
   never string-formatted into the SQL. (Table/column *names* that must be
   interpolated are sanitized first — see `safe_col`/`make_table_name`.)
4. **Auth on every data endpoint.** Use `user_id: str = Depends(get_current_user)`.
   The ONLY public endpoint is `GET /` (health check — the ALB needs it unprotected).
5. **Connections:** for short read endpoints set `conn.autocommit = True` to avoid
   idle-in-transaction locks (we had a 21-hour zombie connection bug from a
   module-level connection that never committed — don't reintroduce that pattern).
6. **Provider abstraction:** call `embed`/`generate` from `providers.py`. Don't call
   OpenAI directly elsewhere.
7. **Secrets:** never hardcode. Read from env. `.env` is gitignored; only
   `.env.example` is committed. `auth.py`'s Auth0 domain/audience are PUBLIC
   identifiers (safe), NOT secrets.
8. **Sync vs async:** Module 1 ingestion is async (slow — hundreds of OpenAI embed
   calls) via SQS + worker. CSV ingestion is sync (fast — parse + bulk insert, no
   external API calls). Don't add a worker for CSVs.

## 6. Generated-SQL safety (CRITICAL — Module 2)

Module 2 executes LLM-generated SQL. This is dangerous and must stay guarded:

- Only a single `SELECT` may execute. Reject anything that doesn't start with
  SELECT, contains a semicolon (stacked statements), or contains any mutating
  keyword (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE/GRANT/REVOKE/COPY/etc).
  See `_is_safe_select` in `datasets.py`.
- The generated SQL must target only the user's own resolved table
  (verify the expected `table_name` appears in the SQL).
- Executions are row-capped.
- **Hardening TODO (not yet done):** run these queries under a read-only Postgres
  role as defense-in-depth. For now the guard is application-level only.

When editing anything in the Module 2 query path, preserve these guards. If asked
to relax them, refuse and explain the risk.

## 7. Endpoints

Module 1: `GET /` (health, public), `POST /query`, `GET /documents`,
`POST /upload` (async), `DELETE /documents/{source}`.

Module 2 (done): `POST /datasets/upload`, `GET /datasets`,
`POST /datasets/query`, `DELETE /datasets/{dataset_id}`.

## 8. What to build next — Module 2 unified query

Goal: a single chat that answers from the user's ENTIRE knowledge base — all their
PDFs (RAG) AND all their CSVs (SQL) — regardless of which session a file was
uploaded in. Files belong to the user, not to a chat session ("Global User
Knowledge Base" pattern). No "+" button in chat for now; uploads stay on the
documents page.

Design decisions already made (do not re-litigate):

1. **Dataset-picker (chosen over routing/query-all).** Before running SQL, a small
   cheap LLM call reads the user's dataset metadata from the `datasets` registry
   (filename + column names only — NOT the data) and picks which dataset(s) are
   relevant to the question, or "none". This scales to many CSVs and skips SQL for
   non-data questions. This is the standard production pattern at this scale
   (table retrieval / schema linking); it upgrades later to embedding-based
   retrieval if the number of datasets grows large.

2. **Parallel retrieval WITHOUT a rewrite of the DB layer.** Run the PDF-RAG path
   and the CSV-SQL path concurrently using `asyncio.to_thread` around the existing
   SYNCHRONOUS DB functions. Do NOT convert the whole DB layer to async/asyncpg.

3. **Synthesis.** Combine the retrieved PDF chunks and the SQL results into one
   context block and make a single LLM call that writes one unified answer. If one
   source returns nothing relevant, the LLM ignores it.

Build order (implement + test each before the next):

- **(a) dataset-picker** — `pick_datasets(user_id, question) -> [dataset_id,...]`.
  Reads `datasets` (scoped by `user_id`), builds a metadata list
  (`[id] filename — columns: ...`), asks the LLM which are relevant, parses the
  reply defensively (ignore non-numbers / out-of-range), returns ids (possibly
  empty). One cheap LLM call.
- **(b) unified orchestrator** — given `user_id` + question:
  run RAG retrieval and (picker → generate_sql → run_select for each picked
  dataset) concurrently via `asyncio.to_thread`; collect PDF chunks + SQL results;
  synthesize one answer. Return `{answer, sources (pdf), sql (per dataset),
  used_datasets}`. Keep all Module 2 SQL safety guards.
- **(c) endpoint** — a new endpoint (e.g. `POST /ask`) that calls the orchestrator,
  auth-gated. Keep the existing `/query` and `/datasets/query` working too.
- **(d) frontend** (separate repo) — wire the chat to the unified endpoint; reuse
  existing components.

## 9. Known gaps / hardening backlog (not blockers)

- No SQS dead-letter queue → a permanently-failing ingestion job retries forever.
- No ingestion status tracking → users aren't told when a PDF upload fails
  (UI polls `/documents` and spins). Consider a status field: processing/done/failed.
- Read-only DB role for generated SQL (defense in depth) not yet added.
- CORS is `*` — lock to the frontend origin for production.
- No HTTPS on the ALB (frontend proxies through Next.js instead) — fine for now.
- Module 2 query does 2 LLM calls (SQL gen + phrasing), ~10s. Optimizable:
  skip phrasing for single-value results.
- Relevance threshold: retrieval always returns top-k, so irrelevant "sources"
  can show when the answer is "I don't know". Add a similarity floor / hide sources.
- **Streaming `/ask` — before real production load (multi-tenant concurrency):**
  the SSE path (`ask_stream` + `_aiter_in_thread` in `app/ask.py`) works but
  isn't hardened for many concurrent streams. Three known items (fine for current
  use, matter under load):
  - *Executor isolation:* each stream's `generate_stream` pump runs on the shared
    default ThreadPoolExecutor — the same pool `asyncio.to_thread` uses for
    `retrieve`/`run_dataset_query`. Enough concurrent streams can starve other
    users' retrieval. Give streaming its own executor.
  - *Cancellation on disconnect:* on client disconnect, `_aiter_in_thread`'s
    `finally: await task` blocks until the LLM fully generates, because the sync
    `generate_stream` is never `.close()`d. Close the generator / cancel the pump
    on `GeneratorExit` so the thread frees immediately.
  - *Queue backpressure:* the bridge uses an unbounded `asyncio.Queue`; a slow
    client lets the whole answer buffer in memory. Bound the queue so the pump
    blocks when the consumer falls behind.

## 10. Local run

```
source venv/bin/activate
export AWS_PROFILE=<local profile>        # local only; Fargate uses the IAM task role
uvicorn app.main:app --reload --port 8000
```

DB migrations that hang via Python psycopg have historically worked reliably via
`psql` directly instead. Set `conn.autocommit` appropriately to avoid lock holds.

## 11. Deploy (when ready)

One Docker image runs both API and worker. Build for `linux/amd64`, push to ECR,
then force a new deployment on BOTH `rag-api-svc` and `rag-worker-svc` (same image
tag won't auto-update without forcing). Add new Python deps to `requirements.txt`
(pandas is already added).

## 12. Working style

- Make one focused change at a time; keep diffs reviewable.
- After changes, run `/code-review` and pay special attention to: per-user
  isolation, parameterized SQL, and the generated-SQL safety guards.
- Explain non-obvious decisions briefly in the response (the human is learning the
  system and values understanding why, not just what).
- Never weaken auth, isolation, or SQL safety to make something work.
