"""
Module 2 unified query — the "Global User Knowledge Base" answer path.

Answers a question from the user's ENTIRE knowledge base in one call: all their
PDFs (Module 1 RAG) AND all their CSVs (Module 2 text-to-SQL), regardless of
which session a file was uploaded in.

Concurrency: the PDF-RAG path and the CSV-SQL path run CONCURRENTLY via
asyncio.to_thread around the existing SYNCHRONOUS DB functions. We deliberately
do NOT convert the DB layer to async/asyncpg (CLAUDE.md §8.2). All Module 2 SQL
safety guards are preserved — the CSV path goes through datasets.run_dataset_query,
which owns those guards.

Flow:
  RAG path:  retrieve(question)                      -> reranked PDF chunks
  CSV path:  pick_datasets -> run_dataset_query(...)  -> raw SQL results
Then a single LLM call synthesizes both into one grounded answer.
"""
import asyncio
import json
import logging

from app.providers import generate, generate_stream
from app.rag import retrieve
from app.datasets import pick_datasets, run_dataset_query, list_datasets_meta

logger = logging.getLogger(__name__)

# Shown when neither the PDFs nor the CSVs have anything relevant. Kept as a
# constant so the streaming and non-streaming paths return the exact same text.
NO_CONTEXT_ANSWER = (
    "I don't have anything in your knowledge base that answers that yet."
)


async def _aiter_in_thread(sync_gen_factory):
    """Bridge a *synchronous* generator onto the event loop without blocking it.

    The provider layer is deliberately sync (CLAUDE.md §8.2), so generate_stream
    blocks on the network between tokens. We run it in a thread and hand tokens
    back through a thread-safe queue, so the ASGI server can flush each token to
    the client while the next one is still being fetched.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def pump():
        try:
            for item in sync_gen_factory():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # surface the failure to the awaiting consumer
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    task = loop.run_in_executor(None, pump)
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await task


async def _csv_path(user_id, question):
    """Pick relevant datasets, then run each one's SQL concurrently.
    Returns a list of raw result dicts (from run_dataset_query). A failure on
    one dataset (e.g. the LLM produced non-safe SQL) is dropped, not fatal — the
    unified answer can still use the other sources.
    """
    try:
        ids = await asyncio.to_thread(pick_datasets, user_id, question)
    except Exception:
        logger.exception("dataset picker failed; skipping CSV path")
        return []
    if not ids:
        return []

    results = await asyncio.gather(
        *(asyncio.to_thread(run_dataset_query, user_id, ds_id, question)
          for ds_id in ids),
        return_exceptions=True,
    )
    good = []
    for ds_id, r in zip(ids, results):  # gather preserves input order
        if isinstance(r, Exception):
            # ValueError = a Module 2 safety guard rejected the SQL (expected,
            # low-noise); anything else is an unexpected DB/logic error.
            level = logging.INFO if isinstance(r, ValueError) else logging.WARNING
            logger.log(level, "dataset %s query dropped: %s", ds_id, r)
        else:
            good.append(r)
    return good


def _build_synthesis_prompt(question, chunks, sql_results, source_names):
    """Assemble the grounded-synthesis prompt from PDF chunks + SQL results.
    Returns None when neither source has anything (caller answers 'I don't know').
    Shared by the streaming and non-streaming paths so both ground on the same
    context (the streaming path lstrips its first token to match _synthesize's
    .strip(); see ask_stream).
    """
    parts = []
    if chunks:
        parts.append("From the user's documents (PDFs):\n" + "\n\n".join(chunks))
    for r in sql_results:
        name = source_names.get(r["dataset_id"], r["table_name"])
        preview = [dict(zip(r["columns"], row)) for row in r["rows"][:20]]
        parts.append(
            f'From spreadsheet "{name}" (SQL: {r["sql"]}):\n'
            + json.dumps(preview, default=str)
        )

    if not parts:
        return None

    context = "\n\n---\n\n".join(parts)
    return (
        "You are answering from the user's personal knowledge base, which may "
        "include text from their PDFs and structured results from their "
        "spreadsheets. Use ONLY the information below. Combine the sources into "
        "one clear, direct answer. If one source is irrelevant to the question, "
        "ignore it. If nothing below actually answers the question, say you "
        "don't know.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _synthesize(question, chunks, sql_results, source_names):
    """Combine PDF chunks + SQL results into ONE grounded answer (single LLM call).
    If neither source has anything, return a plain 'I don't know'.
    """
    prompt = _build_synthesis_prompt(question, chunks, sql_results, source_names)
    if prompt is None:
        return NO_CONTEXT_ANSWER
    return generate(prompt, max_tokens=500).strip()


def _sql_payload(sql_results):
    """Serialize raw SQL results for the API response (tuples -> JSON lists)."""
    return [
        {
            "dataset_id": r["dataset_id"],
            "sql": r["sql"],
            "columns": r["columns"],
            "rows": [list(row) for row in r["rows"]],
        }
        for r in sql_results
    ]


async def _gather(user_id, question):
    """Run the RAG and CSV-SQL retrieval paths concurrently (the un-streamed
    part of the answer). Returns (chunks, sql_results, source_names).

    Tolerant: if one path fails, we still answer from the other rather than 500.
    Every call here is user_id-scoped and goes through the same functions the
    non-streaming path uses, so all Module 2 SQL guards + isolation are intact.
    """
    chunks, sql_results = await asyncio.gather(
        asyncio.to_thread(retrieve, question, user_id),
        _csv_path(user_id, question),
        return_exceptions=True,
    )
    if isinstance(chunks, Exception):
        logger.warning("RAG retrieval failed; answering from CSVs only: %s", chunks)
        chunks = []
    if isinstance(sql_results, Exception):
        # _csv_path is already tolerant, but guard against an unexpected raise.
        logger.warning("CSV path failed; answering from PDFs only: %s", sql_results)
        sql_results = []

    # Friendly filenames for the picked datasets (registry read, no LLM/data).
    source_names = {}
    if sql_results:
        source_names = {m["id"]: m["source"]
                        for m in await asyncio.to_thread(list_datasets_meta, user_id)}

    return chunks, sql_results, source_names


async def ask(user_id, question):
    """Unified query over the user's whole knowledge base.

    Returns:
      {
        answer:        str,                       # the synthesized answer
        sources:       [str, ...],                # PDF chunks used (RAG)
        sql:           [{dataset_id, sql, columns, rows}, ...],  # per dataset
        used_datasets: [dataset_id, ...],
      }
    """
    chunks, sql_results, source_names = await _gather(user_id, question)

    # Final synthesis is a blocking LLM call — keep the event loop free.
    answer_text = await asyncio.to_thread(
        _synthesize, question, chunks, sql_results, source_names
    )

    return {
        "answer": answer_text,
        "sources": chunks,
        "sql": _sql_payload(sql_results),
        "used_datasets": [r["dataset_id"] for r in sql_results],
    }


async def ask_stream(user_id, question, echo_question=None):
    """Streaming twin of ask(): same retrieval, but the final synthesis answer is
    emitted token-by-token instead of all at once.

    `question` is the standalone (history-rewritten) query used for retrieval;
    `echo_question` is the original user text echoed back in the metadata event
    to match the non-streaming response's top-level "question" field (defaults to
    `question` if not supplied).

    Retrieval, dataset-picking, and SQL all run FIRST (not streamed) — only the
    synthesis LLM call streams. Yields structured events for the endpoint to
    serialize as SSE:
        {"type": "token", "text": "..."}   # repeated, the answer as it types
        {"type": "error", "message": ...}  # only if synthesis fails mid-stream
        {"type": "metadata", ...}          # exactly once, LAST (metadata-last)
    The metadata event carries the same sources/sql/used_datasets the
    non-streaming path returns, so the client gets identical grounding info.

    Mid-stream failures are the important case: once StreamingResponse has sent
    the 200 + first token, a raised exception can't become a 500 — the socket
    would just truncate, leaving the client unable to tell a finished answer from
    a silently-failed half-answer. So we catch it, emit an 'error' frame (generic
    message — never the raw exception), and STILL fall through to the metadata
    event so the stream always closes cleanly (the endpoint appends [DONE]).
    """
    # Defaults so the metadata event is always emittable — even if _gather itself
    # raises before we have anything, we still close the stream cleanly.
    chunks, sql_results = [], []
    try:
        chunks, sql_results, source_names = await _gather(user_id, question)

        prompt = _build_synthesis_prompt(question, chunks, sql_results, source_names)
        if prompt is None:
            # No context at all — mirror the non-streaming fallback as one token.
            yield {"type": "token", "text": NO_CONTEXT_ANSWER}
        else:
            # generate_stream is a blocking sync generator; bridge it onto the
            # loop so each token flushes without stalling the event loop.
            first = True
            async for token in _aiter_in_thread(
                lambda: generate_stream(prompt, max_tokens=500)
            ):
                if first:
                    # Match _synthesize's .strip(): the prompt ends with "Answer:",
                    # so the model's first delta is often a leading space/newline.
                    token = token.lstrip()
                    if not token:
                        continue  # keep looking for the first real content
                    first = False
                yield {"type": "token", "text": token}
    except Exception:
        # Covers BOTH a mid-stream synthesis failure (a partial answer already
        # reached the client) AND a pre-yield failure in _gather/prompt-build
        # (nothing sent yet). Either way, once StreamingResponse has sent 200 the
        # raise can't become a 500 — so signal it in-band rather than truncating
        # silently. Log the real cause; send only a generic message. Note:
        # GeneratorExit/CancelledError are BaseException, so a client disconnect
        # is NOT caught here and unwinds normally.
        logger.exception("ask_stream failed")
        yield {
            "type": "error",
            "message": "The answer was interrupted before it finished. "
                       "Please try again.",
        }

    yield {
        "type": "metadata",
        "question": echo_question if echo_question is not None else question,
        "sources": chunks,
        "sql": _sql_payload(sql_results),
        "used_datasets": [r["dataset_id"] for r in sql_results],
    }
