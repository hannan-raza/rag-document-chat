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

from app.providers import generate
from app.rag import retrieve
from app.datasets import pick_datasets, run_dataset_query, list_datasets_meta

logger = logging.getLogger(__name__)


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


def _synthesize(question, chunks, sql_results, source_names):
    """Combine PDF chunks + SQL results into ONE grounded answer (single LLM call).
    If neither source has anything, return a plain 'I don't know'.
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
        return "I don't have anything in your knowledge base that answers that yet."

    context = "\n\n---\n\n".join(parts)
    prompt = (
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
    return generate(prompt, max_tokens=500).strip()


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
    # Run RAG retrieval and the whole CSV path (picker + SQL) concurrently.
    # Tolerant: if one path fails, still answer from the other rather than 500.
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

    # Final synthesis is a blocking LLM call — keep the event loop free.
    answer_text = await asyncio.to_thread(
        _synthesize, question, chunks, sql_results, source_names
    )

    return {
        "answer": answer_text,
        "sources": chunks,
        "sql": [
            {
                "dataset_id": r["dataset_id"],
                "sql": r["sql"],
                "columns": r["columns"],
                "rows": [list(row) for row in r["rows"]],
            }
            for r in sql_results
        ],
        "used_datasets": [r["dataset_id"] for r in sql_results],
    }
