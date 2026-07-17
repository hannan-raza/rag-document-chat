"""
Retrieval + generation pipeline.
Hybrid search (vector + keyword) fused with RRF, then rerank, then grounded answer.
Scoped per-user: every search filters by user_id so users only see their own docs.
Provider-agnostic: embeddings and generation come from app.providers.
"""
from app.providers import embed, generate
from app.db import connect

# NOTE: each search opens its OWN short-lived connection (autocommit, closed in
# finally). We deliberately do NOT keep a module-level shared connection: these
# functions run concurrently across threads (retrieve is called via
# asyncio.to_thread in the unified /ask path, and sync /query runs in FastAPI's
# threadpool), and a single psycopg connection is not safe for concurrent use.
# A per-call connection also avoids the idle-in-transaction / zombie-connection
# issue (CLAUDE.md §5).

def vector_search(question, user_id, k=10):
    q_vec = embed(question)
    conn = connect()
    conn.autocommit = True
    try:
        rows = conn.execute(
            "SELECT id, content FROM documents "
            "WHERE user_id = %s "
            "ORDER BY embedding <=> %s LIMIT %s",
            (user_id, q_vec, k),
        ).fetchall()
    finally:
        conn.close()
    return [(r[0], r[1]) for r in rows]

def keyword_search(question, user_id, k=10):
    conn = connect()
    conn.autocommit = True
    try:
        rows = conn.execute(
            "SELECT id, content FROM documents "
            "WHERE user_id = %s "
            "AND to_tsvector('english', content) @@ plainto_tsquery('english', %s) "
            "ORDER BY ts_rank(to_tsvector('english', content), "
            "plainto_tsquery('english', %s)) DESC LIMIT %s",
            (user_id, question, question, k),
        ).fetchall()
    finally:
        conn.close()
    return [(r[0], r[1]) for r in rows]

def hybrid_search(question, user_id, k=15, rrf_k=60):
    """Run vector + keyword search (scoped to user), fuse with RRF."""
    vec = vector_search(question, user_id, k=10)
    kw = keyword_search(question, user_id, k=10)
    scores, text = {}, {}
    for rank, (doc_id, content) in enumerate(vec):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        text[doc_id] = content
    for rank, (doc_id, content) in enumerate(kw):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        text[doc_id] = content
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [text[doc_id] for doc_id, score in ranked[:k]]

def rerank(question, candidates, top_n=3):
    """LLM reads each candidate against the question and picks the most relevant."""
    if not candidates:
        return []
    numbered = ""
    for i, chunk in enumerate(candidates):
        numbered += f"[{i}] {chunk}\n\n"
    prompt = (
        f"Question: {question}\n\n"
        f"Below are numbered passages. Identify the {top_n} passages MOST relevant "
        f"to answering the question. Reply with ONLY the numbers, best first, "
        f"comma-separated (e.g. 4,1,7). No other text.\n\n{numbered}"
    )
    reply = generate(prompt, max_tokens=50).strip()
    indices = [int(p) for p in reply.replace(" ", "").split(",") if p.isdigit()]
    return [candidates[i] for i in indices if i < len(candidates)]

def retrieve(question, user_id):
    candidates = hybrid_search(question, user_id, k=15)
    return rerank(question, candidates, top_n=3)

def rewrite_query(question, history):
    """Rewrite a follow-up question into a standalone one using chat history.
    Resolves pronouns/references ('him', 'it', 'that') so retrieval works.
    """
    if not history:
        return question

    convo = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up as a standalone question that can be understood without the "
        "history. Resolve any pronouns or references (like 'him', 'it', 'that') "
        "to the actual entities from the history. If the question is already "
        "standalone, return it unchanged. Return ONLY the rewritten question.\n\n"
        f"Conversation history:\n{convo}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )
    return generate(prompt, max_tokens=100, temperature=0).strip()

def answer(question, chunks, history=None):
    if not chunks:
        return "I don't have any documents to answer from yet. Please upload a PDF first."

    context = "\n\n".join(chunks)

    convo = ""
    if history:
        convo = "Conversation so far:\n" + "\n".join(
            f"{t['role']}: {t['content']}" for t in history
        ) + "\n\n"

    prompt = (
        "You are a helpful assistant answering questions based on the provided context. "
        "Use the context below to answer the question as completely as you can. "
        "If the context genuinely contains no relevant information, say you don't know — "
        "but if the answer is present in the context, answer it fully.\n\n"
        f"{convo}"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    return generate(prompt, max_tokens=400)