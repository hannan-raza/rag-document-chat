"""
Retrieval + generation pipeline.
Hybrid search (vector + keyword) fused with RRF, then rerank, then grounded answer.
Provider-agnostic: embeddings and generation come from app.providers.
"""
from app.providers import embed, generate
from app.db import connect

conn = connect()

def vector_search(question, k=10):
    q_vec = embed(question)
    rows = conn.execute(
        "SELECT id, content FROM documents ORDER BY embedding <=> %s LIMIT %s",
        (q_vec, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]

def keyword_search(question, k=10):
    rows = conn.execute(
        "SELECT id, content FROM documents "
        "WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s) "
        "ORDER BY ts_rank(to_tsvector('english', content), "
        "plainto_tsquery('english', %s)) DESC LIMIT %s",
        (question, question, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]

def hybrid_search(question, k=15, rrf_k=60):
    """Run vector + keyword search, fuse the two ranked lists with RRF."""
    vec = vector_search(question, k=10)
    kw = keyword_search(question, k=10)
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

def retrieve(question):
    candidates = hybrid_search(question, k=15)
    return rerank(question, candidates, top_n=3)

def answer(question):
    chunks = retrieve(question)
    context = "\n\n".join(chunks)
    prompt = (
        "You are a helpful assistant answering questions based on the provided context. "
        "Use the context below to answer the question as completely as you can. "
        "If the context genuinely contains no relevant information, say you don't know — "
        "but if the answer is present in the context, answer it fully.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    return generate(prompt, max_tokens=400)

if __name__ == "__main__":
    question = "How many GPUs were used for training, and what type?"
    print("Question:", question)
    print("\nAnswer:", answer(question))
