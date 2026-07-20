"""
FastAPI app: chat query endpoint + document management (upload/list/delete).
"""
import os
import asyncio
import boto3
import json
from fastapi import FastAPI, UploadFile, File, HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from dotenv import load_dotenv
from app.datasets import create_dataset, query_dataset

from app.db import connect
from app.rag import retrieve, answer,rewrite_query
from app.ask import ask, ask_stream
from app.auth import get_current_user

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")
S3_BUCKET = os.getenv("S3_BUCKET")

if AWS_PROFILE:
    boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
else:
    boto_session = boto3.Session(region_name=AWS_REGION)
s3 = boto_session.client("s3")
sqs = boto_session.client("sqs")
QUEUE_URL = os.getenv("QUEUE_URL")

app = FastAPI(title="Grounded RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

def delete_dataset(user_id, dataset_id):
    """Drop a user's dataset: its data table + its registry row, atomically.
    Both run in one transaction (Postgres DDL is transactional), so we never
    leave a registry row pointing at a dropped table. try/finally guarantees the
    connection is closed on every path (no leaked/zombie connections — §5)."""
    conn = connect()
    conn.autocommit = False  # all-or-nothing: registry row + data table together
    try:
        row = conn.execute(
            "SELECT table_name FROM datasets WHERE id = %s AND user_id = %s",
            (dataset_id, user_id),
        ).fetchone()
        if not row:
            return False
        table_name = row[0]
        conn.execute("DELETE FROM datasets WHERE id = %s AND user_id = %s",
                     (dataset_id, user_id))
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}";')
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class Turn(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    question: str
    history: list[Turn] = []

class DatasetQuery(BaseModel):
    dataset_id: int
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(q: Query, user_id: str = Depends(get_current_user)):
    history = [{"role": t.role, "content": t.content} for t in q.history]

    # resolve follow-up into a standalone query
    standalone = rewrite_query(q.question, history)

    # retrieve ONCE, scoped to this user's documents only
    chunks = retrieve(standalone, user_id)

    answer_text = answer(standalone, chunks, history=history)

    return {"question": q.question, "answer": answer_text, "sources": chunks}

async def _sse(events):
    """Serialize the orchestrator's event dicts as Server-Sent Events, then a
    terminal [DONE] sentinel so the client knows the stream is complete.
    Uses jsonable_encoder (the same encoder FastAPI applies to the non-streaming
    JSON response) so Decimal/datetime cell values in SQL rows encode identically
    across both paths — as JSON numbers/ISO strings, not str()-coerced."""
    async for ev in events:
        yield f"data: {json.dumps(jsonable_encoder(ev))}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/ask")
async def ask_unified(q: Query, stream: bool = False,
                      user_id: str = Depends(get_current_user)):
    """Unified chat over the user's ENTIRE knowledge base (all PDFs + all CSVs).
    Thin: resolve any follow-up to a standalone question, then delegate to the
    orchestrator, which runs the RAG and CSV-SQL paths concurrently and
    synthesizes one answer. Scoped to this user throughout.

    ?stream=true streams the synthesis answer token-by-token as SSE (retrieval,
    picking, and SQL still run first); otherwise returns the full JSON at once.
    """
    history = [{"role": t.role, "content": t.content} for t in q.history]
    # rewrite_query is a blocking LLM call — keep the event loop free
    standalone = await asyncio.to_thread(rewrite_query, q.question, history)
    if stream:
        return StreamingResponse(
            # echo_question = the ORIGINAL question, so the streamed metadata
            # matches the non-streaming response's top-level "question" field.
            _sse(ask_stream(user_id, standalone, echo_question=q.question)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await ask(user_id, standalone)
    return {"question": q.question, **result}

@app.get("/documents")
def list_documents(user_id: str = Depends(get_current_user)):
    conn = connect()
    conn.autocommit = True
    rows = conn.execute(
        "SELECT source, count(*) FROM documents "
        "WHERE user_id = %s "
        "GROUP BY source ORDER BY source;",
        (user_id,),
    ).fetchall()
    conn.close()
    return {"documents": [{"source": r[0], "chunks": r[1]} for r in rows]}

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    source = file.filename

    # save the PDF to S3
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=source, Body=pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    # hand off to the worker via SQS — include the owner so chunks get tagged
    try:
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps({
                "action": "ingest",
                "bucket": S3_BUCKET,
                "key": source,
                "user_id": user_id,
            }),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Queueing failed: {e}")

    return {"source": source, "status": "processing"}

@app.delete("/documents/{source}")
def delete_document(source: str, user_id: str = Depends(get_current_user)):
    conn = connect()
    conn.autocommit = True
    result = conn.execute(
        "DELETE FROM documents WHERE source = %s AND user_id = %s;",
        (source, user_id),
    )
    deleted = result.rowcount
    conn.close()

    # only remove the S3 object if this user actually owned chunks for it
    if deleted > 0:
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=source)
        except Exception:
            pass
    return {"source": source, "deleted_chunks": deleted}


@app.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    csv_bytes = file.file.read()
    try:
        result = create_dataset(csv_bytes, file.filename, user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load CSV: {e}")
    return result

@app.post("/datasets/query")
def query_dataset_endpoint(q: DatasetQuery, user_id: str = Depends(get_current_user)):
    """Text-to-SQL against ONE of the user's datasets. Thin: delegate to
    query_dataset, which resolves the table from the user-scoped registry and
    applies all Module 2 SQL safety guards."""
    try:
        return query_dataset(user_id, q.dataset_id, q.question)
    except ValueError as e:
        # "Dataset not found" -> 404 (or another user's id); guard rejections -> 400
        msg = str(e)
        raise HTTPException(status_code=404 if "not found" in msg.lower() else 400,
                            detail=msg)

@app.get("/datasets")
def list_datasets_endpoint(user_id: str = Depends(get_current_user)):
    conn = connect()
    conn.autocommit = True
    rows = conn.execute(
        "SELECT id, source, row_count, columns FROM datasets "
        "WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return {"datasets": [
        {"id": r[0], "source": r[1], "rows": r[2], "columns": r[3]}
        for r in rows
    ]}

@app.delete("/datasets/{dataset_id}")
def delete_dataset_endpoint(dataset_id: int, user_id: str = Depends(get_current_user)):
    ok = delete_dataset(user_id, dataset_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"deleted": dataset_id}
