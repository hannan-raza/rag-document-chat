"""
FastAPI app: chat query endpoint + document management (upload/list/delete).
"""
import os
import io
import boto3
import json
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, UploadFile, File, HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.providers import embed
from app.db import connect
from app.rag import retrieve, answer,rewrite_query
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

def ingest_pdf_bytes(pdf_bytes, source):
    """Chunk, embed, and store a PDF (raw bytes), tagged by source. Refreshes if re-uploaded."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    conn = connect()
    conn.execute("DELETE FROM documents WHERE source = %s;", (source,))
    for chunk in chunks:
        vector = embed(chunk)
        conn.execute(
            "INSERT INTO documents (content, embedding, source) VALUES (%s, %s, %s)",
            (chunk, np.array(vector), source),
        )
    conn.commit()
    conn.close()
    return len(chunks)

class Turn(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    question: str
    history: list[Turn] = []

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
