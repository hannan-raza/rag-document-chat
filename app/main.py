"""
FastAPI app: chat query endpoint + document management (upload/list/delete).
"""
import os
import io
import boto3
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.providers import embed
from app.db import connect
from app.rag import retrieve, answer

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")
S3_BUCKET = os.getenv("S3_BUCKET")

if AWS_PROFILE:
    boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
else:
    boto_session = boto3.Session(region_name=AWS_REGION)
s3 = boto_session.client("s3")

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

class Query(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(q: Query):
    chunks = retrieve(q.question)
    answer_text = answer(q.question)
    return {"question": q.question, "answer": answer_text, "sources": chunks}

@app.get("/documents")
def list_documents():
    conn = connect()
    rows = conn.execute(
        "SELECT source, count(*) FROM documents GROUP BY source ORDER BY source;"
    ).fetchall()
    conn.close()
    return {"documents": [{"source": r[0], "chunks": r[1]} for r in rows]}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    source = file.filename

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=source, Body=pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    try:
        n = ingest_pdf_bytes(pdf_bytes, source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {"source": source, "chunks": n}

@app.delete("/documents/{source}")
def delete_document(source: str):
    conn = connect()
    result = conn.execute("DELETE FROM documents WHERE source = %s;", (source,))
    deleted = result.rowcount
    conn.commit()
    conn.close()
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=source)
    except Exception:
        pass
    return {"source": source, "deleted_chunks": deleted}
