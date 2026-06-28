"""
Async ingestion worker. Polls SQS for messages, downloads PDFs from S3,
chunks + embeds + stores them. Handles 'ingest' and 'delete' actions.
Run from project root:  python -m app.worker
"""
import os
import json
import boto3
import numpy as np
from pgvector.psycopg import register_vector
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from app.providers import embed
from app.db import connect

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
QUEUE_URL = os.getenv("QUEUE_URL")
AWS_PROFILE = os.getenv("AWS_PROFILE")

# AWS clients for S3 + SQS (embeddings come from app.providers)
if AWS_PROFILE:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
else:
    session = boto3.Session(region_name=AWS_REGION)
sqs = session.client("sqs")
s3 = session.client("s3")

def delete_source(source):
    print(f"  Deleting existing chunks for source='{source}'...")
    conn = connect()
    result = conn.execute("DELETE FROM documents WHERE source = %s;", (source,))
    deleted = result.rowcount
    conn.commit()
    conn.close()
    print(f"  Deleted {deleted} chunks.")
    return deleted

def ingest_document(bucket, key):
    print(f"  Downloading s3://{bucket}/{key} ...")
    local_path = "/tmp/" + key.replace("/", "_")
    s3.download_file(bucket, key, local_path)

    print("  Extracting text...")
    reader = PdfReader(local_path)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    print("  Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    print(f"  {len(chunks)} chunks.")

    # refresh: remove this source's old chunks first (not the whole table)
    delete_source(key)

    conn = connect()
    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        conn.execute(
            "INSERT INTO documents (content, embedding, source) VALUES (%s, %s, %s)",
            (chunk, np.array(vector), key),
        )
        if i % 10 == 0:
            print(f"    embedded {i}/{len(chunks)}")
            conn.commit()
    conn.commit()
    conn.close()
    print(f"  Done: {len(chunks)} chunks stored for source='{key}'.")

def handle_message(body):
    action = body.get("action", "ingest")
    if action == "ingest":
        ingest_document(body["bucket"], body["key"])
    elif action == "delete":
        delete_source(body["key"])
    else:
        raise ValueError(f"Unknown action: {action}")

def main():
    print("Worker started. Polling for messages... (Ctrl+C to stop)")
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20,
        )
        messages = response.get("Messages", [])
        if not messages:
            print("  (no messages, still polling...)")
            continue

        msg = messages[0]
        body = json.loads(msg["Body"])
        print(f"\nGot message: {body}")
        try:
            handle_message(body)
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
            print("Message processed and deleted.\n")
        except Exception as e:
            print(f"ERROR processing message: {e}")
            print("Message NOT deleted — it will be retried.\n")

if __name__ == "__main__":
    main()
