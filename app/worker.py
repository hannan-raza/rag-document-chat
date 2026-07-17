"""
Async ingestion worker. Polls SQS for messages, downloads PDFs from S3,
chunks + embeds + stores them (tagged with the owner's user_id).
Handles 'ingest' and 'delete' actions.
Run from project root:  python -m app.worker
"""
import os
import json
import boto3
import numpy as np
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from app.providers import embed
from app.db import connect

# Reject extractions where an abnormal share of tokens are lone characters.
# Clean prose sits near 0.006; a PDF mangled into "Z A I N  M U N I R" hits ~0.96.
# A generous 0.30 floor never trips on real text but catches character-spacing.
MAX_SINGLE_CHAR_RATIO = 0.30

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
QUEUE_URL = os.getenv("QUEUE_URL")
AWS_PROFILE = os.getenv("AWS_PROFILE")

if AWS_PROFILE:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
else:
    session = boto3.Session(region_name=AWS_REGION)
sqs = session.client("sqs")
s3 = session.client("s3")

def delete_source(source, user_id):
    print(f"  Deleting existing chunks for source='{source}' (user={user_id})...")
    conn = connect()
    result = conn.execute(
        "DELETE FROM documents WHERE source = %s AND user_id = %s;",
        (source, user_id),
    )
    deleted = result.rowcount
    conn.commit()
    conn.close()
    print(f"  Deleted {deleted} chunks.")
    return deleted

def extract_pdf_text(path):
    """Extract text from a PDF with pdfplumber. Unlike pypdf, it handles the
    glyph-positioning encodings that otherwise come out character-spaced
    ('Z A I N  M U N I R'), which silently poisons both keyword and vector search."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def single_char_ratio(text):
    """Fraction of whitespace-separated tokens that are a single alphabetic
    character. Near 0 for clean prose; ~1 for character-spaced garbage. Used as
    an extraction-quality gate so we never index mangled text again."""
    tokens = text.split()
    if not tokens:
        return 0.0
    singles = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    return singles / len(tokens)


def ingest_from_path(local_path, key, user_id):
    """Extract -> quality-gate -> chunk -> (replace) -> embed -> store.
    Separated from the S3 download so ingestion can be tested on a local file."""
    print("  Extracting text...")
    full_text = extract_pdf_text(local_path)

    # Quality gate: refuse to index a corrupted (character-spaced) extraction.
    # We check BEFORE deleting existing chunks, so a bad re-ingest never wipes
    # good data — the old chunks stay and nothing garbage is stored.
    ratio = single_char_ratio(full_text)
    if ratio > MAX_SINGLE_CHAR_RATIO:
        raise ValueError(
            f"Refusing to ingest '{key}': {ratio:.0%} of tokens are single "
            f"characters (threshold {MAX_SINGLE_CHAR_RATIO:.0%}) — the PDF text "
            f"looks character-spaced/corrupted. Nothing was stored."
        )

    print("  Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    print(f"  {len(chunks)} chunks (single-char ratio {ratio:.3f}).")

    # refresh: remove this user's old chunks for this source first
    delete_source(key, user_id)

    conn = connect()
    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        conn.execute(
            "INSERT INTO documents (content, embedding, source, user_id) "
            "VALUES (%s, %s, %s, %s)",
            (chunk, np.array(vector), key, user_id),
        )
        if i % 10 == 0:
            print(f"    embedded {i}/{len(chunks)}")
            conn.commit()
    conn.commit()
    conn.close()
    print(f"  Done: {len(chunks)} chunks stored for source='{key}' (user={user_id}).")


def ingest_document(bucket, key, user_id):
    print(f"  Downloading s3://{bucket}/{key} (user={user_id}) ...")
    local_path = "/tmp/" + key.replace("/", "_")
    s3.download_file(bucket, key, local_path)
    ingest_from_path(local_path, key, user_id)

def handle_message(body):
    action = body.get("action", "ingest")
    user_id = body.get("user_id")
    if not user_id:
        raise ValueError("Message missing user_id — cannot process without owner.")
    if action == "ingest":
        ingest_document(body["bucket"], body["key"], user_id)
    elif action == "delete":
        delete_source(body["key"], user_id)
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