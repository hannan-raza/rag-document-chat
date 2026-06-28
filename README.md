# rag-document-chat

A production-shaped **Retrieval-Augmented Generation (RAG)** system for chatting with your PDF documents. Upload PDFs, ask questions in natural language, and get answers grounded in the source material — every answer shows the exact passages it was built from.

Built from scratch (no RAG framework) to demonstrate the full pipeline: ingestion, hybrid retrieval, reranking, grounded generation, evaluation, and cloud deployment on AWS.

---

## What it does

- **Chat with your documents** — upload one or more PDFs and ask questions across all of them.
- **Grounded answers with citations** — every response includes the source passages it used, so answers are auditable and trustworthy.
- **Document management** — add, list, and delete documents; re-uploading a document refreshes only its chunks.
- **Provider-swappable** — embeddings and generation run on **OpenAI** or **AWS Bedrock**, switchable with a single environment variable.

---

## Architecture

The system has two paths: a **synchronous query path** (the API serving the chat UI) and an **asynchronous ingestion path** (a queue-fed worker for bulk document processing).

```
                          ┌─────────────────────────────┐
   Browser (chat UI)  ──► │  FastAPI                     │
                          │   /query  /upload            │
                          │   /documents  (list/delete)  │
                          └──────────┬──────────────────┘
                                     │
              ┌──────────────────────┼───────────────────────┐
              ▼                      ▼                        ▼
     ┌────────────────┐    ┌──────────────────┐     ┌─────────────────┐
     │  Providers     │    │  PostgreSQL +    │     │   S3            │
     │  OpenAI /      │    │  pgvector        │     │   (PDF store)   │
     │  Bedrock       │    │  (vector store)  │     └─────────────────┘
     └────────────────┘    └──────────────────┘

   Async ingestion (bulk):
     producer ──► SQS queue ──► worker (Fargate) ──► S3 download
                                       │
                                       └─► chunk ─► embed ─► store in pgvector
```

### Retrieval pipeline

1. **Hybrid search** — combine semantic **vector search** (pgvector cosine similarity) with **keyword search** (PostgreSQL full-text), so the system catches both meaning and exact terms.
2. **Reciprocal Rank Fusion (RRF)** — merge the two ranked lists by position (`1 / (k + rank)`), not raw scores, so results that rank well in both methods rise to the top.
3. **Reranking** — an LLM reads each candidate passage *against* the question and selects the most relevant, fixing cases where surface similarity misleads.
4. **Grounded generation** — the top passages are passed to the LLM with instructions to answer only from the provided context, with citations returned alongside.

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI (Python) |
| Vector store | PostgreSQL + pgvector (AWS RDS) |
| Embeddings + generation | OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) or AWS Bedrock (Titan, Claude) |
| Document storage | AWS S3 |
| Async ingestion | AWS SQS + a containerized worker on AWS Fargate |
| Container registry | AWS ECR |
| Frontend | Single-file HTML/CSS/JS chat UI |

---

## Project structure

```
rag-document-chat/
├── app/
│   ├── main.py          # FastAPI app: chat + document management endpoints
│   ├── rag.py           # retrieval pipeline (hybrid search + RRF + rerank + answer)
│   ├── providers.py     # provider abstraction (OpenAI / Bedrock), swappable by env var
│   ├── db.py            # shared PostgreSQL + pgvector connection
│   └── worker.py        # async SQS ingestion worker (runs on Fargate)
├── scripts/
│   ├── producer.py      # send ingestion messages to SQS
│   ├── migrate.py       # add the per-document 'source' column
│   └── recreate_table.py# (re)create the vector table at the provider's dimension
├── evaluation/          # evaluation harness (LLM-as-judge, A/B testing)
├── frontend/
│   └── chat.html        # the chat UI
├── Dockerfile
├── requirements.txt
└── .env.example         # configuration template
```

---

## Setup

### Prerequisites
- Python 3.12+
- A PostgreSQL database with the `pgvector` extension (e.g. AWS RDS)
- An OpenAI API key **or** AWS Bedrock access
- (For async ingestion) AWS S3 bucket + SQS queue

### Install

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-chat.git
cd rag-document-chat
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# edit .env with your provider choice, API keys, and database details
```

### Initialize the database

```bash
python -m scripts.recreate_table   # creates the vector table at the right dimension
python -m scripts.migrate          # adds the per-document source column
```

### Run

```bash
# start the API
uvicorn app.main:app --reload

# open frontend/chat.html in a browser (or serve it):
python -m http.server 3000   # then visit http://localhost:3000/frontend/chat.html
```

Upload a PDF in the sidebar, then ask questions.

---

## Switching providers (OpenAI ↔ Bedrock)

The provider is controlled by one environment variable. Because embedding dimensions differ (OpenAI = 1536, Bedrock Titan = 1024), switching requires recreating the table and re-ingesting:

```bash
# in .env
LLM_PROVIDER=bedrock        # or: openai

# then
python -m scripts.recreate_table   # recreates at the new provider's dimension
# re-ingest your documents
```

---

## Async ingestion (production path)

For bulk ingestion, the system uses a decoupled queue + worker instead of synchronous upload:

```bash
# send a document to the ingestion queue
python -m scripts.producer

# run the worker (locally, or deployed on Fargate)
python -m app.worker
```

The worker polls SQS, downloads the PDF from S3, chunks and embeds it, and stores the vectors — with crash-safe retry (messages are only deleted after successful processing). The same worker is containerized (see `Dockerfile`) and runs on AWS Fargate with an IAM task role for credential-free AWS access.

---

## Design notes

- **Hybrid over pure-vector retrieval** — vector search alone blurs exact terms (model names, numbers, acronyms); adding keyword search and reranking measurably improves retrieval on those cases.
- **Provider abstraction** — keeping embeddings/generation behind a single interface means the system isn't locked to one vendor, and the same pipeline can be benchmarked across providers.
- **Per-document management** — each chunk is tagged with its source document, enabling clean per-document refresh and deletion rather than rebuilding the whole index.
- **Production-shaped, not just a script** — decoupled query/ingestion paths, queue-based bulk processing with retries, and containerized deployment reflect how a real system would be structured.

---

## Author

**Hannan Raza** — Backend Engineer
GitHub: [YOUR_GITHUB] · LinkedIn: [YOUR_LINKEDIN]
