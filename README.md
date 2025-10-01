
# FastAPI RAG — Azure OpenAI + FASS (Persistent)

A lightweight Retrieval‑Augmented Generation (RAG) API built with **FastAPI**, **Azure OpenAI** (chat + embeddings), and **FAISS** for persistent local vector stores.

- **/ingest**: chunk + embed PDFs from a folder or file list → save to a persistent vector DB (one collection per PDF or a merged collection).
- **/query**: ask a question; pick which collection(s) to search; returns an answer + cited sources.
- **/collections**: list available collections on disk.

> Works locally (venv or Docker) .


## Table of Contents
- [Prerequisites](#prerequisites)
- [Repo Layout](#repo-layout)
- [Environment Variables](#environment-variables)
- [Quickstart — Local (venv)](#quickstart--local-venv)
- [Quickstart — Docker](#quickstart--docker)
- [Swagger Usage](#swagger-usage)
- [API Reference](#api-reference)
- [Vector Store Paths & Mounts](#vector-store-paths--mounts)

## Prerequisites
- Python **3.11** (for local venv runs)
- Azure OpenAI resource with:
  - A **chat** deployment (e.g. `gpt-4o`)
  - An **embedding** deployment (e.g. `text-embedding-3-large`)
- Docker Desktop (optional, for containerized runs)


## Repo Layout
```
app/
  main.py          # FastAPI app (endpoints)
docs/              # put your PDFs here (or mount a folder as /docs)
vectorstores/      # persistent FAISS collections (or mount as /vectorstores)
requirements.txt
Dockerfile
.env.example
README.md
```


## Environment Variables
Create your `.env` from the example and fill **all** Azure 

`.env`:
```
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini

# App
DATA_DIR=/data                 # for Docker; use ./data for pure local runs
VECTORSTORES_DIR=/vectorstores # for Docker; use ./vectorstores for pure local runs
DEFAULT_TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
MAX_CHUNKS_PER_FILE=2000
```


## Quickstart — Local (venv)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # fill your keys
uvicorn app.main:app --reload --port 8000
```
Open: http://localhost:8000/docs


## Quickstart — Docker

> Ensure your PDFs are in a host folder you’ll mount as **/data** and your vector DB target is mounted as **/vectorstores**.

**PowerShell (Windows):**
```powershell
# from the repo root

docker build -t rag-fastapi-azure:local .
docker run --rm -it -p 8000:8000 \
  --env-file .env \
  -v "$PWD/data:/data" \
  -v "$PWD/vectorstores:/vectorstores" \
  rag-fastapi-azure:local
```

Open: http://localhost:8000/docs


## Swagger Usage

### Ingest (build collections)
- Open **POST `/ingest`** → **Try it out** → Body:
```json
{
  "source_dir": "/data",
  "mode": "per_file"
}
```
This creates **one collection per PDF** (collection name = file name without extension).

**Merge several PDFs into one collection:**
```json
{
  "source_dir": "/data",
  "mode": "merge",
  "collection_name": "my_collection"
}
```

### Query
Open **POST `/query`** → Body:
```json
{
  "question": "What is our PTO policy?",
  "collection_names": "my_collection",
  "top_k": 4
}
```

### List collections
Open **GET `/collections`**.


## API Reference

### `POST /ingest`
Request:
```json
{
  "source_dir": "/data",
  "files": [],
  "mode": "per_file",        // "per_file" or "merge"
  "collection_name": null,   // required when mode = "merge"
  "chunk_size": 1000,
  "chunk_overlap": 150,
  "max_chunks_per_file": 2000
}
```
Response:
```json
{
  "created_collections": ["hr_policy", "benefits"],
  "total_chunks": 342
}
```

### `POST /query`
Request:
```json
{
  "question": "…",
  "collection_names": ["hr_policy"],
  "top_k": 4,
  "chat_deployment": null    // optional override
}
```
Response:
```json
{
  "answer": "…",
  "sources": [
    {"collection":"hr_policy","id":"hr_policy-17","metadata":{"source":"/data/hr_policy.pdf","chunk_index":17}}
  ]
}
```

### `GET /collections`
Response:
```json
{ "collections": ["hr_policy","my_collection"] }
```

## Vector Store Paths & Mounts

The app writes to `VECTORSTORES_DIR`. Common cases:

- **Docker (recommended):** set `VECTORSTORES_DIR=/vectorstores` in `.env` and mount your host folder to `/vectorstores`.
- **Local venv:** set `VECTORSTORES_DIR=./vectorstores` and it will write next to your code.

**Symptom:** collection exists in API but no files on host → your app wrote to `/app/vectorstores` while you mounted the host folder to `/vectorstores`.  
**Fix:** either set `VECTORSTORES_DIR=/vectorstores` **or** mount to `/app/vectorstores`.
