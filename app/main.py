import os
import json
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import faiss
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import AzureOpenAI

# -----------------------
# Setup & configuration
# -----------------------
load_dotenv()


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large").strip()
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini").strip()

DEFAULT_DOCS_DIR = Path(os.getenv("DEFAULT_DOCS_DIR", "./docs"))
VECTOR_DBS_DIR = Path(os.getenv("VECTOR_DBS_DIR", "./vector_dbs"))
DEFAULT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DBS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))

# Azure OpenAI client
if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY):
    raise RuntimeError("Missing Azure OpenAI environment variables. Check .env file.")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_ENDPOINT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)



# -----------------------
# FastAPI init
# -----------------------
app = FastAPI(title="RAG FastAPI (Azure OpenAI + FAISS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Models
# -----------------------
class IngestRequest(BaseModel):
    collection_name: Optional[str] = Field(
        default=None,
        description="Name for the vector DB collection. Defaults to folder name."
    )
    folder_path: Optional[str] = Field(
        default=None,
        description="Folder with PDFs. Defaults to DEFAULT_DOCS_DIR."
    )
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP)
    replace: bool = Field(
        default=True,
        description="If True, replaces existing collection with same name."
    )

class QueryRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = Field(default=DEFAULT_TOP_K)
    max_answer_tokens: int = Field(default=400)
    temperature: float = Field(default=0.0)

class CollectionsResponse(BaseModel):
    collections: List[str]

# -----------------------
# Utilities
# -----------------------
def collection_dir(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return VECTOR_DBS_DIR / safe

def load_pdfs_from_folder(folder: Path) -> List[Dict[str, Any]]:
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
    docs = []
    for p in folder.glob("**/*.pdf"):
        try:
            reader = PdfReader(str(p))
            text_pages = []
            for i, page in enumerate(reader.pages):
                try:
                    text_pages.append(page.extract_text() or "")
                except Exception:
                    text_pages.append("")
            full_text = "\n".join(text_pages)
            if full_text.strip():
                docs.append({"path": str(p), "text": full_text})
        except Exception as e:
            # skip corrupt/unreadable PDFs, but continue
            print(f"[WARN] Could not read {p}: {e}")
    if not docs:
        raise HTTPException(status_code=400, detail=f"No readable PDFs in {folder}")
    return docs

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Simple character-based sliding window
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        # trim awkward whitespace
        chunk = " ".join(chunk.split())
        if chunk:
            print(f"Chunk [{start}:{end}] ({len(chunk)} chars)")
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    # Normalize to 8k batch max inputs if needed; here simple batch in one go unless huge
    response = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=texts,
    )
    vecs = np.array([d.embedding for d in response.data], dtype=np.float32)
    # Normalize to unit length for cosine similarity via inner product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    return vecs

def save_faiss_collection(name: str, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
    cdir = collection_dir(name)
    cdir.mkdir(parents=True, exist_ok=True)
    index_path = cdir / "index.faiss"
    meta_path = cdir / "meta.json"

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

def load_faiss_collection(name: str):
    cdir = collection_dir(name)
    index_path = cdir / "index.faiss"
    meta_path = cdir / "meta.json"
    if not index_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    index = faiss.read_index(str(index_path))
    with meta_path.open("r", encoding="utf-8") as f:
        metadatas = json.load(f)
    return index, metadatas

def list_collections() -> List[str]:
    if not VECTOR_DBS_DIR.exists():
        return []
    return [p.name for p in VECTOR_DBS_DIR.iterdir() if (p / "index.faiss").exists()]

def build_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for h in hits:
        src = h.get("source", "unknown.pdf")
        blocks.append(f"[Source: {src}] {h.get('text', '')}")
    return "\n\n".join(blocks)

def answer_with_context(question: str, context: str, max_tokens: int, temperature: float) -> str:
    system_prompt = (
        "You are a precise assistant. Answer the user's question strictly from the provided context. "
        "If the context is insufficient or the answer is not present, say you don't have enough information. "
        "Cite sources in brackets like [Source filename.pdf]. Keep answers concise and factual."
    )
   
    user_prompt = f"Question: {question}\n\nContext:\n{context}"

    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# -----------------------
# Routes
# -----------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/collections", response_model=CollectionsResponse)
def get_collections():
    return CollectionsResponse(collections=list_collections())

@app.post("/ingest")
def ingest(req: IngestRequest):
    folder = Path(req.folder_path) if req.folder_path else DEFAULT_DOCS_DIR
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder}")

    # default collection name = folder name
    collection_name = req.collection_name or folder.name

    cdir = collection_dir(collection_name)
    if req.replace and cdir.exists():
        shutil.rmtree(cdir)

    docs = load_pdfs_from_folder(folder)

    # chunk all docs
    all_chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for d in docs:
        chunks = chunk_text(d["text"], req.chunk_size, req.chunk_overlap)
        for i, ch in enumerate(chunks):
            metadatas.append({
                "id": str(uuid.uuid4()),
                "source": os.path.basename(d["path"]),
                "chunk_index": i,
                "text": ch,
            })
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No text extracted from PDFs to index.")

    # embed & save
    embeddings = embed_texts(all_chunks)
    save_faiss_collection(collection_name, embeddings, metadatas)

    return {
        "collection_name": collection_name,
        "num_documents": len(docs),
        "num_chunks": len(all_chunks),
        "chunk_size": req.chunk_size,
        "chunk_overlap": req.chunk_overlap,
        "vector_dim": int(embeddings.shape[1]),
        "message": f"Collection '{collection_name}' created."
    }

@app.post("/query")
def query(req: QueryRequest):
    index, metadatas = load_faiss_collection(req.collection_name)

    q_vec = embed_texts([req.question])  # normalized
    D, I = index.search(q_vec, req.top_k)
    idxs = I[0].tolist()

    hits = []
    for rank, idx in enumerate(idxs):
        if idx < 0 or idx >= len(metadatas):
            continue
        md = metadatas[idx]
        hits.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "source": md.get("source"),
            "chunk_index": md.get("chunk_index"),
            "text": md.get("text"),
        })

    context = build_context(hits)
    answer = answer_with_context(
        question=req.question,
        context=context,
        max_tokens=req.max_answer_tokens,
        temperature=req.temperature,
    )
    return {
        "collection_name": req.collection_name,
        "top_k": req.top_k,
        "matches": hits,
        "answer": answer
    }
