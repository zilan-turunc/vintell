import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from sse_starlette.sse import EventSourceResponse
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_TEXT_MODEL = os.getenv("EMBED_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VDB_DIR = os.getenv("RAG_VDB_DIR", "rag/vectordb")

app = FastAPI()
client = OpenAI()
embedder = SentenceTransformer(EMBED_TEXT_MODEL)
chroma = chromadb.PersistentClient(path=VDB_DIR)
coll = chroma.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

@app.get("/")
def root():
    return {"message": "Vintell RAG API is running. Visit /docs for Swagger UI."}

def retrieve(query, k):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    res = coll.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas"])
    return [{"text": d, "meta": m} for d, m in zip(res["documents"][0], res["metadatas"][0])]

@app.get("/rag/stream")
async def rag_stream(q: str = Query(...), top_k: int = Query(4), temperature: float = Query(0.2)):
    contexts = retrieve(q, top_k)
    context_blob = "\n\n".join([
        f"({c['meta'].get('source', 'source')}#{c['meta'].get('chunk', 0)})\n{c['text']}"
        for c in contexts
    ])
    messages = [
        {"role": "system", "content": "Answer using ONLY provided contexts when relevant. Cite as (source#chunk)."},
        {"role": "user", "content": f"Query: {q}\n\nContexts:\n{context_blob}"}
    ]

    async def event_generator():
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and delta.content:
                yield {"event": "token", "data": delta.content}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())
