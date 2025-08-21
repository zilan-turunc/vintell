import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# load root .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
EMBED_TEXT_MODEL = os.getenv("EMBED_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VDB_DIR = os.getenv("RAG_VDB_DIR", "rag/vectordb")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "docs")

client = OpenAI()
embedder = SentenceTransformer(EMBED_TEXT_MODEL)

# Use PersistentClient in chromadb 0.5.x; persistence is automatic
chroma = chromadb.PersistentClient(path=VDB_DIR)
coll = chroma.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def retrieve(query, k=TOP_K):
    # Ensure we pass a plain Python list of floats to Chroma
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for d, m in zip(docs, metas):
        out.append({"text": d, "meta": m})
    return out

def build_payload(user_query, params, contexts):
    return {
        "query": user_query,
        "params": params,
        "contexts": [
            {
                "source": c["meta"].get("source", "unknown"),
                "chunk": c["meta"].get("chunk", 0),
                "text": c["text"],
            }
            for c in contexts
        ],
    }

def stream_answer(payload):
    system = "Answer using ONLY provided contexts when relevant. Cite as (source#chunk)."
    if payload["contexts"]:
        context_blob = "\n\n".join(
            [f"({c['source']}#{c['chunk']})\n{c['text']}" for c in payload["contexts"]]
        )
    else:
        context_blob = "(no contexts available)"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {payload['query']}\n\nContexts:\n{context_blob}"},
    ]

    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=float(payload["params"].get("temperature", TEMPERATURE)),
        stream=True,
    )

    for chunk in stream:
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if delta and getattr(delta, "content", None):
            print(delta.content, end="", flush=True)
    print()

if __name__ == "__main__":
    q = input("Your query: ").strip()
    ctx = retrieve(q, k=TOP_K)
    payload = build_payload(q, {"top_k": TOP_K, "temperature": TEMPERATURE}, ctx)
    print("---- streaming answer ----")
    stream_answer(payload)

