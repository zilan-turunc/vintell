# backend/rag/image_search.py
from dotenv import load_dotenv
import os, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

OUT_DIR = os.getenv("RAG_IMG_INDEX_DIR") 

# IMPORTANT: point to the correct index dir under backend/rag/
OUT_DIR = os.getenv("RAG_IMG_INDEX_DIR", "backend/rag/img_index")
OUT_DIR = "backend/rag/img_index"
MODEL_NAME = os.getenv("EMBED_IMAGE_MODEL", "clip-ViT-B-32")

def search(image_path, top_k=5):
    index_path = Path(OUT_DIR) / "index.faiss"
    meta_path = Path(OUT_DIR) / "meta.json"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index or meta not found in {OUT_DIR}")

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metas = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    img = Image.open(image_path).convert("RGB")
    q = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    D, I = index.search(q.reshape(1, -1), top_k)

    # Return path + score (path can be relative or absolute from your meta.json)
    out = []
    for j, i in enumerate(I[0]):
        if i < 0 or i >= len(metas):
            continue
        out.append({"path": metas[i].get("path", ""), "score": float(D[0][j])})
    return out

if __name__ == "__main__":
    p = input("Image path to search: ").strip().strip('"')
    results = search(p, top_k=5)
    print("\nTop matches:")
    for r in results:
        print(f"{r['score']:.3f}  {r['path']}")
