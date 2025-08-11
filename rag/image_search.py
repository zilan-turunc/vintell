import os, json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss

# load root .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

OUT_DIR = os.getenv("RAG_IMG_INDEX_DIR", "rag/img_index")
MODEL_NAME = os.getenv("EMBED_IMAGE_MODEL", "clip-ViT-B-32")

def search(image_path, top_k=5):
    index = faiss.read_index(f"{OUT_DIR}/index.faiss")
    metas = json.load(open(f"{OUT_DIR}/meta.json","r",encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)

    img = Image.open(image_path).convert("RGB")
    q = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    D, I = index.search(q.reshape(1,-1), top_k)
    return [{"path": metas[i]["path"], "score": float(D[0][j])} for j, i in enumerate(I[0])]

if __name__ == "__main__":
    p = input("Image path to search: ").strip().strip('"')
    results = search(p, top_k=5)
    print("\nTop matches:")
    for r in results:
        print(f"{r['score']:.3f}  {r['path']}")
