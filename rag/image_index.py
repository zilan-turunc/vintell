import os, glob, json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss

# load root .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

IMG_DIR = os.getenv("RAG_IMG_DIR", "rag/images")
OUT_DIR = os.getenv("RAG_IMG_INDEX_DIR", "rag/img_index")
MODEL_NAME = os.getenv("EMBED_IMAGE_MODEL", "clip-ViT-B-32")

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    paths = [p for p in glob.glob(f"{IMG_DIR}/**/*", recursive=True)
             if Path(p).suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    if not paths:
        print("No images in rag/images")
        return

    model = SentenceTransformer(MODEL_NAME)
    vecs, metas = [], []

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            emb = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
            vecs.append(emb)
            metas.append({"path": os.path.relpath(p)})
        except Exception:
            continue

    vecs = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, f"{OUT_DIR}/index.faiss")
    with open(f"{OUT_DIR}/meta.json","w",encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"Indexed {len(metas)} images → {OUT_DIR}")

if __name__ == "__main__":
    main()
