# backend/rag/rag_service.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import csv
import faiss
import numpy as np
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer

# ---- paths ----
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "fashion_items.csv"
INDEX_DIR = DATA_DIR / "rag_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_FILE = INDEX_DIR / "clip_text.index"
META_FILE = INDEX_DIR / "meta.tsv"

# ---- model: CLIP (text & image in same space) ----
_model = None
def model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("clip-ViT-B-32")
    return _model

_items: List[Tuple[str, str, str]] = []   # (id, title, desc)
_index = None

def _load_csv() -> List[Tuple[str, str, str]]:
    items = []
    if not CSV_PATH.exists():
        return items
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            iid = (row.get("id") or "").strip()
            title = (row.get("title") or "").strip()
            desc = (row.get("desc") or "").strip()
            if title:
                items.append((iid, title, desc))
    return items

def _save_meta(items: List[Tuple[str, str, str]]):
    with open(META_FILE, "w", encoding="utf-8") as f:
        f.write("id\ttitle\tdesc\n")
        for iid, t, d in items:
            f.write(f"{iid}\t{t}\t{d}\n")

def _load_meta() -> List[Tuple[str, str, str]]:
    if not META_FILE.exists():
        return []
    lines = META_FILE.read_text(encoding="utf-8").splitlines()[1:]
    out = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) == 3:
            out.append((parts[0], parts[1], parts[2]))
    return out

def _build_index(items: List[Tuple[str, str, str]]):
    texts = [f"{title}. {desc}".strip() for _, title, desc in items]
    embs = model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype("float32"))
    return index

def _save_index(index):
    out_path = FAISS_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))

def _load_index():
    if not FAISS_FILE.exists():
        return None
    return faiss.read_index(str(FAISS_FILE))

def _ensure_ready():
    global _items, _index
    if _index is not None:
        return
    idx = _load_index()
    meta = _load_meta()
    if idx and meta:
        _items, _index = meta, idx
        return
    items = _load_csv()
    if not items:
        _items, _index = [], None
        return
    idx = _build_index(items)
    _save_index(idx)
    _save_meta(items)
    _items, _index = items, idx

def _search(vec: np.ndarray, k: int = 8) -> List[int]:
    _ensure_ready()
    if _index is None or not _items:
        return []
    if vec.ndim == 1:
        vec = vec[None, :]
    D, I = _index.search(vec.astype("float32"), k)
    return [int(i) for i in I[0] if i != -1]

# -------- public api used by your routes --------
def text_search(query: str, k: int = 8) -> List[str]:
    _ensure_ready()
    if not query.strip() or _index is None:
        return []
    q = model().encode([query], convert_to_numpy=True, normalize_embeddings=True)
    ids = _search(q[0], k=k)
    return [_items[i][1] for i in ids]  # return titles for now

def image_search(image_bytes: bytes, k: int = 8) -> List[str]:
    _ensure_ready()
    if _index is None:
        return []
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return []
    q = model().encode([img], convert_to_numpy=True, normalize_embeddings=True)
    ids = _search(q[0], k=k)
    return [_items[i][1] for i in ids]  # titles
