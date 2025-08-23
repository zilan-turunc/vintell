# backend/image_search_api.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# import your FAISS search from the new location
from backend.rag.image_search import search as vector_search

# -----------------------------
# Config (override with env vars)
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_IMAGES_DIR = Path(os.getenv("PUBLIC_IMAGES_DIR", PROJECT_ROOT / "rag" / "images"))
STATIC_URL_PREFIX = os.getenv("STATIC_URL_PREFIX", "/images/")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", PROJECT_ROOT.parent / "uploads"))
UPLOAD_URL_PREFIX = os.getenv("UPLOAD_URL_PREFIX", "/uploads/")


UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

def _to_web_url(path: str | Path) -> str:
    """Turn stored path (from meta.json) into a URL the browser can fetch."""
    s = str(path).replace("\\", "/")
    if s.startswith("http://") or s.startswith("https://"):
        return s

    p = Path(s)
    # relative path -> make it look like /images/whatever.jpg if it's under images
    if not p.is_absolute():
        # if meta paths are like "backend/rag/images/coat.jpg", strip leading parts
        parts = p.parts
        try:
            # find 'images' in the path and keep from there
            i = parts.index("images")
            rel = "/".join(parts[i+1:])  # after 'images'
            return (STATIC_URL_PREFIX.rstrip("/") + "/" + rel).replace("//", "/")
        except ValueError:
            # no 'images' component, just prefix
            return (STATIC_URL_PREFIX.rstrip("/") + "/" + p.name).replace("//", "/")

    # absolute file path under PUBLIC_IMAGES_DIR
    try:
        abs_p = p.resolve()
    except Exception:
        abs_p = p
    try:
        pub_root = PUBLIC_IMAGES_DIR.resolve()
    except Exception:
        pub_root = PUBLIC_IMAGES_DIR

    if abs_p.is_file() and str(abs_p).startswith(str(pub_root)):
        rel = abs_p.relative_to(pub_root).as_posix()
        return (STATIC_URL_PREFIX.rstrip("/") + "/" + rel).replace("//", "/")

    # fallback: just return filename under /images/
    return (STATIC_URL_PREFIX.rstrip("/") + "/" + abs_p.name).replace("//", "/")

def _save_upload(file: UploadFile) -> Path:
    """Save uploaded query image."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    suffix = os.path.splitext(file.filename)[1] or ".png"
    dest = (UPLOAD_DIR / f"query_{os.getpid()}_{file.filename}").with_suffix(suffix)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest

@router.post("/image-search")
async def image_search_endpoint(image_file: UploadFile = File(...), top_k: int = 8) -> JSONResponse:
    """
    Accepts an image upload, runs similarity search, returns JSON:
    {
      "query_image_url": "/uploads/query_xxx.png",
      "results": [
        {"image_url": "/images/item1.jpg", "name": "item1", "score": 0.93},
        ...
      ]
    }
    """
    saved = _save_upload(image_file)
    try:
        raw_results = vector_search(str(saved), top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    normalized: List[Dict[str, Any]] = []
    for r in raw_results:
        path = r.get("path") or r.get("image_path") or r.get("image") or ""
        url = _to_web_url(path)
        name = os.path.splitext(os.path.basename(str(path) or ""))[0].replace("_", " ").strip() or "match"
        normalized.append({
            "image_url": url,
            "name": name,
            "score": float(r.get("score", 0.0))
        })

    payload = {
        "query_image_url": (UPLOAD_URL_PREFIX.rstrip("/") + "/" + saved.name),
        "results": normalized
    }
    return JSONResponse(payload)
