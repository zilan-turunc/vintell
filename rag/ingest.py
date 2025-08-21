import os, glob, uuid
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from markdown_it import MarkdownIt
import chromadb
from sentence_transformers import SentenceTransformer

# load root .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
EMBED_TEXT_MODEL = os.getenv("EMBED_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = os.getenv("RAG_DATA_DIR", "rag/data")
VDB_DIR = os.getenv("RAG_VDB_DIR", "rag/vectordb")

def read_pdf(path):
    reader = PdfReader(path)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def read_md_or_txt(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    # turn markdown into plain text-ish
    return MarkdownIt().render(text)

def chunk_text(t, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    t = " ".join((t or "").split())
    chunks, i = [], 0
    while i < len(t):
        chunk = t[i:i+size]
        if chunk.strip():
            chunks.append(chunk)
        i += max(1, size - overlap)
    return chunks

def main():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.Client(Settings(persist_directory=VDB_DIR))
    coll = client.get_or_create_collection(name="docs")
    embedder = SentenceTransformer(EMBED_TEXT_MODEL)

    docs, metas, ids = [], [], []

    for path in glob.glob(f"{DATA_DIR}/**/*", recursive=True):
        if os.path.isdir(path): continue
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            text = read_pdf(path)
        elif ext in {".md", ".txt"}:
            text = read_md_or_txt(path)
        else:
            continue

        for idx, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            metas.append({"source": os.path.relpath(path), "chunk": idx})
            ids.append(str(uuid.uuid4()))

    if not docs:
        print("No ingestible files in rag/data. Add PDFs/MD/TXT first.")
        return

    embs = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    coll.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
    client.persist()
    print(f"Ingested {len(docs)} chunks into {VDB_DIR}")

if __name__ == "__main__":
    main()