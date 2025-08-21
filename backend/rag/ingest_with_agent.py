import os
import csv
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from moodboard_agent import MoodboardAgent

# Load .env and model
load_dotenv()
VDB_DIR = os.getenv("RAG_VDB_DIR", "rag/vectordb")
EMBED_MODEL = os.getenv("EMBED_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "docs"

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=VDB_DIR)
coll = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

agent = MoodboardAgent()
csv_path = "data/fashion_items.csv"

with open(csv_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        item_id = row["id"]
        title = row["title"]
        desc = row["desc"]

        print(f"[→] Enriching: {title}")
        metadata_text = agent.run(desc)
        document = f"# {title}\n\n{metadata_text.strip()}"
        embedding = embedder.encode([document], normalize_embeddings=True).tolist()[0]

        coll.upsert(
            documents=[document],
            ids=[item_id],
            metadatas=[{"title": title}]
        )

print("\n✅ CSV-based ingestion complete. All items indexed into ChromaDB.")
