# RAG & Image Similarity (Minimal)

## Setup
1. **Fill in root `.env`**  
```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
EMBED_TEXT_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_IMAGE_MODEL=clip-ViT-B-32
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4
TEMPERATURE=0.2
RAG_DATA_DIR=rag/data
RAG_VDB_DIR=rag/vectordb
RAG_IMG_DIR=rag/images
RAG_IMG_INDEX_DIR=rag/img_index
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Add data**

Place text files (.pdf, .md, .txt) in: rag/data/
Place images (.jpg, .png, .webp) in: rag/images/

4. **Run scripts**

```bash
python rag/ingest.py
python rag/search.py
uvicorn rag.server:app --reload
```

5. **API Endpoints**
```bash
GET /rag/stream?q=your+question&top_k=4&temperature=0.2
GET /image/search?q=your+question&top_k=4
GET /image/search_by_image?image_path=your/image/path.jpg&top_k=4
GET /image/search_by_image_and_text?image_path=your/image/path.jpg&text_query=your+question&top_k=4
```

6. **Image Similarity**

```bash
python rag/image_index.py
python rag/image_search.py
```

## Quickstart
```bash
pip install -r requirements.txt
python rag/ingest.py
uvicorn rag.server:app --host 0.0.0.0 --port 8000 --reload
```

## Optional: cURL test for SSE endpoint

```bash
curl "http://localhost:8000/rag/stream?q=hello&top_k=4&temperature=0.2"
```