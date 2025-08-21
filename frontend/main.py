# frontend/main.py
from fastapi import FastAPI, Form, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from dotenv import load_dotenv
import os
import httpx # Make sure to 'pip install httpx'

# --- Resolve paths ---
HERE = Path(__file__).resolve()
FRONTEND_DIR = HERE.parent
TEMPLATES_DIR = FRONTEND_DIR / "templates"
PROJECT_ROOT = FRONTEND_DIR.parent

# Load .env from project root (…/.env)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# --- FastAPI + Jinja ---
app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Backend Imports ---
from backend.features.text_to_fashion import suggest_fashion_items
from backend.agents.moodboard_agent import MoodboardAgent
# Import the image search function from your RAG service file
from backend.rag.rag_service import image_search as find_similar_images

# --- RAG API Configuration ---
# The URL for your separate RAG backend server (from rag/server.py)
RAG_API_URL = "http://127.0.0.1:8000"


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

# --- RAG Streaming Proxy Endpoint ---
@app.get("/rag-stream-proxy")
async def rag_stream_proxy(q: str = Query(...)):
    """
    This endpoint acts as a proxy to the RAG backend stream.
    """
    async def stream_generator():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", f"{RAG_API_URL}/rag/stream", params={"q": q}) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.ConnectError as e:
            error_message = f"event: error\ndata: Cannot connect to RAG service: {e}\n\n"
            yield error_message.encode('utf-8')

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# --- Image Search Endpoint ---
@app.post("/image-search", response_class=HTMLResponse)
async def image_search(image_file: UploadFile = File(...)):
    try:
        image_bytes = await image_file.read()
        similar_item_titles = find_similar_images(image_bytes=image_bytes)
        
        if similar_item_titles:
            results_html = "<ul>" + "".join(f"<li>{title}</li>" for title in similar_item_titles) + "</ul>"
            body = f"<h2>Similar Items Found:</h2>{results_html}"
        else:
            body = "<h2>No similar items found.</h2>"

    except Exception as e:
        body = f"<h2>Oops, an error occurred during image search:</h2><pre>{e}</pre>"
    return body + "<br><a href='/'>Back</a>"


@app.post("/text-to-fashion", response_class=HTMLResponse)
async def text_to_fashion(description: str = Form(...)):
    try:
        suggestions = suggest_fashion_items(description)
        body = f"<h2>Suggested Items:</h2><pre>{suggestions}</pre>"
    except Exception as e:
        body = f"<h2>Oops</h2><pre>{e}</pre>"
    return body + "<a href='/'>Back</a>"


@app.post("/moodboard-tags", response_class=HTMLResponse)
async def moodboard_tags(style_description: str = Form(...)):
    try:
        agent = MoodboardAgent()
        tags = agent.run(style_description)
        body = f"<h2>Suggested Hashtags:</h2><pre>{tags}</pre>"
    except Exception as e:
        body = f"<h2>Oops</h2><pre>{e}</pre>"
    return body + "<a href='/'>Back</a>"


@app.get("/health", response_class=HTMLResponse)
async def health():
    return "ok"