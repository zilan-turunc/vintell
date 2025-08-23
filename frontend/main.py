# frontend/main.py
from fastapi import FastAPI, Form, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv
import os
import httpx  # pip install httpx

# --- Create app FIRST ---
app = FastAPI()

# --- Mount static and include JSON image-search BEFORE defining other routes ---
from backend.image_search_api import router as image_search_router
app.mount("/images", StaticFiles(directory="backend/rag/images"), name="images")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.include_router(image_search_router)

# --- Resolve paths / templates ---
HERE = Path(__file__).resolve()
FRONTEND_DIR = HERE.parent
TEMPLATES_DIR = FRONTEND_DIR / "templates"
PROJECT_ROOT = FRONTEND_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Backend features ---
from backend.features.text_to_fashion import suggest_fashion_items
from backend.agents.moodboard_agent import MoodboardAgent

# If you STILL had a legacy HTML /image-search route here, REMOVE it
# (Your new JSON /image-search is already included via router above.)

# --- RAG API Configuration ---
RAG_API_URL = "http://127.0.0.1:8000"

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

# --- RAG Streaming Proxy Endpoint ---
@app.get("/rag-stream-proxy")
async def rag_stream_proxy(q: str = Query(...)):
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

# --- Text→Fashion ---
@app.post("/text-to-fashion", response_class=HTMLResponse)
async def text_to_fashion(description: str = Form(...)):
    try:
        suggestions = suggest_fashion_items(description)
        body = f"<h2>Suggested Items:</h2><pre>{suggestions}</pre>"
    except Exception as e:
        body = f"<h2>Oops</h2><pre>{e}</pre>"
    return body + "<a href='/'>Back</a>"

# --- Moodboard Tags ---
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
