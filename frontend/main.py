# frontend/main.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from dotenv import load_dotenv
import os

# --- Resolve paths ---
HERE = Path(__file__).resolve()
FRONTEND_DIR = HERE.parent
TEMPLATES_DIR = FRONTEND_DIR / "templates"
PROJECT_ROOT = FRONTEND_DIR.parent

# Load .env from project root (…/.env)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# FastAPI + Jinja
app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Imports from backend (needs backend to be a package) ---
# If you get import errors, add empty __init__.py files under:
# backend/, backend/features/, backend/agents/
from backend.features.text_to_fashion import suggest_fashion_items
from backend.agents.moodboard_agent import MoodboardAgent


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


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
