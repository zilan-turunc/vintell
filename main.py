import subprocess
import time

# Start backend on port 8000
# The correct file is server.py, not rag_service.py
print("--- Starting RAG Backend Server on port 8000 ---")
backend = subprocess.Popen([
    "uvicorn", "backend.rag.server:app", "--reload", "--port", "8000"
])

# Give backend a moment to spin up before starting frontend
print("Waiting for backend to initialize...")
time.sleep(5) # Giving it a few seconds to load models

# Start frontend on port 8001 (blocks until stopped)
print("\n--- Starting Frontend Server on port 8001 ---")
# Use subprocess.run for the frontend so the script waits here
frontend_process = subprocess.run([
    "uvicorn", "frontend.main:app", "--reload", "--port", "8001"
])

# Clean up backend when frontend stops (when you press Ctrl+C)
print("\n--- Frontend stopped, terminating backend. ---")
backend.terminate()