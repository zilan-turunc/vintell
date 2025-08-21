import subprocess
import time

# Start backend
backend = subprocess.Popen([
    "uvicorn", "backend.rag.rag_service:app", "--reload"
])

# Give backend a moment to spin up before starting frontend
time.sleep(2)

# Start frontend (blocks until stopped)
subprocess.run([
    "uvicorn", "frontend.main:app", "--reload"
])

# Clean up backend when frontend stops
backend.terminate()
