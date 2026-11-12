"""
backend/app/main.py
Main FastAPI Application for BoardMind AI.
"""

import sys
from pathlib import Path

# Add project root and backend dir to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
backend_dir = Path(__file__).resolve().parent.parent
for p in [str(root_dir), str(backend_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.health import router as health_router
from backend.app.api.detection import router as detection_router
from backend.app.api.recognition import router as recognition_router
from backend.app.api.notes import router as notes_router
from backend.app.api.process import router as process_router

from backend.app.detection.whiteboard_detector import get_whiteboard_detector
from backend.app.recognition.handwriting_recognizer import get_handwriting_recognizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-load detector and recognizer on CPU
    print("[Startup] Initializing BoardMind AI ML engines...")
    try:
        _ = get_whiteboard_detector()
        _ = get_handwriting_recognizer()
        print("[Startup] All models initialized successfully on CPU.")
    except Exception as e:
        print(f"[Startup Warning] Model initialization encountered an issue: {e}")
    yield
    # Shutdown
    print("[Shutdown] Cleaning up resources.")

app = FastAPI(
    title="BoardMind AI — Whiteboard to Smart Notes API",
    description="Educational AI platform that converts classroom whiteboard photographs into clean, readable, organized study notes.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(health_router)
app.include_router(detection_router)
app.include_router(recognition_router)
app.include_router(notes_router)
app.include_router(process_router)

@app.get("/", tags=["Root"])
async def root():
    return JSONResponse({
        "product": "BoardMind AI",
        "tagline": "From Whiteboard Scribbles to Smart Notes",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/api/health"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8001, reload=True)
