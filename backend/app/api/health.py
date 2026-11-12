"""
backend/app/api/health.py
Health and System Status API Endpoint.
"""

import platform
import psutil
from fastapi import APIRouter
from backend.app.schemas.models import HealthResponse
from backend.app.detection.whiteboard_detector import get_whiteboard_detector
from backend.app.recognition.handwriting_recognizer import get_handwriting_recognizer

router = APIRouter(prefix="/api", tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def get_health():
    detector = get_whiteboard_detector()
    recognizer = get_handwriting_recognizer()

    return HealthResponse(
        status="healthy",
        app_name="BoardMind AI",
        version="1.0.0",
        whiteboard_detector={
            "loaded": True,
            "device": detector.device,
            "weights_path": str(detector.model_path),
            "classes": detector.class_names
        },
        handwriting_recognizer=recognizer.get_metadata(),
        hardware={
            "os": platform.system(),
            "cpu_architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
            "ram_used_percent": psutil.virtual_memory().percent
        }
    )

