"""
backend/app/api/process.py
End-to-End Processing API Endpoint for BoardMind AI.
"""

import io
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from backend.app.schemas.models import ProcessResponse
from backend.app.services.pipeline_service import get_pipeline_service

router = APIRouter(prefix="/api", tags=["Pipeline"])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE = 20 * 1024 * 1024

@router.post("/process", response_model=ProcessResponse)
async def process_whiteboard(
    file: UploadFile = File(...),
    topic_hint: Optional[str] = Form(None),
    conf_threshold: float = Form(0.25),
    api_key: Optional[str] = Form(None)
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Supported: JPEG, PNG, WEBP."
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size limit of {MAX_FILE_SIZE // (1024*1024)} MB."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    service = get_pipeline_service()
    result = service.process_full_pipeline(
        image=image,
        topic_hint=topic_hint,
        conf_threshold=conf_threshold,
        api_key=api_key
    )

    return result

