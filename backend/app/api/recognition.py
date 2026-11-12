"""
backend/app/api/recognition.py
Handwriting Text Recognition API Endpoint.
"""

import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from backend.app.schemas.models import ExtractTextResponse
from backend.app.recognition.handwriting_recognizer import get_handwriting_recognizer

router = APIRouter(prefix="/api", tags=["Recognition"])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE = 20 * 1024 * 1024

@router.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(file: UploadFile = File(...)):
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

    recognizer = get_handwriting_recognizer()
    response = recognizer.recognize(image)
    return response

