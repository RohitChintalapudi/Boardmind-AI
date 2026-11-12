"""
backend/app/api/detection.py
Whiteboard Detection API Endpoint.
"""

import io
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from PIL import Image

from backend.app.schemas.models import DetectResponse
from backend.app.detection.whiteboard_detector import get_whiteboard_detector

router = APIRouter(prefix="/api", tags=["Detection"])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

@router.post("/detect", response_model=DetectResponse)
async def detect_whiteboard(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25)
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

    detector = get_whiteboard_detector()
    detections, primary_box, annotated_pil = detector.detect(
        image, conf_threshold=conf_threshold
    )

    annotated_b64 = detector.image_to_base64(annotated_pil)

    return DetectResponse(
        success=True,
        num_detections=len(detections),
        detections=detections,
        primary_detection=primary_box,
        image_width=image.width,
        image_height=image.height,
        annotated_image=annotated_b64,
        message=f"Successfully detected {len(detections)} whiteboard(s)." if detections else "No whiteboard detected with current confidence threshold."
    )

