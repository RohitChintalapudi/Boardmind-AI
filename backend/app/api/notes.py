"""
backend/app/api/notes.py
Smart Note Generation and Formatting API Endpoint with AI Concept Explainer.
"""

from fastapi import APIRouter
from backend.app.schemas.models import GenerateNotesRequest, GenerateNotesResponse
from backend.app.processing.text_formatter import get_text_formatter
from backend.app.processing.ai_explainer import get_ai_concept_explainer

router = APIRouter(prefix="/api", tags=["Notes"])

@router.post("/generate-notes", response_model=GenerateNotesResponse)
async def generate_notes(request: GenerateNotesRequest):
    formatter = get_text_formatter()
    notes = formatter.format_notes(
        raw_text=request.raw_text,
        topic_hint=request.topic_hint
    )

    explainer = get_ai_concept_explainer()
    ai_res = explainer.explain(
        image=None,
        raw_text=request.raw_text,
        topic_hint=request.topic_hint or notes.title,
        api_key=request.api_key
    )

    if ai_res.get("explanation_markdown"):
        notes.ai_concept_explanation = ai_res["explanation_markdown"]
        notes.ai_engine_used = ai_res.get("engine")
        notes.markdown_content = ai_res["explanation_markdown"]
        if ai_res.get("title") and (not request.topic_hint or notes.title.isdigit() or len(notes.title) < 5):
            notes.title = ai_res.get("title")

    return GenerateNotesResponse(
        success=True,
        notes=notes,
        message="Structured study notes and concept explanation generated successfully."
    )
