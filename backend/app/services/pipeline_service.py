"""
backend/app/services/pipeline_service.py
End-to-End Orchestrator for BoardMind AI Whiteboard Processing.
Integrates:
Detection -> Cropping -> Preprocessing -> HTR -> Note Generation -> AI Concept Explainer.
"""

import time
from typing import Tuple, List, Optional
from PIL import Image

from backend.app.detection.whiteboard_detector import get_whiteboard_detector
from backend.app.processing.image_preprocessor import get_image_preprocessor
from backend.app.recognition.handwriting_recognizer import get_handwriting_recognizer
from backend.app.processing.text_formatter import get_text_formatter
from backend.app.processing.ai_explainer import get_ai_concept_explainer
from backend.app.schemas.models import (
    DetectResponse,
    ExtractTextResponse,
    StructuredNotes,
    ProcessResponse
)


class PipelineService:
    def __init__(self):
        self.detector = get_whiteboard_detector()
        self.preprocessor = get_image_preprocessor()
        self.recognizer = get_handwriting_recognizer()
        self.formatter = get_text_formatter()
        self.ai_explainer = get_ai_concept_explainer()

    def process_full_pipeline(
        self,
        image: Image.Image,
        topic_hint: Optional[str] = None,
        conf_threshold: float = 0.25,
        api_key: Optional[str] = None
    ) -> ProcessResponse:
        """
        Executes the complete BoardMind AI workflow.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []
        orig_w, orig_h = image.size

        # 1. Whiteboard Detection
        detections, primary_box, annotated_pil = self.detector.detect(
            image, conf_threshold=conf_threshold
        )
        annotated_b64 = self.detector.image_to_base64(annotated_pil)

        detect_resp = DetectResponse(
            success=True,
            num_detections=len(detections),
            detections=detections,
            primary_detection=primary_box,
            image_width=orig_w,
            image_height=orig_h,
            annotated_image=annotated_b64,
            message=f"Detected {len(detections)} whiteboard region(s)." if detections else "No whiteboard detected. Using full image frame."
        )

        # 2. Crop & Preprocessing
        if primary_box:
            cropped_pil = self.preprocessor.crop_whiteboard(image, primary_box)
        else:
            cropped_pil = image.copy()
            warnings.append("No whiteboard bounding box identified. Processing the entire image frame.")

        enhanced_pil, stroke_density = self.preprocessor.enhance_handwriting(cropped_pil)
        crop_b64 = self.preprocessor.image_to_base64(enhanced_pil)

        # 3. Handwriting Recognition
        extract_resp = self.recognizer.recognize(enhanced_pil)
        extract_resp.stroke_density = stroke_density

        if extract_resp.status == "untrained_baseline":
            warnings.append("HTR Model Status: Baseline interface active. Handwriting line detection succeeded, but full text transcription requires a trained HTR checkpoint.")

        # 4. Note Structuring
        structured_notes = self.formatter.format_notes(
            raw_text=extract_resp.raw_text,
            topic_hint=topic_hint
        )

        # 5. AI Concept Knowledge & Lecture Explanation Layer
        ai_res = self.ai_explainer.explain(
            image=enhanced_pil,
            raw_text=extract_resp.raw_text,
            topic_hint=topic_hint or structured_notes.title,
            api_key=api_key
        )

        explanation_md = ai_res.get("explanation_markdown", "")
        engine_used = ai_res.get("engine", "BoardMind AI")
        structured_notes.ai_concept_explanation = explanation_md
        structured_notes.ai_engine_used = engine_used

        # Elevate markdown with comprehensive AI lecture breakdown
        if explanation_md:
            structured_notes.markdown_content = explanation_md
            if ai_res.get("title") and (not topic_hint or structured_notes.title.isdigit() or len(structured_notes.title) < 5):
                structured_notes.title = ai_res.get("title")

        end_time = time.perf_counter()
        duration_ms = round((end_time - start_time) * 1000.0, 2)

        return ProcessResponse(
            success=True,
            detection=detect_resp,
            cropped_whiteboard=crop_b64,
            text_extraction=extract_resp,
            structured_notes=structured_notes,
            ai_concept_explanation=explanation_md,
            processing_time_ms=duration_ms,
            warnings=warnings
        )


_pipeline_service_instance = None

def get_pipeline_service() -> PipelineService:
    global _pipeline_service_instance
    if _pipeline_service_instance is None:
        _pipeline_service_instance = PipelineService()
    return _pipeline_service_instance
