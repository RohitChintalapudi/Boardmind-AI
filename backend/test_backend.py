"""
backend/test_backend.py
Comprehensive automated test suite for BoardMind AI backend services and endpoints.
"""

import sys
import io
import json
import asyncio
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.schemas.models import GenerateNotesRequest
from backend.app.api.health import get_health
from backend.app.api.notes import generate_notes
from backend.app.services.pipeline_service import get_pipeline_service
from backend.app.detection.whiteboard_detector import get_whiteboard_detector
from backend.app.recognition.handwriting_recognizer import get_handwriting_recognizer

async def run_async_tests():
    print("=" * 70, flush=True)
    print("TESTING BOARDMIND AI BACKEND SERVICES & PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Health check
    print("\n[1] Testing get_health()...", flush=True)
    health = await get_health()
    print(f"Health response: {health.status}", flush=True)
    print(f"Detector weights: {health.whiteboard_detector['weights_path']}", flush=True)
    print(f"HTR status: {health.handwriting_recognizer['status']}", flush=True)
    assert health.status == "healthy"

    # 2. Detector test
    sample_img_path = Path("notebooks/board-srmap.jpg")
    assert sample_img_path.exists(), f"{sample_img_path} not found"
    img = Image.open(sample_img_path).convert("RGB")
    
    print(f"\n[2] Testing WhiteboardDetector on {sample_img_path}...", flush=True)
    detector = get_whiteboard_detector()
    detections, primary_box, annotated_img = detector.detect(img)
    print(f"Num detections: {len(detections)}", flush=True)
    if primary_box:
        print(f"Primary Box: [{primary_box.xmin}, {primary_box.ymin}, {primary_box.xmax}, {primary_box.ymax}] conf: {primary_box.confidence}", flush=True)
    assert len(detections) >= 1, "Expected at least 1 whiteboard detection"

    # 3. Handwriting recognizer test
    print("\n[3] Testing HandwritingRecognizer...", flush=True)
    recognizer = get_handwriting_recognizer()
    rec_res = recognizer.recognize(img)
    print(f"Recognizer model: {rec_res.model_name}", flush=True)
    print(f"Recognizer status: {rec_res.status}", flush=True)
    print(f"Stroke density: {rec_res.stroke_density}%", flush=True)
    print(f"Detected line regions: {len(rec_res.lines)}", flush=True)
    assert rec_res.success is True

    # 4. Text formatter test
    print("\n[4] Testing TextFormatter (Smart Notes)...", flush=True)
    sample_text = """Binary Search Trees
A binary search tree is defined as a node-based binary tree data structure.
Properties:
1. The left subtree contains only nodes with keys lesser than the node's key.
2. The right subtree contains only nodes with keys greater than the node's key.
Time Complexity: O(log n)
Formula: h = floor(log2(n))
Summary:
- Efficient search, insertion, and deletion
- In-order traversal produces sorted keys"""

    req = GenerateNotesRequest(raw_text=sample_text, topic_hint="Binary Search Trees (BST)")
    notes_resp = await generate_notes(req)
    notes = notes_resp.notes
    print(f"Title: {notes.title}", flush=True)
    print(f"Definitions found: {len(notes.definitions)}", flush=True)
    print(f"Steps/Properties found: {len(notes.numbered_steps)}", flush=True)
    print(f"Formulas found: {len(notes.formulas_or_key_terms)}", flush=True)
    print(f"Bullets found: {len(notes.bullet_points)}", flush=True)
    assert len(notes.definitions) >= 1
    assert len(notes.numbered_steps) == 2
    assert len(notes.formulas_or_key_terms) >= 1

    # 5. Full PipelineService test
    print("\n[5] Testing Full PipelineService...", flush=True)
    pipeline = get_pipeline_service()
    res = pipeline.process_full_pipeline(img, topic_hint="Classroom Demo")
    print(f"Pipeline Success: {res.success}", flush=True)
    print(f"Processing time: {res.processing_time_ms} ms", flush=True)
    print(f"Detection num: {res.detection.num_detections}", flush=True)
    print(f"Whiteboard crop present: {bool(res.cropped_whiteboard)}", flush=True)
    print(f"Text Extraction Status: {res.text_extraction.status}", flush=True)
    print(f"Structured Notes Title: {res.structured_notes.title}", flush=True)
    print(f"Warnings reported: {res.warnings}", flush=True)
    assert res.success is True
    assert res.detection.num_detections >= 1
    assert bool(res.cropped_whiteboard) is True

    print("\n" + "=" * 70, flush=True)
    print("ALL BACKEND COMPONENT & PIPELINE TESTS PASSED!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    asyncio.run(run_async_tests())
