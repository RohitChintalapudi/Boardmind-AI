"""
src/recognition/handwriting_recognizer.py
Re-exports BaseHandwritingRecognizer and ModularHandwritingRecognizer
from backend.app.recognition.handwriting_recognizer
"""
from backend.app.recognition.handwriting_recognizer import (
    BaseHandwritingRecognizer,
    ModularHandwritingRecognizer,
    get_handwriting_recognizer
)

__all__ = [
    "BaseHandwritingRecognizer",
    "ModularHandwritingRecognizer",
    "get_handwriting_recognizer"
]
