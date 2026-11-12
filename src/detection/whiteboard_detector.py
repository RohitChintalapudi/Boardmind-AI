"""
src/detection/whiteboard_detector.py
Re-exports WhiteboardDetector from backend.app.detection.whiteboard_detector
"""
from backend.app.detection.whiteboard_detector import (
    WhiteboardDetector,
    get_whiteboard_detector
)

__all__ = ["WhiteboardDetector", "get_whiteboard_detector"]
