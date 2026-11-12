"""
backend/app/detection/whiteboard_detector.py
Whiteboard Detection Service using Ultralytics YOLO on CPU.
Loads champion model once at initialization.
"""

import io
import base64
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

from backend.app.schemas.models import BoundingBox

class WhiteboardDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.device = 'cpu'
        self.model_path = self._resolve_model_path(model_path)
        print(f"[Detector] Initializing WhiteboardDetector with weights: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: 'Whiteboard'}
        print(f"[Detector] WhiteboardDetector successfully loaded on {self.device}. Classes: {self.class_names}")

    def _resolve_model_path(self, user_path: Optional[str]) -> Path:
        if user_path and Path(user_path).exists():
            return Path(user_path)

        candidates = [
            Path("models/whiteboard_detector_best.pt"),
            Path("runs/whiteboard_optimization/exp1_aug_perspective_scale/weights/best.pt"),
            Path("runs/whiteboard_detection_test/weights/best.pt"),
            Path("yolov8n.pt")
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        raise FileNotFoundError("No trained whiteboard detection checkpoint found.")

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.6
    ) -> Tuple[List[BoundingBox], Optional[BoundingBox], Image.Image]:
        """
        Runs whiteboard detection inference on a PIL image.
        Returns:
            - List of all detected BoundingBox objects
            - Primary BoundingBox (most salient whiteboard)
            - Annotated PIL Image with visual bounding boxes
        """
        orig_w, orig_h = image.size
        # Run inference on CPU
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            imgsz=416,
            verbose=False
        )[0]

        detections: List[BoundingBox] = []
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i].tolist()
                conf = float(confs[i])
                cls_id = int(clss[i])
                cls_name = self.class_names.get(cls_id, "Whiteboard")

                # Clamp to image boundaries
                xmin = max(0.0, min(xmin, float(orig_w)))
                ymin = max(0.0, min(ymin, float(orig_h)))
                xmax = max(0.0, min(xmax, float(orig_w)))
                ymax = max(0.0, min(ymax, float(orig_h)))

                norm_box = [
                    round(xmin / orig_w, 4),
                    round(ymin / orig_h, 4),
                    round(xmax / orig_w, 4),
                    round(ymax / orig_h, 4)
                ]

                bbox = BoundingBox(
                    xmin=round(xmin, 1),
                    ymin=round(ymin, 1),
                    xmax=round(xmax, 1),
                    ymax=round(ymax, 1),
                    normalized=norm_box,
                    confidence=round(conf, 4),
                    class_id=cls_id,
                    class_name=cls_name
                )
                detections.append(bbox)

                # Draw high-contrast bounding box on image
                draw.rectangle([xmin, ymin, xmax, ymax], outline="#0284c7", width=4)
                # Label banner
                label = f"{cls_name} {conf:.1%}"
                text_bbox = draw.textbbox((xmin + 4, max(0.0, ymin - 20)), label)
                draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2], fill="#0284c7")
                draw.text((xmin + 4, max(0.0, ymin - 20)), label, fill="white")

        # Select primary detection (prioritize confidence and size)
        primary_box = None
        if detections:
            # Sort by area * confidence
            def score(b: BoundingBox):
                area = (b.xmax - b.xmin) * (b.ymax - b.ymin)
                return area * (b.confidence ** 2)
            primary_box = max(detections, key=score)

        return detections, primary_box, annotated_image

    def image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        buffered = io.BytesIO()
        if format.upper() == "JPEG" and image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format=format, quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"

# Global singleton detector instance
_detector_instance: Optional[WhiteboardDetector] = None

def get_whiteboard_detector() -> WhiteboardDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = WhiteboardDetector()
    return _detector_instance

