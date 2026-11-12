import sys
from pathlib import Path
print("1. Python started", flush=True)

import backend.app.schemas.models
print("2. Models imported", flush=True)

from backend.app.detection.whiteboard_detector import get_whiteboard_detector
print("3. Detector imported", flush=True)
d = get_whiteboard_detector()
print("4. Detector loaded", flush=True)

from PIL import Image
img = Image.open("notebooks/board-srmap.jpg")
print("5. Image opened", flush=True)
boxes, primary, ann = d.detect(img)
print(f"6. Detected {len(boxes)} boxes", flush=True)

