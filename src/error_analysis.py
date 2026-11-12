"""
src/error_analysis.py
Automated failure case analysis, confusion matrix evaluation, and prediction visualization
for BoardMind AI Whiteboard Detection.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def run_error_analysis(
    model_path='runs/whiteboard_detection_test/weights/best.pt',
    output_dir='runs/whiteboard_optimization/error_analysis',
    val_images_dir='datasets/whiteboard/valid/images',
    val_labels_dir='datasets/whiteboard/valid/labels',
    conf_thresh=0.25,
    iou_thresh=0.5
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    val_imgs = sorted(list(Path(val_images_dir).glob('*.*')))
    labels_dir = Path(val_labels_dir)

    print(f"Analyzing {len(val_imgs)} validation images...")

    tp, fp, fn = 0, 0, 0
    failure_cases = []

    for img_p in val_imgs:
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            content = lbl_p.read_text().strip()
            if content:
                for line in content.splitlines():
                    p = line.split()
                    if len(p) == 5:
                        gt_boxes.append([float(x) for x in p[1:]]) # cx, cy, w, h

        # Predict
        res = model.predict(source=str(img_p), conf=conf_thresh, imgsz=416, device='cpu', verbose=False)[0]
        pred_boxes = res.boxes.xywhn.cpu().numpy() if len(res.boxes) > 0 else []
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else []

        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)

        if num_gt == 0 and num_pred == 0:
            # True negative background
            pass
        elif num_gt == 0 and num_pred > 0:
            # False positive on background
            fp += num_pred
            failure_cases.append({
                'image': img_p.name,
                'type': 'False Positive on Background',
                'conf': float(confs[0]) if len(confs) > 0 else 0,
                'path': str(img_p)
            })
        elif num_gt > 0 and num_pred == 0:
            # False negative (missed whiteboard)
            fn += num_gt
            failure_cases.append({
                'image': img_p.name,
                'type': 'False Negative (Missed Board)',
                'conf': 0.0,
                'path': str(img_p)
            })
        else:
            tp += min(num_gt, num_pred)
            if num_pred > num_gt:
                fp += (num_pred - num_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    summary = {
        'total_images': len(val_imgs),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'num_failure_cases': len(failure_cases)
    }

    # Save summary
    with open(output_path / 'error_summary.json', 'w') as f:
        json.dump({'metrics': summary, 'failures': failure_cases}, f, indent=2)

    print("Error Analysis Complete:")
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == '__main__':
    run_error_analysis()

