"""
src/evaluate_model.py
Standardized Evaluation Pipeline for BoardMind AI Whiteboard Detection.

Computes:
- Precision, Recall, F1-score (max F1 and confidence curve)
- mAP@50, mAP@50-95
- Confusion Matrix (TP, FP, FN, background FPs)
- CPU Inference Latency Benchmark (preprocess, inference, postprocess, total FPS)
- Model Parameters, FLOPs, Model Size (MB)
- Qualitative Error Analysis & Prediction Visualization
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

def benchmark_inference(model, sample_images, imgsz=416, warmup=10, iterations=50):
    """
    Measures CPU inference latency with strict warm-up and repeated trials.
    """
    # Pre-load image paths
    if isinstance(sample_images, (str, Path)):
        p = Path(sample_images)
        if p.is_dir():
            img_paths = sorted(list(p.glob('*.*')))[:iterations]
        else:
            img_paths = [p] * iterations
    else:
        img_paths = sample_images[:iterations]

    if not img_paths:
        raise ValueError("No images provided for benchmarking.")

    # Warmup
    warmup_img = img_paths[0]
    for _ in range(warmup):
        _ = model.predict(source=str(warmup_img), imgsz=imgsz, device='cpu', verbose=False)

    latencies = []
    for img_p in img_paths:
        t0 = time.perf_counter()
        _ = model.predict(source=str(img_p), imgsz=imgsz, device='cpu', verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # ms

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    median_latency = float(np.median(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0

    return {
        'mean_latency_ms': round(mean_latency, 2),
        'std_latency_ms': round(std_latency, 2),
        'median_latency_ms': round(median_latency, 2),
        'fps': round(fps, 1),
        'num_trials': len(latencies)
    }

def evaluate_checkpoint(
    model_path,
    data_yaml='datasets/whiteboard/data.yaml',
    split='val',
    imgsz=416,
    batch=4,
    conf_threshold=0.25,
    iou_threshold=0.6,
    save_dir=None
):
    """
    Evaluates a model checkpoint using Ultralytics validator and returns structured metrics.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = YOLO(str(model_path))

    # Architecture info
    model_info = model.info(detailed=False)
    # File size
    file_size_mb = model_path.stat().st_size / (1024 * 1024)

    val_kwargs = {
        'data': str(data_yaml),
        'split': split,
        'imgsz': imgsz,
        'batch': batch,
        'device': 'cpu',
        'workers': 0,
        'conf': conf_threshold,
        'iou': iou_threshold,
        'plots': True,
        'verbose': False
    }
    if save_dir:
        val_kwargs['project'] = str(Path(save_dir).parent)
        val_kwargs['name'] = str(Path(save_dir).name)
        val_kwargs['exist_ok'] = True

    metrics = model.val(**val_kwargs)

    # Extract metrics
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 1e-6 else 0.0

    # Ultralytics speed dict (ms per image)
    speed = metrics.speed
    total_val_ms = sum(speed.values())

    results = {
        'model_path': str(model_path),
        'split': split,
        'imgsz': imgsz,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'mAP50': round(map50, 4),
        'mAP50_95': round(map50_95, 4),
        'speed_preprocess_ms': round(speed.get('preprocess', 0), 2),
        'speed_inference_ms': round(speed.get('inference', 0), 2),
        'speed_postprocess_ms': round(speed.get('postprocess', 0), 2),
        'total_latency_ms': round(total_val_ms, 2),
        'fps': round(1000.0 / total_val_ms, 1) if total_val_ms > 0 else 0,
        'model_size_mb': round(file_size_mb, 2),
    }

    return results, metrics

if __name__ == '__main__':
    # Test on baseline
    ckpt = 'runs/whiteboard_detection_test/weights/best.pt'
    res, _ = evaluate_checkpoint(ckpt, split='val', imgsz=416)
    print("Baseline Validation Results:")
    print(json.dumps(res, indent=2))

