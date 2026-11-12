"""
src/train_experiment.py
Modular Experiment Runner for BoardMind AI Whiteboard Detection Optimization.

Designed specifically for CPU training under Windows constraints:
- workers=0, device='cpu', cache=False
- Outputs saved under runs/whiteboard_optimization/<experiment_name>
- Maintains full experiment logs and JSON registry
"""

import os
import sys
import json
import time
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# Add src to sys.path if not present
sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_model import evaluate_checkpoint, benchmark_inference

EXPERIMENTS_REGISTRY = Path('runs/whiteboard_optimization/experiment_registry.json')

def load_registry():
    if EXPERIMENTS_REGISTRY.exists():
        with open(EXPERIMENTS_REGISTRY, 'r') as f:
            return json.load(f)
    return {}

def save_registry(registry):
    EXPERIMENTS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_REGISTRY, 'w') as f:
        json.dump(registry, f, indent=2)

def run_experiment(
    experiment_name,
    model_source='yolov8n.pt',
    data_yaml='datasets/whiteboard/data.yaml',
    epochs=5,
    imgsz=416,
    batch=4,
    lr0=0.01,
    lrf=0.01,
    optimizer='auto',
    cos_lr=False,
    weight_decay=0.0005,
    warmup_epochs=1.0,
    freeze=None,
    # Augmentation parameters
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    erasing=0.4,
    description="",
    category="general"
):
    """
    Executes a controlled training experiment and records validation metrics.
    """
    project_dir = Path.cwd().resolve() / 'runs' / 'whiteboard_optimization'
    output_dir = project_dir / experiment_name

    print("\n" + "=" * 70)
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print(f"Category:    {category}")
    print(f"Description: {description}")
    print(f"Base Model:  {model_source}")
    print(f"Resolution:  {imgsz}x{imgsz} | Batch: {batch} | Epochs: {epochs}")
    print("=" * 70)

    t_start = time.time()

    # Load model
    model = YOLO(model_source)

    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': 'cpu',
        'workers': 0,
        'cache': False,
        'amp': False,
        'optimizer': optimizer,
        'lr0': lr0,
        'lrf': lrf,
        'cos_lr': cos_lr,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'freeze': freeze,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v,
        'degrees': degrees,
        'translate': translate,
        'scale': scale,
        'shear': shear,
        'perspective': perspective,
        'flipud': flipud,
        'fliplr': fliplr,
        'mosaic': mosaic,
        'mixup': mixup,
        'erasing': erasing,
        'project': str(project_dir),
        'name': experiment_name,
        'exist_ok': True,
        'verbose': True,
        'plots': True
    }

    # Execute training
    train_results = model.train(**train_args)
    t_end = time.time()
    training_time_sec = round(t_end - t_start, 2)

    actual_save_dir = Path(train_results.save_dir) if hasattr(train_results, 'save_dir') else output_dir
    best_checkpoint = actual_save_dir / 'weights' / 'best.pt'
    if not best_checkpoint.exists():
        best_checkpoint = actual_save_dir / 'weights' / 'last.pt'

    print(f"\nTraining completed in {training_time_sec}s. Best weights: {best_checkpoint}")

    # Standardized validation on validation set
    val_results, _ = evaluate_checkpoint(
        best_checkpoint,
        data_yaml=data_yaml,
        split='val',
        imgsz=imgsz,
        batch=batch,
        save_dir=output_dir / 'val'
    )

    # CPU Latency benchmark
    sample_images = list(Path('datasets/whiteboard/valid/images').glob('*.*'))[:20]
    bench_results = benchmark_inference(YOLO(str(best_checkpoint)), sample_images, imgsz=imgsz, iterations=20)

    # Combine into record
    experiment_record = {
        'experiment_name': experiment_name,
        'category': category,
        'description': description,
        'model_source': str(model_source),
        'imgsz': imgsz,
        'batch': batch,
        'epochs': epochs,
        'training_time_sec': training_time_sec,
        'hyperparameters': {
            'optimizer': optimizer,
            'lr0': lr0,
            'lrf': lrf,
            'cos_lr': cos_lr,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'freeze': freeze
        },
        'augmentations': {
            'hsv_h': hsv_h, 'hsv_s': hsv_s, 'hsv_v': hsv_v,
            'degrees': degrees, 'translate': translate, 'scale': scale,
            'perspective': perspective, 'fliplr': fliplr, 'mosaic': mosaic,
            'mixup': mixup, 'erasing': erasing
        },
        'metrics_val': val_results,
        'latency_benchmark': bench_results,
        'best_checkpoint': str(best_checkpoint)
    }

    # Update registry
    registry = load_registry()
    registry[experiment_name] = experiment_record
    save_registry(registry)

    print("\n--- EXPERIMENT SUMMARY ---")
    print(f"Precision: {val_results['precision']:.4f} | Recall: {val_results['recall']:.4f}")
    print(f"mAP@50:    {val_results['mAP50']:.4f} | mAP@50-95: {val_results['mAP50_95']:.4f}")
    print(f"F1 Score:  {val_results['f1_score']:.4f}")
    print(f"Inference: {bench_results['mean_latency_ms']} ms/img ({bench_results['fps']} FPS)")
    print("=" * 70 + "\n")

    return experiment_record

if __name__ == '__main__':
    print("Experiment Runner loaded.")
