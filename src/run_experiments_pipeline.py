"""
src/run_experiments_pipeline.py
Systematic execution pipeline for BoardMind AI Whiteboard Detection experiments.
Runs controlled experiments sequentially on CPU, tracking metrics and logging results.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent))

from train_experiment import run_experiment, load_registry, save_registry

def run_all():
    print("=" * 80)
    print("BOARDMIND AI: SYSTEMATIC CV EXPERIMENTATION PIPELINE")
    print("Environment: Intel Core i9-13900H CPU | Framework: Ultralytics YOLOv8/11")
    print("=" * 80)

    experiments = [
        # --- PHASE 3: AUGMENTATION EXPERIMENTS ---
        {
            'name': 'exp1_aug_perspective_scale',
            'model_source': 'runs/whiteboard_detection_test/weights/best.pt',
            'category': 'data_augmentation',
            'description': 'Geometric augmentation: perspective (0.001), rotation (10 deg), scale (0.7), translate (0.2) to break full-frame bias',
            'epochs': 3,
            'imgsz': 416,
            'batch': 8,
            'lr0': 0.002,
            'lrf': 0.01,
            'cos_lr': True,
            'perspective': 0.001,
            'degrees': 10.0,
            'scale': 0.7,
            'translate': 0.2,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'erasing': 0.2
        },
        {
            'name': 'exp2_aug_lighting_glare',
            'model_source': 'runs/whiteboard_detection_test/weights/best.pt',
            'category': 'data_augmentation',
            'description': 'Photometric augmentation: elevated HSV value/saturation jitter (0.6, 0.6) and random erasing (0.5) for glare/occlusion',
            'epochs': 3,
            'imgsz': 416,
            'batch': 8,
            'lr0': 0.002,
            'lrf': 0.01,
            'cos_lr': True,
            'hsv_v': 0.6,
            'hsv_s': 0.6,
            'erasing': 0.5,
            'degrees': 0.0,
            'perspective': 0.0,
            'scale': 0.5,
            'translate': 0.1,
            'fliplr': 0.5,
            'mosaic': 1.0
        },
        {
            'name': 'exp3_aug_combined_robust',
            'model_source': 'runs/whiteboard_detection_test/weights/best.pt',
            'category': 'data_augmentation',
            'description': 'Combined robust augmentation: geometric perspective + scaling + photometric glare simulation + mosaic',
            'epochs': 3,
            'imgsz': 416,
            'batch': 8,
            'lr0': 0.002,
            'lrf': 0.01,
            'cos_lr': True,
            'perspective': 0.001,
            'degrees': 10.0,
            'scale': 0.7,
            'translate': 0.2,
            'hsv_v': 0.5,
            'hsv_s': 0.6,
            'erasing': 0.4,
            'fliplr': 0.5,
            'mosaic': 1.0
        },

        # --- PHASE 4: RESOLUTION OPTIMIZATION ---
        {
            'name': 'exp4_res_320',
            'model_source': 'runs/whiteboard_detection_test/weights/best.pt',
            'category': 'resolution_optimization',
            'description': 'Resolution optimization: 320x320 input size to maximize CPU frame rate and evaluate localization impact',
            'epochs': 3,
            'imgsz': 320,
            'batch': 8,
            'lr0': 0.002,
            'lrf': 0.01,
            'cos_lr': True,
            'scale': 0.6,
            'perspective': 0.0005,
            'degrees': 5.0,
            'fliplr': 0.5,
            'mosaic': 1.0
        },

        # --- PHASE 4: ARCHITECTURE COMPARISON ---
        {
            'name': 'exp5_arch_yolo11n',
            'model_source': 'yolo11n.yaml',
            'category': 'architecture_comparison',
            'description': 'Architecture comparison: YOLO11n (2.62M params, 6.7 GFLOPs) vs YOLOv8n (3.14M params, 8.7 GFLOPs)',
            'epochs': 3,
            'imgsz': 416,
            'batch': 8,
            'lr0': 0.01,
            'lrf': 0.01,
            'optimizer': 'auto',
            'fliplr': 0.5,
            'mosaic': 1.0
        },

        # --- PHASE 4: HYPERPARAMETER & SCHEDULER OPTIMIZATION ---
        {
            'name': 'exp6_hp_adamw_cos',
            'model_source': 'yolov8n.pt',
            'category': 'hyperparameter_optimization',
            'description': 'Hyperparameter optimization: COCO pretrained YOLOv8n with AdamW optimizer, cosine learning rate scheduler, weight decay=0.001',
            'epochs': 3,
            'imgsz': 416,
            'batch': 8,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'cos_lr': True,
            'weight_decay': 0.001,
            'warmup_epochs': 1.0,
            'fliplr': 0.5,
            'mosaic': 1.0
        }
    ]

    registry = load_registry()

    for idx, exp in enumerate(experiments, 1):
        name = exp['name']
        print(f"\n>>> [{idx}/{len(experiments)}] Running {name}...")

        # Skip if already completed in registry
        if name in registry and (Path('runs/whiteboard_optimization') / name / 'weights' / 'best.pt').exists():
            print(f"Experiment {name} already completed in registry. Skipping.")
            continue

        try:
            record = run_experiment(
                experiment_name=name,
                model_source=exp['model_source'],
                category=exp['category'],
                description=exp['description'],
                epochs=exp.get('epochs', 3),
                imgsz=exp.get('imgsz', 416),
                batch=exp.get('batch', 8),
                lr0=exp.get('lr0', 0.01),
                lrf=exp.get('lrf', 0.01),
                optimizer=exp.get('optimizer', 'auto'),
                cos_lr=exp.get('cos_lr', False),
                weight_decay=exp.get('weight_decay', 0.0005),
                warmup_epochs=exp.get('warmup_epochs', 1.0),
                freeze=exp.get('freeze', None),
                hsv_h=exp.get('hsv_h', 0.015),
                hsv_s=exp.get('hsv_s', 0.7),
                hsv_v=exp.get('hsv_v', 0.4),
                degrees=exp.get('degrees', 0.0),
                translate=exp.get('translate', 0.1),
                scale=exp.get('scale', 0.5),
                shear=exp.get('shear', 0.0),
                perspective=exp.get('perspective', 0.0),
                flipud=exp.get('flipud', 0.0),
                fliplr=exp.get('fliplr', 0.5),
                mosaic=exp.get('mosaic', 1.0),
                mixup=exp.get('mixup', 0.0),
                erasing=exp.get('erasing', 0.4)
            )
            print(f"Successfully finished {name}.")
        except Exception as e:
            print(f"ERROR running {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ALL PLANNED EXPERIMENTS COMPLETED!")
    print("=" * 80)

if __name__ == '__main__':
    run_all()

