"""
src/run_phase3_augmentations.py
Executes Phase 3: Controlled Augmentation Experiments for BoardMind AI.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from train_experiment import run_experiment, load_registry

def run_phase3():
    print("=" * 80)
    print("PHASE 3: CONTROLLED DATA AUGMENTATION EXPERIMENTS")
    print("=" * 80)

    aug_experiments = [
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
        }
    ]

    registry = load_registry()

    for idx, exp in enumerate(aug_experiments, 1):
        name = exp['name']
        print(f"\n>>> [{idx}/{len(aug_experiments)}] Executing {name}...")
        if name in registry and (Path('runs/whiteboard_optimization') / name / 'weights' / 'best.pt').exists():
            print(f"Experiment {name} already exists. Skipping.")
            continue

        run_experiment(
            experiment_name=name,
            model_source=exp['model_source'],
            category=exp['category'],
            description=exp['description'],
            epochs=exp.get('epochs', 3),
            imgsz=exp.get('imgsz', 416),
            batch=exp.get('batch', 8),
            lr0=exp.get('lr0', 0.002),
            lrf=exp.get('lrf', 0.01),
            cos_lr=exp.get('cos_lr', True),
            perspective=exp.get('perspective', 0.0),
            degrees=exp.get('degrees', 0.0),
            scale=exp.get('scale', 0.5),
            translate=exp.get('translate', 0.1),
            hsv_v=exp.get('hsv_v', 0.4),
            hsv_s=exp.get('hsv_s', 0.7),
            erasing=exp.get('erasing', 0.4),
            fliplr=exp.get('fliplr', 0.5),
            mosaic=exp.get('mosaic', 1.0)
        )

    print("\nPhase 3 Augmentation Experiments Finished.")

if __name__ == '__main__':
    run_phase3()

