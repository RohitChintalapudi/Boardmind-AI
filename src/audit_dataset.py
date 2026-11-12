"""
src/audit_dataset.py
Comprehensive audit script for BoardMind AI Whiteboard Detection dataset and baseline model.
Performs:
1. Project structure & YAML verification
2. Image & label matching and integrity checks
3. Annotation format and consistency audit (BBox vs Polygon, coord bounds, class distribution)
4. Duplicate / near-duplicate image analysis (MD5, base-name leakage across splits)
5. Bounding box distributions (aspect ratios, normalized areas, multi-board count)
6. Baseline checkpoint inspection (architecture, hyperparameters, validation metrics)
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import yaml
import cv2
import numpy as np
import torch
from PIL import Image

def get_image_hash(img_path):
    with open(img_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_base_filename(filename):
    # Roboflow filename pattern: <base_name>.rf.<hash>.<ext>
    stem = Path(filename).stem
    if '.rf.' in stem:
        return stem.split('.rf.')[0]
    return stem

def audit():
    project_root = Path.cwd()
    dataset_root = project_root / 'datasets' / 'whiteboard'
    yaml_path = dataset_root / 'data.yaml'

    print("=" * 70)
    print("BOARDMIND AI: COMPREHENSIVE DATASET & BASELINE AUDIT")
    print("=" * 70)
    print(f"Project Root: {project_root}")
    print(f"Dataset Root: {dataset_root}")
    print(f"YAML Path:    {yaml_path} (Exists: {yaml_path.exists()})")

    # 1. Inspect YAML
    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    print("\n[1] YAML CONFIGURATION:")
    print(yaml.dump(data_cfg, default_flow_style=False))

    splits = ['train', 'valid', 'test']
    split_data = {}

    # 2. Check Image & Label Matching
    print("\n[2] SPLIT COUNTS & FILE MATCHING:")
    for split in splits:
        img_dir = dataset_root / split / 'images'
        lbl_dir = dataset_root / split / 'labels'

        img_files = sorted(list(img_dir.glob('*.*')))
        lbl_files = sorted(list(lbl_dir.glob('*.txt')))

        img_stems = {f.stem: f for f in img_files}
        lbl_stems = {f.stem: f for f in lbl_files}

        missing_labels = set(img_stems.keys()) - set(lbl_stems.keys())
        extra_labels = set(lbl_stems.keys()) - set(img_stems.keys())

        print(f"  Split: {split:<6} | Images: {len(img_files):<5} | Labels: {len(lbl_files):<5} | Missing lbls: {len(missing_labels)} | Extra lbls: {len(extra_labels)}")
        split_data[split] = {
            'img_files': img_files,
            'lbl_files': lbl_files,
            'img_stems': img_stems,
            'lbl_stems': lbl_stems
        }

    # 3. Annotation Format Audit
    print("\n[3] ANNOTATION FORMAT & INTEGRITY AUDIT:")
    global_class_counts = Counter()
    format_counts = defaultdict(Counter)
    empty_label_files = defaultdict(list)
    corrupted_labels = defaultdict(list)
    out_of_bounds_labels = defaultdict(list)

    bbox_stats = defaultdict(list) # split -> list of (w, h, area, aspect_ratio, num_boxes_in_img)

    for split in splits:
        lbl_files = split_data[split]['lbl_files']
        for lbl_path in lbl_files:
            try:
                content = lbl_path.read_text().strip()
                if not content:
                    empty_label_files[split].append(lbl_path.name)
                    continue

                lines = [l.strip() for l in content.splitlines() if l.strip()]
                num_boxes = len(lines)
                
                for line_idx, line in enumerate(lines):
                    parts = line.split()
                    if not parts:
                        continue
                    cls_id = int(float(parts[0]))
                    global_class_counts[cls_id] += 1
                    coords = [float(p) for p in parts[1:]]

                    if len(parts) == 5:
                        format_counts[split]['bbox_standard_5'] += 1
                        cx, cy, w, h = coords
                        xmin = cx - w / 2.0
                        ymin = cy - h / 2.0
                        xmax = cx + w / 2.0
                        ymax = cy + h / 2.0
                    elif len(parts) >= 7 and len(coords) % 2 == 0:
                        format_counts[split]['polygon_segmentation'] += 1
                        xs = coords[0::2]
                        ys = coords[1::2]
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)
                        w = xmax - xmin
                        h = ymax - ymin
                        cx = (xmin + xmax) / 2.0
                        cy = (ymin + ymax) / 2.0
                    else:
                        format_counts[split]['irregular_format'] += 1
                        corrupted_labels[split].append((lbl_path.name, line_idx, f"Invalid token count {len(parts)}"))
                        continue

                    # Check coordinate bounds
                    if any(c < -0.01 or c > 1.01 for c in [xmin, ymin, xmax, ymax]):
                        out_of_bounds_labels[split].append((lbl_path.name, line_idx, [xmin, ymin, xmax, ymax]))

                    aspect_ratio = (w / h) if h > 1e-6 else 0
                    area = w * h
                    bbox_stats[split].append({
                        'file': lbl_path.name,
                        'cx': cx, 'cy': cy, 'w': w, 'h': h,
                        'area': area, 'aspect_ratio': aspect_ratio,
                        'num_boxes': num_boxes
                    })
            except Exception as e:
                corrupted_labels[split].append((lbl_path.name, -1, str(e)))

        print(f"  Split {split}:")
        print(f"    Formats: {dict(format_counts[split])}")
        print(f"    Empty label files: {len(empty_label_files[split])}")
        print(f"    Corrupted/invalid: {len(corrupted_labels[split])}")
        print(f"    Out of bounds (>1.01 or <-0.01): {len(out_of_bounds_labels[split])}")

    print(f"  Overall Class distribution: {dict(global_class_counts)}")

    # 4. Image Dimensions and Integrity
    print("\n[4] IMAGE RESOLUTION & INTEGRITY CHECK:")
    image_resolutions = defaultdict(Counter)
    corrupted_images = []
    
    for split in splits:
        for img_path in split_data[split]['img_files']:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    channels = len(im.getbands())
                    image_resolutions[split][(w, h, channels)] += 1
            except Exception as e:
                corrupted_images.append((img_path, str(e)))
                
        print(f"  Split: {split}")
        for res, count in image_resolutions[split].most_common(5):
            print(f"    Resolution (W, H, Channels) {res}: {count} images ({count/len(split_data[split]['img_files'])*100:.1f}%)")
    print(f"  Total corrupted images: {len(corrupted_images)}")

    # 5. Duplicate and Data Leakage Check
    print("\n[5] DUPLICATE & SPLIT LEAKAGE CHECK:")
    md5_to_paths = defaultdict(list)
    base_name_to_splits = defaultdict(lambda: defaultdict(list))

    for split in splits:
        for img_path in split_data[split]['img_files']:
            h = get_image_hash(img_path)
            md5_to_paths[h].append((split, img_path.name))

            base = get_base_filename(img_path.name)
            base_name_to_splits[base][split].append(img_path.name)

    # Exact duplicates
    exact_dups = {h: paths for h, paths in md5_to_paths.items() if len(paths) > 1}
    cross_split_exact = []
    intra_split_exact = defaultdict(int)

    for h, paths in exact_dups.items():
        splits_involved = set(s for s, p in paths)
        if len(splits_involved) > 1:
            cross_split_exact.append(paths)
        else:
            intra_split_exact[list(splits_involved)[0]] += 1

    print(f"  Exact identical image hashes found: {len(exact_dups)} sets")
    print(f"  Cross-split EXACT duplicates: {len(cross_split_exact)} pairs/groups")
    for group in cross_split_exact[:5]:
        print(f"    Group: {group}")
    print(f"  Intra-split EXACT duplicates: {dict(intra_split_exact)}")

    # Base name (Roboflow augmentation leakage)
    print(f"\n  Unique base image stems (original source photos): {len(base_name_to_splits)}")
    cross_split_bases = []
    for base, s_dict in base_name_to_splits.items():
        if len(s_dict.keys()) > 1:
            cross_split_bases.append((base, {s: len(p) for s, p in s_dict.items()}))

    print(f"  CRITICAL: Base photos appearing in MULTIPLE splits (Data Leakage): {len(cross_split_bases)}")
    for base, counts in cross_split_bases[:10]:
        print(f"    Base '{base}': {counts}")

    # 6. Bounding Box Distribution Analysis
    print("\n[6] BOUNDING BOX GEOMETRY & DISTRIBUTION ANALYSIS:")
    for split in splits:
        stats = bbox_stats[split]
        if not stats:
            continue
        areas = [s['area'] for s in stats]
        aspect_ratios = [s['aspect_ratio'] for s in stats]
        widths = [s['w'] for s in stats]
        heights = [s['h'] for s in stats]
        boxes_per_img = [s['num_boxes'] for s in stats]

        print(f"  Split: {split} ({len(stats)} total annotations):")
        print(f"    Normalized Area  : min={np.min(areas):.4f}, median={np.median(areas):.4f}, mean={np.mean(areas):.4f}, max={np.max(areas):.4f}")
        print(f"    Aspect Ratio(W/H): min={np.min(aspect_ratios):.2f}, median={np.median(aspect_ratios):.2f}, mean={np.mean(aspect_ratios):.2f}, max={np.max(aspect_ratios):.2f}")
        print(f"    Width            : min={np.min(widths):.4f}, median={np.median(widths):.4f}, max={np.max(widths):.4f}")
        print(f"    Height           : min={np.min(heights):.4f}, median={np.median(heights):.4f}, max={np.max(heights):.4f}")
        
        # Area categories: small (< 0.05), medium (0.05 - 0.30), large (> 0.30)
        small = sum(1 for a in areas if a < 0.05)
        medium = sum(1 for a in areas if 0.05 <= a <= 0.30)
        large = sum(1 for a in areas if a > 0.30)
        print(f"    Size buckets: Small (<5% img area): {small} ({small/len(areas)*100:.1f}%), Medium (5-30%): {medium} ({medium/len(areas)*100:.1f}%), Large (>30%): {large} ({large/len(areas)*100:.1f}%)")

        # Whiteboards per image
        box_count_dist = Counter(boxes_per_img)
        print(f"    Boxes per image distribution: {dict(box_count_dist)}")

    # 7. Audit Existing Checkpoint
    checkpoint_path = project_root / 'runs' / 'whiteboard_detection_test' / 'weights' / 'best.pt'
    args_yaml_path = project_root / 'runs' / 'whiteboard_detection_test' / 'args.yaml'
    results_csv_path = project_root / 'runs' / 'whiteboard_detection_test' / 'results.csv'

    print("\n[7] EXISTING BASELINE CHECKPOINT AUDIT:")
    print(f"  Checkpoint path: {checkpoint_path}")
    print(f"  Exists: {checkpoint_path.exists()}")

    if checkpoint_path.exists():
        ckpt_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {ckpt_size_mb:.2f} MB")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            print(f"  Checkpoint keys: {list(ckpt.keys())}")
            print(f"  Epoch: {ckpt.get('epoch')}")
            print(f"  Best fitness: {ckpt.get('fitness')}")
            if 'train_args' in ckpt:
                t_args = ckpt['train_args']
                print(f"  Model architecture: {t_args.get('model')}")
                print(f"  Input image size: {t_args.get('imgsz')}")
                print(f"  Batch size: {t_args.get('batch')}")
                print(f"  Epochs: {t_args.get('epochs')}")
                print(f"  Device: {t_args.get('device')}")
        except Exception as e:
            print(f"  Error reading checkpoint: {e}")

    if results_csv_path.exists():
        print(f"\n  Final Training Metrics from {results_csv_path.name}:")
        with open(results_csv_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            if lines:
                headers = [h.strip() for h in lines[0].split(',')]
                last_line = [v.strip() for v in lines[-1].split(',')]
                for h, val in zip(headers, last_line):
                    print(f"    {h:<25}: {val}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    audit()

