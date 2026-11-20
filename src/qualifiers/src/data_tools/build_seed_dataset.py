#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - SEED DATASET BUILDER
Creates an initial multi-object seed dataset for the 7 Qualifier classes:
  0: bottle
  1: cup
  2: phone
  3: book
  4: mouse
  5: box
  6: toy_car

Sources:
1. Filters & maps target classes from local COCO128 images, resized to 320x240.
2. Generates synthetic tabletop compositions with varied counts, classes, and spatial zones (left, center, right).
3. Splits into train, val, test subsets with reproducible seed.
"""
import os
import sys
import glob
import random
import cv2
import numpy as np

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
ORIG_W = 320
ORIG_H = 240

COCO_TO_COMPETITION = {
    39: 0,  # bottle -> bottle
    41: 1,  # cup -> cup
    67: 2,  # cell phone -> phone
    73: 3,  # book -> book
    64: 4,  # mouse -> mouse
    28: 5,  # suitcase -> box
    2: 6    # car -> toy_car
}

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]

def create_synthetic_object(obj_cls, w, h):
    """Generates a synthetic graphic representation of the 7 objects."""
    img = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
    
    # Random realistic colors
    if obj_cls == 0:  # bottle (cylindrical bottle with cap)
        color = (random.randint(150, 240), random.randint(180, 240), random.randint(200, 255), 255)
        cap_color = (random.randint(40, 200), random.randint(40, 100), random.randint(180, 255), 255)
        # Body
        body_top = int(h * 0.25)
        cv2.rectangle(img, (int(w * 0.15), body_top), (int(w * 0.85), h - 2), color, -1)
        # Neck & Cap
        cv2.rectangle(img, (int(w * 0.35), int(h * 0.08)), (int(w * 0.65), body_top), color, -1)
        cv2.rectangle(img, (int(w * 0.30), 2), (int(w * 0.70), int(h * 0.08)), cap_color, -1)
        # Highlight line
        cv2.line(img, (int(w * 0.3), body_top + 4), (int(w * 0.3), h - 6), (255, 255, 255, 180), 2)

    elif obj_cls == 1:  # cup (trapezoid or rectangle with handle)
        color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255), 255)
        cv2.rectangle(img, (int(w * 0.15), int(h * 0.1)), (int(w * 0.75), h - 2), color, -1)
        # Handle on right
        cv2.ellipse(img, (int(w * 0.75), int(h * 0.5)), (int(w * 0.2), int(h * 0.25)), 0, -90, 90, (120, 120, 120, 255), 3)

    elif obj_cls == 2:  # phone (rounded dark rectangle with screen)
        cv2.rectangle(img, (2, 2), (w - 2, h - 2), (30, 30, 30, 255), -1)
        # Screen
        screen_color = (random.randint(40, 80), random.randint(60, 120), random.randint(100, 160), 255)
        cv2.rectangle(img, (5, 8), (w - 5, h - 8), screen_color, -1)
        # Camera notch / dot
        cv2.circle(img, (w // 2, 5), 2, (10, 10, 10, 255), -1)

    elif obj_cls == 3:  # book (thick rectangular block with spine)
        cover_color = (random.randint(30, 180), random.randint(40, 140), random.randint(120, 220), 255)
        cv2.rectangle(img, (2, 2), (w - 2, h - 2), cover_color, -1)
        # Spine stripe
        cv2.line(img, (int(w * 0.12), 2), (int(w * 0.12), h - 2), (230, 210, 160, 255), 3)

    elif obj_cls == 4:  # mouse (oval with button divider)
        body_color = (random.randint(40, 90), random.randint(40, 90), random.randint(40, 90), 255)
        cv2.ellipse(img, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, body_color, -1)
        # Scroll wheel / divider
        cv2.line(img, (w // 2, 4), (w // 2, int(h * 0.45)), (20, 20, 20, 255), 2)
        cv2.circle(img, (w // 2, int(h * 0.25)), 2, (200, 50, 50, 255), -1)

    elif obj_cls == 5:  # box (cardboard brown rectangle with tape)
        brown = (random.randint(70, 110), random.randint(130, 170), random.randint(180, 210), 255)
        cv2.rectangle(img, (2, 2), (w - 2, h - 2), brown, -1)
        # Tape line
        cv2.line(img, (2, h // 2), (w - 2, h // 2), (100, 150, 190, 255), 4)

    elif obj_cls == 6:  # toy_car (car silhouette with wheels)
        car_color = (random.randint(40, 80), random.randint(40, 80), random.randint(180, 255), 255)
        # Lower body
        cv2.rectangle(img, (2, int(h * 0.4)), (w - 2, int(h * 0.85)), car_color, -1)
        # Roof / Cabin
        cv2.rectangle(img, (int(w * 0.25), int(h * 0.15)), (int(w * 0.75), int(h * 0.45)), (220, 230, 240, 255), -1)
        # Wheels
        wheel_r = max(3, int(h * 0.14))
        cv2.circle(img, (int(w * 0.25), int(h * 0.85)), wheel_r, (20, 20, 20, 255), -1)
        cv2.circle(img, (int(w * 0.75), int(h * 0.85)), wheel_r, (20, 20, 20, 255), -1)

    return img

def create_synthetic_scene(num_objects=None):
    """
    Creates a 320x240 tabletop scene with synthetic objects placed realistically.
    Returns: BGR image, list of (cls_id, cx, cy, w, h)
    """
    # Background: Table surface with texture/gradient
    bg = np.zeros((ORIG_H, ORIG_W, 3), dtype=np.uint8)
    base_color = np.array([random.randint(160, 210), random.randint(180, 220), random.randint(190, 230)], dtype=np.float32)
    # Add subtle vertical gradient
    for y in range(ORIG_H):
        ratio = 0.85 + 0.3 * (y / ORIG_H)
        bg[y, :] = np.clip(base_color * ratio, 0, 255).astype(np.uint8)
    
    # Add subtle table texture noise (simulates OV7670 camera grain)
    noise = np.random.normal(0, 4, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if num_objects is None:
        num_objects = random.choices([0, 1, 2, 3, 4], weights=[0.05, 0.35, 0.35, 0.20, 0.05])[0]

    annotations = []
    occupied_boxes = []

    # Choose classes for this scene
    available_classes = list(range(7))
    random.shuffle(available_classes)

    for i in range(num_objects):
        obj_cls = available_classes[i % len(available_classes)]
        
        # Determine realistic object size at 320x240
        if obj_cls in [0, 1]:  # bottle, cup (taller than wide)
            obj_w = random.randint(28, 48)
            obj_h = random.randint(55, 90)
        elif obj_cls in [2]:  # phone
            obj_w = random.randint(25, 42)
            obj_h = random.randint(50, 75)
        elif obj_cls in [3, 5]:  # book, box (larger rectangle)
            obj_w = random.randint(45, 80)
            obj_h = random.randint(40, 70)
        elif obj_cls in [4]:  # mouse (small oval)
            obj_w = random.randint(25, 38)
            obj_h = random.randint(35, 52)
        else:  # toy_car
            obj_w = random.randint(45, 75)
            obj_h = random.randint(30, 48)

        # Try to find a non-overlapping or lightly-overlapping position
        placed = False
        for attempt in range(25):
            px = random.randint(10, ORIG_W - obj_w - 10)
            py = random.randint(25, ORIG_H - obj_h - 15)

            # Check overlap with existing objects
            overlap = False
            for (ox1, oy1, ox2, oy2) in occupied_boxes:
                if not (px + obj_w < ox1 or px > ox2 or py + obj_h < oy1 or py > oy2):
                    # Intersecting
                    overlap = True
                    break
            if not overlap or attempt > 20:  # Allow partial occlusion on last attempts
                placed = True
                break

        if not placed:
            continue

        obj_img = create_synthetic_object(obj_cls, obj_w, obj_h)
        
        # Alpha blend onto background
        alpha = obj_img[:, :, 3] / 255.0
        for c in range(3):
            bg[py:py+obj_h, px:px+obj_w, c] = (
                alpha * obj_img[:, :, c] + (1.0 - alpha) * bg[py:py+obj_h, px:px+obj_w, c]
            ).astype(np.uint8)

        occupied_boxes.append((px, py, px + obj_w, py + obj_h))

        # Normalized coordinates for YOLO
        cx = (px + obj_w / 2.0) / ORIG_W
        cy = (py + obj_h / 2.0) / ORIG_H
        nw = obj_w / ORIG_W
        nh = obj_h / ORIG_H
        annotations.append((obj_cls, cx, cy, nw, nh))

    return bg, annotations

def extract_coco128_subset():
    """Extracts target objects from local COCO128 and resizes to 320x240."""
    coco_labels = glob.glob("/home/bim/Projects/EAI-2026/datasets/coco128/labels/train2017/*.txt")
    coco_img_dir = "/home/bim/Projects/EAI-2026/datasets/coco128/images/train2017"

    extracted = []
    for lf in coco_labels:
        sid = os.path.splitext(os.path.basename(lf))[0]
        img_p = os.path.join(coco_img_dir, f"{sid}.jpg")
        if not os.path.exists(img_p):
            continue

        matched_boxes = []
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if cid in COCO_TO_COMPETITION:
                        target_cid = COCO_TO_COMPETITION[cid]
                        cx, cy, w, h = map(float, parts[1:5])
                        matched_boxes.append((target_cid, cx, cy, w, h))

        if matched_boxes:
            img = cv2.imread(img_p)
            if img is not None:
                resized_img = cv2.resize(img, (ORIG_W, ORIG_H), interpolation=cv2.INTER_AREA)
                extracted.append((resized_img, matched_boxes))

    return extracted

def build_dataset(total_scenes=160, train_ratio=0.70, val_ratio=0.15, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 70)
    print("   BUILDING QUALIFIERS SEED DATASET (7 CLASSES, 320x240)")
    print("=" * 70)

    # 1. Harvest COCO128 target objects
    coco_samples = extract_coco128_subset()
    print(f"[*] Harvested {len(coco_samples)} real object images from COCO128.")

    # 2. Synthesize tabletop scenes to enrich balanced coverage
    synthetic_needed = max(0, total_scenes - len(coco_samples))
    print(f"[*] Synthesizing {synthetic_needed} tabletop multi-object scenes...")
    synth_samples = []
    for _ in range(synthetic_needed):
        img, boxes = create_synthetic_scene()
        synth_samples.append((img, boxes))

    all_samples = coco_samples + synth_samples
    random.shuffle(all_samples)

    # 3. Partition into Train, Val, Test
    n_train = int(len(all_samples) * train_ratio)
    n_val = int(len(all_samples) * val_ratio)
    n_test = len(all_samples) - n_train - n_val

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:]
    }

    print(f"[*] Splitting dataset: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    out_base = os.path.join(PROJECT_ROOT, "data")
    for split_name, samples in splits.items():
        img_dir = os.path.join(out_base, split_name, "images")
        lbl_dir = os.path.join(out_base, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        class_hist = {i: 0 for i in range(7)}
        total_boxes = 0

        for idx, (img, boxes) in enumerate(samples):
            sample_id = f"qual_{split_name}_{idx+1:04d}"
            img_path = os.path.join(img_dir, f"{sample_id}.png")
            lbl_path = os.path.join(lbl_dir, f"{sample_id}.txt")

            cv2.imwrite(img_path, img)

            with open(lbl_path, "w") as f:
                for (cid, cx, cy, w, h) in boxes:
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    class_hist[cid] += 1
                    total_boxes += 1

        print(f"  [✔] {split_name.upper()}: {len(samples)} images, {total_boxes} boxes. Class counts: {class_hist}")

    print("\n[✔] SEED DATASET GENERATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    build_dataset(total_scenes=160)
