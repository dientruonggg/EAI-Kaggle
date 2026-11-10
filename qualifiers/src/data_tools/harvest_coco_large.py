#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - COCO LARGE DATASET HARVESTER
Streams and downloads real COCO 2017 images filtered strictly for the 7 Qualifier classes:
  0: bottle   (COCO 39)
  1: cup      (COCO 41)
  2: phone    (COCO 67)
  3: book     (COCO 73)
  4: mouse    (COCO 64)
  5: box      (COCO 28 - suitcase)
  6: toy_car  (COCO 2 - car)

Features:
  - Connection pooling with requests.Session (Keep-Alive)
  - Automatic retries & exponential backoff
  - Balanced stratified sampling across rare categories
  - Automatic resizing to 320x240 (competition standard)
  - Accurate YOLO label conversion
"""

import os
import sys
import glob
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cv2
import numpy as np

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
LABELS_ROOT = "/home/bim/Projects/EAI-2026/datasets/coco_labels"

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
TARGET_W = 320
TARGET_H = 240

# Thread-local storage for HTTP sessions
thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        retries = Retry(total=4, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
        thread_local.session = session
    return thread_local.session

def scan_coco_labels(labels_dir, split="train2017"):
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"[*] Scanning {len(label_files)} label files in {split}...")

    matching_images = {}
    class_counts = {cid: 0 for cid in range(7)}

    for lf in label_files:
        base_id = os.path.splitext(os.path.basename(lf))[0]
        matched_boxes = []

        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    coco_cid = int(parts[0])
                    if coco_cid in COCO_TO_COMPETITION:
                        target_cid = COCO_TO_COMPETITION[coco_cid]
                        cx, cy, w, h = map(float, parts[1:5])
                        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and w > 0 and h > 0:
                            matched_boxes.append((target_cid, cx, cy, w, h))
                            class_counts[target_cid] += 1

        if matched_boxes:
            matching_images[base_id] = matched_boxes

    print(f"[+] Found {len(matching_images)} images with target classes in {split}.")
    print("    Class instances breakdown:")
    for cid in range(7):
        print(f"      - {CLASS_NAMES[cid]} (id {cid}): {class_counts[cid]} instances")

    return matching_images

def download_and_process_image(item):
    img_id, boxes, split, out_img_dir, out_lbl_dir = item
    url = f"http://images.cocodataset.org/{split}/{img_id}.jpg"
    
    img_out_path = os.path.join(out_img_dir, f"coco_{img_id}.jpg")
    lbl_out_path = os.path.join(out_lbl_dir, f"coco_{img_id}.txt")

    if os.path.exists(img_out_path) and os.path.exists(lbl_out_path):
        return True, img_id

    session = get_session()
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=12)
            if resp.status_code == 200 and len(resp.content) > 1000:
                img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                if img is None or img.size == 0:
                    return False, img_id

                resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(img_out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

                with open(lbl_out_path, "w") as f:
                    for cid, cx, cy, w, h in boxes:
                        f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                return True, img_id
        except Exception as e:
            if attempt == 2:
                return False, img_id
            time.sleep(0.3 * (attempt + 1))

    return False, img_id

def harvest_and_build(
    max_images=1200,
    val_split_ratio=0.15,
    num_threads=16,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("   COCO REAL DATASET HARVESTING & BALANCING FOR QUALIFIERS")
    print("=" * 70)

    val_dir = os.path.join(LABELS_ROOT, "coco/labels/val2017")
    train_dir = os.path.join(LABELS_ROOT, "coco/labels/train2017")

    val_candidates = scan_coco_labels(val_dir, split="val2017") if os.path.exists(val_dir) else {}
    train_candidates = scan_coco_labels(train_dir, split="train2017") if os.path.exists(train_dir) else {}

    rare_order = [4, 2, 5, 1, 0, 3, 6]

    all_items = []
    for img_id, boxes in val_candidates.items():
        all_items.append((img_id, boxes, "val2017"))
    for img_id, boxes in train_candidates.items():
        all_items.append((img_id, boxes, "train2017"))

    print(f"[*] Total available candidate images with target classes: {len(all_items)}")

    # Stratified selection
    buckets = {cid: [] for cid in range(7)}
    for item in all_items:
        cids = set(b[0] for b in item[1])
        for cid in cids:
            buckets[cid].append(item)

    chosen_set = set()
    final_selected = []

    per_class_target = max_images // 7 + 60
    for cid in rare_order:
        random.shuffle(buckets[cid])
        count = 0
        for item in buckets[cid]:
            img_id = item[0]
            if img_id not in chosen_set:
                chosen_set.add(img_id)
                final_selected.append(item)
                count += 1
                if count >= per_class_target or len(final_selected) >= max_images:
                    break
        if len(final_selected) >= max_images:
            break

    if len(final_selected) < max_images:
        random.shuffle(all_items)
        for item in all_items:
            img_id = item[0]
            if img_id not in chosen_set:
                chosen_set.add(img_id)
                final_selected.append(item)
                if len(final_selected) >= max_images:
                    break

    print(f"[+] Selected {len(final_selected)} highly balanced images for download.")

    random.shuffle(final_selected)
    n_val = int(len(final_selected) * val_split_ratio)
    val_set = final_selected[:n_val]
    train_set = final_selected[n_val:]

    out_train_img = os.path.join(PROJECT_ROOT, "data/train/images")
    out_train_lbl = os.path.join(PROJECT_ROOT, "data/train/labels")
    out_val_img = os.path.join(PROJECT_ROOT, "data/val/images")
    out_val_lbl = os.path.join(PROJECT_ROOT, "data/val/labels")

    os.makedirs(out_train_img, exist_ok=True)
    os.makedirs(out_train_lbl, exist_ok=True)
    os.makedirs(out_val_img, exist_ok=True)
    os.makedirs(out_val_lbl, exist_ok=True)

    tasks = []
    for item in train_set:
        tasks.append((item[0], item[1], item[2], out_train_img, out_train_lbl))
    for item in val_set:
        tasks.append((item[0], item[1], item[2], out_val_img, out_val_lbl))

    print(f"[*] Downloading & processing {len(tasks)} images with {num_threads} threads...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_and_process_image, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            ok, img_id = f.result()
            if ok:
                success_count += 1
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"    Progress: {done}/{len(tasks)} ({done*100/len(tasks):.1f}%) | Success: {success_count}")

    print(f"[✔] Successfully processed & saved {success_count} real COCO images!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    harvest_and_build(max_images=args.max_images, num_threads=args.threads)
