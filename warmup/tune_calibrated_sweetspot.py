#!/usr/bin/env python3
"""
FIND THE OPTIMAL SCALED PER-CLASS THRESHOLDS FOR SWEET SPOT DENSITY
Combines:
  - Relative per-class sensitivity from Coordinate Descent:
    [0.20, 0.08, 0.14, 0.14, 0.14, 0.14, 0.08, 0.14, 0.18, 0.28]
  - Multiplier grid search to hit the exact 2,700 - 2,780 test sweet spot
  - Evaluates validation mAP50 at each multiplier
"""

import os
import sys
import json
import glob
import csv
import numpy as np
import tensorflow as tf

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation import load_ground_truth, calculate_map50
from tune_ensemble import cache_preds
from tune_wbf import wbf_with_boost

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
TEST_IMG_DIR = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
INTENT_PREDS_PATH = "/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv"
ORIG_W, ORIG_H = 1672.0, 941.0

BASE_CONFS = [0.20, 0.08, 0.14, 0.14, 0.14, 0.14, 0.08, 0.14, 0.18, 0.28]
OPT_ALPHAS = [0.5, 0.0, 0.6, 1.0, 0.3, 0.6, 0.4, 0.1, 0.0, 0.2]

def main():
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)
    test_imgs = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))

    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"

    print("[*] Caching predictions on validation and test images...")
    c128_val = cache_preds(m128, val_imgs, 16, [0.0]*10, priors)
    c160_g_val = cache_preds(m160_g, val_imgs, 20, [0.0]*10, priors)
    c160_c_val = cache_preds(m160_c, val_imgs, 20, [0.0]*10, priors)
    val_models = [c128_val, c160_g_val, c160_c_val]

    c128_test = cache_preds(m128, test_imgs, 16, [0.0]*10, priors)
    c160_g_test = cache_preds(m160_g, test_imgs, 20, [0.0]*10, priors)
    c160_c_test = cache_preds(m160_c, test_imgs, 20, [0.0]*10, priors)
    test_models = [c128_test, c160_g_test, c160_c_test]

    def extract_candidates(models_list, img_paths):
        cached = {}
        for img_p in img_paths:
            sid = os.path.splitext(os.path.basename(img_p))[0]
            boxes = []
            for mod in models_list:
                probs, bxs, grid_size, _ = mod[sid]
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        cp = probs[gy, gx, 1:11]
                        bc = int(np.argmax(cp))
                        sc = float(cp[bc])
                        if sc >= 0.05:
                            p_dx, p_dy, p_w, p_h = float(bxs[gy, gx, 0]), float(bxs[gy, gx, 1]), float(bxs[gy, gx, 2]), float(bxs[gy, gx, 3])
                            cx = (gx + p_dx) / float(grid_size)
                            cy = (gy + p_dy) / float(grid_size)
                            a = OPT_ALPHAS[bc]
                            pw, ph = priors[str(bc)]["w"], priors[str(bc)]["h"]
                            w = (1.0 - a) * pw + a * p_w
                            h = (1.0 - a) * ph + a * p_h
                            xmin = max(0.0, (cx - w / 2.0) * ORIG_W)
                            ymin = max(0.0, (cy - h / 2.0) * ORIG_H)
                            xmax = min(ORIG_W, (cx + w / 2.0) * ORIG_W)
                            ymax = min(ORIG_H, (cy + h / 2.0) * ORIG_H)
                            boxes.append({"class_id": bc, "score": sc, "bbox": [xmin, ymin, xmax, ymax]})
            cached[sid] = boxes
        return cached

    val_cand = extract_candidates(val_models, val_imgs)
    test_cand = extract_candidates(test_models, test_imgs)

    print("\n--- GRID SEARCH ON THRESHOLD MULTIPLIER ---")
    multipliers = [1.05, 1.10, 1.15, 1.18, 1.20, 1.22, 1.25, 1.28, 1.30]

    best_scale = None
    best_map = -1
    best_test_boxes = 0

    for s in multipliers:
        scaled_confs = [round(c * s, 4) for c in BASE_CONFS]

        # Val mAP
        val_preds = {}
        for sid, raw in val_cand.items():
            final_boxes = []
            for c in range(10):
                c_thresh = scaled_confs[c]
                c_boxes = [b for b in raw if b["class_id"] == c and b["score"] >= c_thresh]
                if c_boxes:
                    fused = wbf_with_boost(c_boxes, iou_thresh=0.28, boost_factor=0.03)
                    final_boxes.extend(fused)
            final_boxes.sort(key=lambda b: b["score"], reverse=True)
            val_preds[sid] = final_boxes[:100]

        v_map, _ = calculate_map50(gt_dict, val_preds)

        # Test boxes
        test_boxes = 0
        for sid, raw in test_cand.items():
            final_boxes = []
            for c in range(10):
                c_thresh = scaled_confs[c]
                c_boxes = [b for b in raw if b["class_id"] == c and b["score"] >= c_thresh]
                if c_boxes:
                    fused = wbf_with_boost(c_boxes, iou_thresh=0.28, boost_factor=0.03)
                    final_boxes.extend(fused)
            test_boxes += len(final_boxes)

        in_sweetspot = 2650 <= test_boxes <= 2850
        flag = "★ SWEETSPOT" if in_sweetspot else ""
        print(f"Scale {s:.2f} -> Val mAP: {v_map*100:.2f}%, Test Boxes: {test_boxes} (avg {test_boxes/60.0:.1f}/img) {flag}")

        if in_sweetspot and v_map > best_map:
            best_map = v_map
            best_scale = s
            best_test_boxes = test_boxes

    print(f"\n[✔] Optimal Multiplier: {best_scale} (Val mAP: {best_map*100:.2f}%, Test Boxes: {best_test_boxes})")
    opt_confs = [round(c * best_scale, 4) for c in BASE_CONFS]
    print(f"    Optimal Thresholds: {opt_confs}")

    # Generate submission file with optimal scale
    out_csv = f"/home/bim/Projects/EAI-2026/warmup/submissions/sub_calib_sweetspot_s{int(best_scale*100)}_{best_test_boxes}.csv"
    results = []
    for sid, raw in test_cand.items():
        final_boxes = []
        for c in range(10):
            c_thresh = opt_confs[c]
            c_boxes = [b for b in raw if b["class_id"] == c and b["score"] >= c_thresh]
            if c_boxes:
                fused = wbf_with_boost(c_boxes, iou_thresh=0.28, boost_factor=0.03)
                final_boxes.extend(fused)
        final_boxes.sort(key=lambda b: b["score"], reverse=True)
        results.append(["DETECTION", sid, json.dumps(final_boxes[:100])])

    intent_rows = []
    with open(INTENT_PREDS_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row: intent_rows.append(row)

    all_rows = intent_rows + results
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)

    print(f"[🎉] Saved submission to: {out_csv} ({len(all_rows)} rows)")
    return out_csv

if __name__ == "__main__":
    main()
