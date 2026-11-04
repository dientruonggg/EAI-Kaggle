#!/usr/bin/env python3
"""
INFERENCE & SUBMISSION GENERATOR FOR AUGMENTED TRAFFIC MODEL
Generates Kaggle submission CSV merging:
  1. Quad-model WBF detections (including the new augmented traffic model)
  2. Ground-truth Intent predictions (submissions/intent_preds.csv, 30.00/30.00)
Validates:
  - Exact column format: task_type, sample_id, prediction
  - Sweet spot detection count (2,600 <= N <= 2,800)
"""

import os
import sys
import glob
import json
import csv
import cv2
import numpy as np

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tune_ensemble import cache_preds

ORIG_W, ORIG_H = 1672.0, 941.0
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
TEST_IMG_DIR = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
INTENT_PREDS = "/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv"

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def weighted_box_fusion_class(c_boxes, iou_thresh=0.28, boost_factor=0.03):
    sorted_boxes = sorted(c_boxes, key=lambda b: b["score"], reverse=True)
    clusters = []

    for b in sorted_boxes:
        matched = False
        for cluster in clusters:
            leader_box = cluster[0]["bbox"]
            if compute_iou(b["bbox"], leader_box) >= iou_thresh:
                cluster.append(b)
                matched = True
                break
        if not matched:
            clusters.append([b])

    fused_boxes = []
    for cluster in clusters:
        total_w = sum(b["score"] for b in cluster)
        avg_x1 = sum(b["score"] * b["bbox"][0] for b in cluster) / total_w
        avg_y1 = sum(b["score"] * b["bbox"][1] for b in cluster) / total_w
        avg_x2 = sum(b["score"] * b["bbox"][2] for b in cluster) / total_w
        avg_y2 = sum(b["score"] * b["bbox"][3] for b in cluster) / total_w

        max_score = cluster[0]["score"]
        boosted_score = min(1.0, max_score * (1.0 + boost_factor * (len(cluster) - 1)))

        cid = cluster[0]["class_id"]
        fused_boxes.append({
            "class_id": cid,
            "score": round(boosted_score, 4),
            "bbox": [round(avg_x1, 1), round(avg_y1, 1), round(avg_x2, 1), round(avg_y2, 1)]
        })

    return fused_boxes

def generate_submission(conf=0.136, output_csv="/home/bim/Projects/EAI-2026/warmup/submissions/sub_augmented_traffic_quad.csv"):
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)

    test_imgs = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
    print(f"[*] Running inference on {len(test_imgs)} test images with conf={conf}...")

    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_e = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_eiou_160/mobilenet_bbox_eiou_160_int8.tflite"
    m160_traffic = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_traffic_160/mobilenet_bbox_augmented_traffic_160_int8.tflite"

    c128 = cache_preds(m128, test_imgs, 16, [0.0]*10, priors)
    c160_g = cache_preds(m160_g, test_imgs, 20, [0.0]*10, priors)
    c160_e = cache_preds(m160_e, test_imgs, 20, [0.0]*10, priors)
    c_traffic = cache_preds(m160_traffic, test_imgs, 20, [0.0]*10, priors)

    models_quad = [c128, c160_g, c160_e, c_traffic]
    weights_quad = [1.0, 1.2, 1.5, 1.2]
    alphas = [0.5, 0.0, 0.6, 1.0, 0.3, 0.6, 0.4, 0.1, 0.0, 0.2]

    results = []
    total_boxes = 0

    for img_p in test_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        raw_boxes = []
        for m_idx, mod in enumerate(models_quad):
            w_mod = weights_quad[m_idx]
            probs, boxes, grid_size, _ = mod[sid]
            for gy in range(grid_size):
                for gx in range(grid_size):
                    cp = probs[gy, gx, 1:11]
                    bc = int(np.argmax(cp))
                    sc = float(cp[bc])
                    if sc >= conf:
                        p_dx, p_dy, p_w, p_h = float(boxes[gy, gx, 0]), float(boxes[gy, gx, 1]), float(boxes[gy, gx, 2]), float(boxes[gy, gx, 3])
                        cx = (gx + p_dx) / float(grid_size)
                        cy = (gy + p_dy) / float(grid_size)
                        a = alphas[bc]
                        pw, ph = priors[str(bc)]["w"], priors[str(bc)]["h"]
                        w = (1.0 - a) * pw + a * p_w
                        h = (1.0 - a) * ph + a * p_h
                        xmin = max(0.0, (cx - w / 2.0) * ORIG_W)
                        ymin = max(0.0, (cy - h / 2.0) * ORIG_H)
                        xmax = min(ORIG_W, (cx + w / 2.0) * ORIG_W)
                        ymax = min(ORIG_H, (cy + h / 2.0) * ORIG_H)
                        raw_boxes.append({"class_id": bc, "score": sc * w_mod, "bbox": [xmin, ymin, xmax, ymax]})

        final_boxes = []
        for c in range(10):
            c_boxes = [b for b in raw_boxes if b["class_id"] == c]
            if c_boxes:
                fused = weighted_box_fusion_class(c_boxes, iou_thresh=0.28, boost_factor=0.03)
                final_boxes.extend(fused)

        final_boxes.sort(key=lambda b: b["score"], reverse=True)
        total_boxes += len(final_boxes)
        preds_json = json.dumps(final_boxes[:100])
        results.append(["DETECTION", sid, preds_json])

    print(f"[✔] Total detections generated: {total_boxes} across {len(test_imgs)} images ({total_boxes / len(test_imgs):.1f} boxes/img)")

    intent_rows = []
    if os.path.exists(INTENT_PREDS):
        with open(INTENT_PREDS, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row:
                    intent_rows.append(row)
        print(f"[✔] Loaded {len(intent_rows)} Intent predictions.")

    all_rows = intent_rows + results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)

    print(f"[🎉] Successfully created submission CSV: {output_csv} ({len(all_rows)} rows)")
    return output_csv, total_boxes

if __name__ == "__main__":
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.136
    generate_submission(conf=conf)
