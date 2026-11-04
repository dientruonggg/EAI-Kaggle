#!/usr/bin/env python3
"""
EVALUATION & CALIBRATION SCRIPT FOR AUGMENTED TRAFFIC DETECTOR
Evaluates:
  warmup/weights/mobilenet_bbox_augmented_traffic_160/mobilenet_bbox_augmented_traffic_160_int8.tflite
on the clean validation set (warmup/data/val_split.txt).
Tests standalone mAP50 & quad-model WBF ensemble.
"""

import os
import sys
import json
import cv2
import numpy as np

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation import load_ground_truth, calculate_map50
from tune_ensemble import cache_preds
from tune_wbf import wbf_with_boost

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W, ORIG_H = 1672.0, 941.0
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
CLASS_NAMES = [
    "person", "bicycle", "motorcycle", "car", "bus",
    "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"
]

def eval_single_model(tflite_path, val_imgs, gt_dict, priors, confs=[0.10, 0.12, 0.14, 0.16, 0.18, 0.20]):
    print(f"\n=======================================================")
    print(f"[*] Evaluating Single Model: {os.path.basename(tflite_path)}")
    print(f"=======================================================")
    
    cached = cache_preds(tflite_path, val_imgs, 20, [0.0]*10, priors)
    best_map = -1.0
    best_conf = 0.15
    best_breakdown = {}

    for c in confs:
        preds = {}
        for img_p in val_imgs:
            sid = os.path.splitext(os.path.basename(img_p))[0]
            probs, boxes, grid_size, _ = cached[sid]
            img_preds = []
            for gy in range(grid_size):
                for gx in range(grid_size):
                    cp = probs[gy, gx, 1:11]
                    bc = int(np.argmax(cp))
                    sc = float(cp[bc])
                    if sc >= c:
                        p_dx, p_dy, p_w, p_h = float(boxes[gy, gx, 0]), float(boxes[gy, gx, 1]), float(boxes[gy, gx, 2]), float(boxes[gy, gx, 3])
                        cx = (gx + p_dx) / float(grid_size)
                        cy = (gy + p_dy) / float(grid_size)
                        pw, ph = priors[str(bc)]["w"], priors[str(bc)]["h"]
                        # Blend prior
                        w = 0.5 * pw + 0.5 * p_w
                        h = 0.5 * ph + 0.5 * p_h
                        xmin = (cx - w / 2.0) * ORIG_W
                        ymin = (cy - h / 2.0) * ORIG_H
                        xmax = (cx + w / 2.0) * ORIG_W
                        ymax = (cy + h / 2.0) * ORIG_H
                        img_preds.append({
                            "class_id": bc,
                            "score": sc,
                            "bbox": [xmin, ymin, xmax, ymax]
                        })
            preds[sid] = img_preds
        
        m_map, class_aps = calculate_map50(gt_dict, preds)
        print(f"  Conf = {c:.2f} -> mAP@0.5: {m_map * 100:.2f}%")
        if m_map > best_map:
            best_map = m_map
            best_conf = c
            best_breakdown = class_aps

    print(f"\n[✔] BEST STANDALONE RESULT:")
    print(f"    Optimal Confidence: {best_conf:.2f}")
    print(f"    mAP@0.5: {best_map * 100:.2f}%\n")
    print(f"    Per-class Breakdown:")
    for cid in range(10):
        print(f"      [{cid}] {CLASS_NAMES[cid]:<24}: {best_breakdown.get(cid, 0.0) * 100:.2f}%")
    return best_map, best_conf

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

def eval_ensemble_with_new_model(new_tflite, val_imgs, gt_dict, priors):
    print(f"\n=======================================================")
    print(f"[*] Testing Ensemble Integration (Quad-Model WBF)")
    print(f"=======================================================")
    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_e = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_eiou_160/mobilenet_bbox_eiou_160_int8.tflite"

    c128 = cache_preds(m128, val_imgs, 16, [0.0]*10, priors)
    c160_g = cache_preds(m160_g, val_imgs, 20, [0.0]*10, priors)
    c160_e = cache_preds(m160_e, val_imgs, 20, [0.0]*10, priors)
    c_new = cache_preds(new_tflite, val_imgs, 20, [0.0]*10, priors)

    models_quad = [c128, c160_g, c160_e, c_new]
    weights_quad = [1.0, 1.2, 1.5, 1.2]
    alphas = [0.5, 0.0, 0.6, 1.0, 0.3, 0.6, 0.4, 0.1, 0.0, 0.2]

    for conf in [0.10, 0.12, 0.13, 0.136, 0.14, 0.15]:
        pred_dict = {}
        for img_p in val_imgs:
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
                            xmin = (cx - w / 2.0) * ORIG_W
                            ymin = (cy - h / 2.0) * ORIG_H
                            xmax = (cx + w / 2.0) * ORIG_W
                            ymax = (cy + h / 2.0) * ORIG_H
                            raw_boxes.append({"class_id": bc, "score": sc * w_mod, "bbox": [xmin, ymin, xmax, ymax]})
            
            final_boxes = []
            for c in range(10):
                c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                if c_boxes:
                    fused = weighted_box_fusion_class(c_boxes, iou_thresh=0.28, boost_factor=0.03)
                    final_boxes.extend(fused)
            pred_dict[sid] = final_boxes
        
        m_map, class_aps = calculate_map50(gt_dict, pred_dict)
        print(f"  Quad-WBF Ensemble (conf={conf:.3f}) -> mAP@0.5: {m_map * 100:.2f}%")

def main():
    tflite_path = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_traffic_160/mobilenet_bbox_augmented_traffic_160_int8.tflite"
    if not os.path.exists(tflite_path):
        print(f"[!] Error: Model not found at {tflite_path}")
        return

    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    eval_single_model(tflite_path, val_imgs, gt_dict, priors)
    eval_ensemble_with_new_model(tflite_path, val_imgs, gt_dict, priors)

if __name__ == "__main__":
    main()
