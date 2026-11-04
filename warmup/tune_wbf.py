import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import cv2
import numpy as np
import tensorflow as tf
from src.evaluation import load_ground_truth, calculate_map50
from tune_ensemble import cache_preds, ALPHAS_128, ALPHAS_160_G, ALPHAS_160_C
from test_wbf import compute_iou

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W, ORIG_H = 1672.0, 941.0
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"

def wbf_with_boost(c_boxes, iou_thresh=0.35, boost_factor=0.15):
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
            "score": boosted_score,
            "bbox": [avg_x1, avg_y1, avg_x2, avg_y2]
        })
        
    return fused_boxes

def tune_wbf():
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"

    c128 = cache_preds(m128, val_imgs, 16, ALPHAS_128, priors)
    c160_g = cache_preds(m160_g, val_imgs, 20, ALPHAS_160_G, priors)
    c160_c = cache_preds(m160_c, val_imgs, 20, ALPHAS_160_C, priors)
    all_models = [c128, c160_g, c160_c]

    best_mAP = 0.0
    best_cfg = None

    for conf in [0.10, 0.11, 0.12, 0.13]:
        for iou in [0.30, 0.33, 0.35, 0.38]:
            for boost in [0.05, 0.10, 0.15, 0.20]:
                pred_dict = {}
                for img_p in val_imgs:
                    sid = os.path.splitext(os.path.basename(img_p))[0]
                    raw_boxes = []
                    for mod in all_models:
                        probs, boxes, grid_size, alphas = mod[sid]
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
                                    raw_boxes.append({
                                        "class_id": bc,
                                        "score": sc,
                                        "bbox": [xmin, ymin, xmax, ymax]
                                    })

                    final_boxes = []
                    for c in range(10):
                        c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                        if not c_boxes: continue
                        fused = wbf_with_boost(c_boxes, iou_thresh=iou, boost_factor=boost)
                        final_boxes.extend(fused)

                    final_boxes.sort(key=lambda b: b["score"], reverse=True)
                    pred_dict[sid] = final_boxes[:100]

                mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
                if mAP50 > best_mAP:
                    best_mAP = mAP50
                    best_cfg = (conf, iou, boost, class_aps)
                print(f"conf={conf:.2f}, iou={iou:.2f}, boost={boost:.2f} -> mAP@0.5 = {mAP50 * 100:.2f}%")

    conf, iou, boost, aps = best_cfg
    print(f"\n[★ OPTIMAL WBF PARAMETERS] conf={conf:.2f}, iou={iou:.2f}, boost={boost:.2f} -> mAP@0.5 = {best_mAP * 100:.2f}%")
    CLASS_NAMES = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"]
    for c in range(10):
        print(f"  Class {c} ({CLASS_NAMES[c]}): AP50 = {aps.get(c, 0.0) * 100:.2f}%")

if __name__ == "__main__":
    tune_wbf()
