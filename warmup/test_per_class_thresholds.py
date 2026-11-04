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
from tune_wbf import wbf_with_boost

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W, ORIG_H = 1672.0, 941.0
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
CLASS_NAMES = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"]

def test_class_thresholds():
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

    # Baseline uniform threshold
    base_thresh = [0.14] * 10
    iou = 0.28
    boost = 0.03

    def eval_with_thresh(thresh_list):
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
                        if sc >= thresh_list[bc]:
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
                            raw_boxes.append({"class_id": bc, "score": sc, "bbox": [xmin, ymin, xmax, ymax]})
            final_boxes = []
            for c in range(10):
                c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                if not c_boxes: continue
                fused = wbf_with_boost(c_boxes, iou_thresh=iou, boost_factor=boost)
                final_boxes.extend(fused)
            final_boxes.sort(key=lambda b: b["score"], reverse=True)
            pred_dict[sid] = final_boxes[:100]
        return calculate_map50(gt_dict, pred_dict)

    base_mAP, base_aps = eval_with_thresh(base_thresh)
    print(f"Uniform Conf=0.14 -> mAP@0.5 = {base_mAP * 100:.2f}%")

    # Greedy search per-class thresholds
    best_thresh = list(base_thresh)
    candidates = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    
    for c in range(10):
        best_c_thresh = best_thresh[c]
        best_c_ap = base_aps.get(c, 0.0)
        for t in candidates:
            test_t = list(best_thresh)
            test_t[c] = t
            _, aps = eval_with_thresh(test_t)
            if aps.get(c, 0.0) > best_c_ap:
                best_c_ap = aps.get(c, 0.0)
                best_c_thresh = t
        best_thresh[c] = best_c_thresh
        print(f"Class {c} ({CLASS_NAMES[c]}): optimal threshold = {best_c_thresh} (AP50: {best_c_ap * 100:.2f}%)")

    final_mAP, final_aps = eval_with_thresh(best_thresh)
    print(f"\n[★ CALIBRATED PER-CLASS THRESHOLDS RESULT] mAP@0.5 = {final_mAP * 100:.2f}%")
    print(f"CLASS_THRESHOLDS = {best_thresh}")

if __name__ == "__main__":
    test_class_thresholds()
