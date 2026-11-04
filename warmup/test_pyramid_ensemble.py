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
from tune_ensemble import cache_preds, ALPHAS_128, ALPHAS_160_C, get_interpreter
from test_wbf import compute_iou
from tune_wbf import wbf_with_boost

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W, ORIG_H = 1672.0, 941.0
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
CLASS_NAMES = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"]

ALPHAS_96 = [0.2, 0.8, 0.3, 0.5, 0.8, 0.5, 0.4, 0.1, 0.2, 0.0]

def eval_multi_resolution_pyramid():
    with open(PRIORS_PATH, "r") as f: priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f: val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    m96 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_96/mobilenet_bbox_96_int8.tflite"
    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"

    models_list = []
    if os.path.exists(m96):
        print(f"[*] Caching M96_CIoU (grid 12x12)...")
        models_list.append(cache_preds(m96, val_imgs, 12, ALPHAS_96, priors))
    if os.path.exists(m128):
        print(f"[*] Caching M128_Full (grid 16x16)...")
        models_list.append(cache_preds(m128, val_imgs, 16, ALPHAS_128, priors))
    if os.path.exists(m160_c):
        print(f"[*] Caching M160_CIoU (grid 20x20)...")
        models_list.append(cache_preds(m160_c, val_imgs, 20, ALPHAS_160_C, priors))

    print(f"\n=== Evaluating Multi-Resolution Pyramid (96x96 + 128x128 + 160x160) with WBF ===")

    for conf in [0.12, 0.13, 0.14, 0.15]:
        pred_dict = {}
        for img_p in val_imgs:
            sid = os.path.splitext(os.path.basename(img_p))[0]
            raw_boxes = []
            for mod in models_list:
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
                            raw_boxes.append({"class_id": bc, "score": sc, "bbox": [xmin, ymin, xmax, ymax]})
            final_boxes = []
            for c in range(10):
                c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                if not c_boxes: continue
                fused = wbf_with_boost(c_boxes, iou_thresh=0.28, boost_factor=0.03)
                final_boxes.extend(fused)
            final_boxes.sort(key=lambda b: b["score"], reverse=True)
            pred_dict[sid] = final_boxes[:100]

        mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
        print(f"[Pyramid WBF] conf={conf:.2f} -> mAP@0.5 = {mAP50 * 100:.2f}%")
        if conf == 0.14:
            for c in range(10):
                print(f"  Class {c} ({CLASS_NAMES[c]}): AP50 = {class_aps.get(c, 0.0) * 100:.2f}%")

if __name__ == "__main__":
    eval_multi_resolution_pyramid()
