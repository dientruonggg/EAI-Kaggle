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

def test_4models():
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"
    m160_full_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_full_160/mobilenet_bbox_ciou_full_160_int8.tflite"

    c128 = cache_preds(m128, val_imgs, 16, ALPHAS_128, priors)
    c160_g = cache_preds(m160_g, val_imgs, 20, ALPHAS_160_G, priors)
    c160_c = cache_preds(m160_c, val_imgs, 20, ALPHAS_160_C, priors)
    c160_fc = cache_preds(m160_full_c, val_imgs, 20, ALPHAS_160_C, priors)
    all_models = [c128, c160_g, c160_c, c160_fc]

    for conf in [0.13, 0.14, 0.15]:
        for iou in [0.28, 0.30]:
            for boost in [0.02, 0.03, 0.04]:
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
                                    raw_boxes.append({"class_id": bc, "score": sc, "bbox": [xmin, ymin, xmax, ymax]})
                    final_boxes = []
                    for c in range(10):
                        c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                        if not c_boxes: continue
                        fused = wbf_with_boost(c_boxes, iou_thresh=iou, boost_factor=boost)
                        final_boxes.extend(fused)
                    final_boxes.sort(key=lambda b: b["score"], reverse=True)
                    pred_dict[sid] = final_boxes[:100]
                mAP50, _ = calculate_map50(gt_dict, pred_dict)
                print(f"[4-Model WBF] conf={conf:.2f}, iou={iou:.2f}, boost={boost:.2f} -> mAP={mAP50*100:.2f}%")

if __name__ == "__main__":
    test_4models()
