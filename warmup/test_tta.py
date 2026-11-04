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
from tune_ensemble import cache_preds, ALPHAS_128, ALPHAS_160_G, ALPHAS_160_C, get_interpreter
from test_wbf import compute_iou
from tune_wbf import wbf_with_boost

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W, ORIG_H = 1672.0, 941.0
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
CLASS_NAMES = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"]

def predict_image(interp, in_det, c_det, b_det, img_rgb, img_size, grid_size, alphas, priors, conf=0.14, flip=False):
    in_sc, in_zp = in_det.get("quantization", (0.0, 0))
    c_sc, c_zp = c_det.get("quantization", (0.0, 0))
    b_sc, b_zp = b_det.get("quantization", (0.0, 0))

    if flip:
        img_rgb = cv2.flip(img_rgb, 1)

    resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    in_tensor = np.round(norm / in_sc + in_zp).astype(np.int8)[None, ...]
    interp.set_tensor(in_det["index"], in_tensor)
    interp.invoke()

    probs = (interp.get_tensor(c_det["index"])[0].astype(np.float32) - c_zp) * c_sc
    boxes = (interp.get_tensor(b_det["index"])[0].astype(np.float32) - b_zp) * b_sc

    boxes_list = []
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

                if flip:
                    # Invert horizontal coordinates back
                    real_xmin = ORIG_W - xmax
                    real_xmax = ORIG_W - xmin
                    xmin, xmax = real_xmin, real_xmax

                boxes_list.append({
                    "class_id": bc,
                    "score": round(sc, 4),
                    "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]
                })
    return boxes_list

def test_tta_on_val():
    with open(PRIORS_PATH, "r") as f: priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f: val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"

    i128, in128, c128, b128 = get_interpreter(m128)
    i160_g, in160_g, c160_g, b160_g = get_interpreter(m160_g)
    i160_c, in160_c, c160_c, b160_c = get_interpreter(m160_c)

    models_info = [
        (i128, in128, c128, b128, 128, 16, ALPHAS_128),
        (i160_g, in160_g, c160_g, b160_g, 160, 20, ALPHAS_160_G),
        (i160_c, in160_c, c160_c, b160_c, 160, 20, ALPHAS_160_C),
    ]

    for use_tta in [False, True]:
        for conf in [0.13, 0.14, 0.15]:
            pred_dict = {}
            for img_p in val_imgs:
                sid = os.path.splitext(os.path.basename(img_p))[0]
                img = cv2.imread(img_p)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                raw_boxes = []

                for interp, in_d, c_d, b_d, sz, grid, alphas in models_info:
                    # Original pass
                    b_orig = predict_image(interp, in_d, c_d, b_d, img_rgb, sz, grid, alphas, priors, conf=conf, flip=False)
                    raw_boxes.extend(b_orig)

                    if use_tta:
                        # Flipped TTA pass
                        b_flip = predict_image(interp, in_d, c_d, b_d, img_rgb, sz, grid, alphas, priors, conf=conf, flip=True)
                        raw_boxes.extend(b_flip)

                final_boxes = []
                for c in range(10):
                    c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                    if not c_boxes: continue
                    fused = wbf_with_boost(c_boxes, iou_thresh=0.28, boost_factor=0.03)
                    final_boxes.extend(fused)

                final_boxes.sort(key=lambda b: b["score"], reverse=True)
                pred_dict[sid] = final_boxes[:100]

            mAP50, _ = calculate_map50(gt_dict, pred_dict)
            mode_str = "WITH TTA" if use_tta else "WITHOUT TTA"
            print(f"[{mode_str}] conf={conf:.2f} -> mAP@0.5 = {mAP50 * 100:.2f}%")

if __name__ == "__main__":
    test_tta_on_val()
