import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
from src.evaluation import load_ground_truth, calculate_map50
from tune_ensemble import cache_preds
from tune_wbf import wbf_with_boost

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W, ORIG_H = 1672.0, 941.0
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
CLASS_NAMES = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"]

def main():
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_e = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_eiou_160/mobilenet_bbox_eiou_160_int8.tflite"

    c128 = cache_preds(m128, val_imgs, 16, [0.0]*10, priors)
    c160_g = cache_preds(m160_g, val_imgs, 20, [0.0]*10, priors)
    c160_e = cache_preds(m160_e, val_imgs, 20, [0.0]*10, priors)
    models = [c128, c160_g, c160_e]

    current_alphas = [0.5, 0.0, 0.6, 1.0, 0.3, 0.6, 0.4, 0.1, 0.0, 0.2]
    conf = 0.13
    iou_t = 0.28
    boost = 0.04

    def eval_alphas(alphas):
        pred_dict = {}
        for img_p in val_imgs:
            sid = os.path.splitext(os.path.basename(img_p))[0]
            raw_boxes = []
            for mod in models:
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
                            raw_boxes.append({"class_id": bc, "score": sc, "bbox": [xmin, ymin, xmax, ymax]})
            final_boxes = []
            for c in range(10):
                c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                if not c_boxes: continue
                fused = wbf_with_boost(c_boxes, iou_thresh=iou_t, boost_factor=boost)
                final_boxes.extend(fused)
            final_boxes.sort(key=lambda b: b["score"], reverse=True)
            pred_dict[sid] = final_boxes[:100]
        mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
        return mAP50, class_aps

    base_map, base_aps = eval_alphas(current_alphas)
    print(f"Base Alphas mAP@0.5 = {base_map*100:.2f}%")

    improved = True
    iteration = 0
    while improved and iteration < 3:
        improved = False
        iteration += 1
        print(f"\n--- Coordinate Descent Iteration {iteration} ---")
        for c in range(10):
            best_c_alpha = current_alphas[c]
            best_c_map = base_map
            for cand_a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if cand_a == current_alphas[c]:
                    continue
                test_alphas = list(current_alphas)
                test_alphas[c] = cand_a
                mAP, aps = eval_alphas(test_alphas)
                if mAP > best_c_map + 1e-4:
                    best_c_map = mAP
                    best_c_alpha = cand_a
            if best_c_alpha != current_alphas[c]:
                print(f"  Class {c} ({CLASS_NAMES[c]}): alpha {current_alphas[c]} -> {best_c_alpha} (mAP: {base_map*100:.2f}% -> {best_c_map*100:.2f}%)")
                current_alphas[c] = best_c_alpha
                base_map = best_c_map
                improved = True

    print(f"\n>>> FINAL OPTIMIZED ALPHAS: {current_alphas}")
    print(f">>> FINAL mAP@0.5: {base_map*100:.2f}%")
    final_map, final_aps = eval_alphas(current_alphas)
    for i, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:20s}: {final_aps[i]*100:.2f}% (alpha={current_alphas[i]})")

if __name__ == "__main__":
    main()
