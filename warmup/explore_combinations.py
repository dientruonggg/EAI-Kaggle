import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import itertools
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

OPT_ALPHAS = [0.5, 0.0, 0.6, 1.0, 0.3, 0.6, 0.4, 0.1, 0.0, 0.2]

def run_eval():
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    models_info = {
        "128_aug": ("/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite", 16),
        "128_ciou": ("/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_128/mobilenet_bbox_ciou_128_int8.tflite", 16),
        "160_giou": ("/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite", 20),
        "160_ciou": ("/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite", 20),
        "160_eiou": ("/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_eiou_160/mobilenet_bbox_eiou_160_int8.tflite", 20),
    }

    cached = {}
    for name, (path, gsize) in models_info.items():
        if os.path.exists(path):
            print(f"Caching {name}...")
            cached[name] = cache_preds(path, val_imgs, gsize, [0.0]*10, priors)
        else:
            print(f"Skipping {name}, not found.")

    def evaluate_model_set(selected_names, conf=0.14, boost=0.03, iou_t=0.28):
        pred_dict = {}
        for img_p in val_imgs:
            sid = os.path.splitext(os.path.basename(img_p))[0]
            raw_boxes = []
            for name in selected_names:
                mod = cached[name]
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
                            a = OPT_ALPHAS[bc]
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

    print("\n--- SINGLE MODELS (conf=0.14) ---")
    for name in cached:
        mAP50, _ = evaluate_model_set([name], conf=0.14)
        print(f"{name:12s}: mAP@0.5 = {mAP50*100:.2f}%")

    print("\n--- TRIPLE COMBINATIONS (conf=0.14, boost=0.03) ---")
    results = []
    for comb in itertools.combinations(cached.keys(), 3):
        mAP50, class_aps = evaluate_model_set(comb, conf=0.14, boost=0.03)
        results.append((mAP50, comb, class_aps))
        print(f"{'+'.join(comb):40s}: mAP@0.5 = {mAP50*100:.2f}%")

    results.sort(reverse=True, key=lambda x: x[0])
    best_map, best_comb, best_aps = results[0]
    print(f"\n>>> BEST TRIPLE: {'+'.join(best_comb)} = {best_map*100:.2f}%")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:20s}: {best_aps[i]*100:.2f}%")

    print("\n--- DOUBLE COMBINATIONS (conf=0.14, boost=0.03) ---")
    for comb in itertools.combinations(cached.keys(), 2):
        mAP50, _ = evaluate_model_set(comb, conf=0.14, boost=0.03)
        print(f"{'+'.join(comb):30s}: mAP@0.5 = {mAP50*100:.2f}%")

if __name__ == "__main__":
    run_eval()
