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

OPT_ALPHAS = [0.5, 0.0, 0.6, 1.0, 0.3, 0.6, 0.4, 0.1, 0.0, 0.2]

def eval_student():
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    student_tflite = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_self_training_160/mobilenet_bbox_self_training_160_int8.tflite"
    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"

    if not os.path.exists(student_tflite):
        print(f"[-] Student model {student_tflite} not ready yet.")
        return

    print("[*] Caching predictions for Student and champion models...")
    c_student = cache_preds(student_tflite, val_imgs, 20, [0.0]*10, priors)
    c128 = cache_preds(m128, val_imgs, 16, [0.0]*10, priors)
    c160_g = cache_preds(m160_g, val_imgs, 20, [0.0]*10, priors)
    c160_c = cache_preds(m160_c, val_imgs, 20, [0.0]*10, priors)

    def evaluate(models, conf=0.14, boost=0.03, iou_t=0.28):
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

    print("\n--- STANDALONE STUDENT (conf=0.14) ---")
    map_stud, aps_stud = evaluate([c_student], conf=0.14)
    print(f"Student Standalone: mAP@0.5 = {map_stud*100:.2f}%")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:20s}: {aps_stud[i]*100:.2f}%")

    print("\n--- ENSEMBLE WITH STUDENT: 128_aug + 160_giou + Student_160 ---")
    for conf in [0.13, 0.14, 0.15]:
        map_ens, aps_ens = evaluate([c128, c160_g, c_student], conf=conf)
        print(f"Ensemble with Student (conf={conf:.2f}): mAP@0.5 = {map_ens*100:.2f}%")

    print("\n--- ENSEMBLE QUAD: 128_aug + 160_giou + 160_ciou + Student_160 ---")
    for conf in [0.14, 0.15, 0.16]:
        map_quad, _ = evaluate([c128, c160_g, c160_c, c_student], conf=conf)
        print(f"Quad Ensemble with Student (conf={conf:.2f}): mAP@0.5 = {map_quad*100:.2f}%")

if __name__ == "__main__":
    eval_student()
