import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import json
import csv
import cv2
import numpy as np
import tensorflow as tf
from infer_calibrated_champion import get_interpreter, predict_single_model, weighted_box_fusion_class, M128_PATH, M160_GIOU_PATH, PRIORS_PATH, TEST_IMAGES_DIR, INTENT_PREDS_PATH, OPT_ALPHAS

STUDENT_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_self_training_160/mobilenet_bbox_self_training_160_int8.tflite"

def run_submission(conf_thresh=0.14, output_path="/home/bim/Projects/EAI-2026/warmup/submissions/sub_student_c14.csv"):
    print(f"[*] Generating Student Self-Training Ensemble Submission (conf={conf_thresh})...")
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)

    i128, in128, c128, b128 = get_interpreter(M128_PATH)
    ig, ing, cg, bg = get_interpreter(M160_GIOU_PATH)
    istud, instud, cstud, bstud = get_interpreter(STUDENT_PATH)

    test_imgs = sorted(glob.glob(TEST_IMAGES_DIR + "/*.png"))
    print(f"[*] Processing {len(test_imgs)} test images...")

    results = []
    total_boxes = 0

    for idx, img_p in enumerate(test_imgs):
        sample_id = os.path.splitext(os.path.basename(img_p))[0]
        img_bgr = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        b_128 = predict_single_model(i128, in128, c128, b128, img_rgb, 128, 16, OPT_ALPHAS, priors, min_conf=conf_thresh)
        b_160g = predict_single_model(ig, ing, cg, bg, img_rgb, 160, 20, OPT_ALPHAS, priors, min_conf=conf_thresh)
        b_stud = predict_single_model(istud, instud, cstud, bstud, img_rgb, 160, 20, OPT_ALPHAS, priors, min_conf=conf_thresh)

        raw = b_128 + b_160g + b_stud
        final = []
        for c in range(10):
            cb = [b for b in raw if b["class_id"] == c]
            if not cb:
                continue
            fused = weighted_box_fusion_class(cb, iou_thresh=0.28, boost_factor=0.03)
            final.extend(fused)

        final.sort(key=lambda b: b["score"], reverse=True)
        total_boxes += len(final)
        results.append(["DETECTION", sample_id, json.dumps(final[:100])])

    print(f"[✔] Total boxes: {total_boxes} (avg {total_boxes/len(test_imgs):.1f} boxes/image)")

    intent_rows = []
    if os.path.exists(INTENT_PREDS_PATH):
        with open(INTENT_PREDS_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            intent_rows = list(reader)
        print(f"[✔] Loaded {len(intent_rows)} Intent predictions.")

    all_rows = intent_rows + results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)

    print(f"[✔] Successfully created submission: {output_path} ({len(all_rows)} rows)")
    return output_path

if __name__ == "__main__":
    run_submission(conf_thresh=0.14)
