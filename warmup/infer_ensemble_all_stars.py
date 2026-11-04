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
from src.micro_fomo.infer_bbox import nms

ORIG_WIDTH = 1672.0
ORIG_HEIGHT = 941.0

# 3 Full-Trained INT8 Models + 1 Fine-Tuned Model
M128_FULL_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
M160_FULL_GIOU_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_160/mobilenet_bbox_augmented_full_160_int8.tflite"
M160_FULL_CIOU_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_full_160/mobilenet_bbox_ciou_full_160_int8.tflite"
M160_BEST_GIOU_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
M160_BEST_CIOU_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"

PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
TEST_IMAGES_DIR = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
INTENT_PREDS_PATH = "/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv"
OUTPUT_PATH = "/home/bim/Projects/EAI-2026/warmup/submissions/submission_ensemble_all_stars.csv"

ALPHAS_128 = [0.2, 0.8, 0.3, 0.5, 0.8, 0.5, 0.4, 0.1, 0.2, 0.0]
ALPHAS_160_GIOU = [0.3, 0.7, 0.2, 1.0, 0.2, 0.7, 0.5, 0.0, 0.7, 0.5]
ALPHAS_160_CIOU = [0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.3, 0.5]

def get_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    c_det = [d for d in out_dets if d["shape"][-1] == 11][0]
    b_det = [d for d in out_dets if d["shape"][-1] == 4][0]
    return interp, in_det, c_det, b_det

def predict_single_model(interp, in_det, c_det, b_det, img_rgb, img_size, grid_size, alphas, priors, conf_thresh):
    in_sc, in_zp = in_det.get("quantization", (0.0, 0))
    c_sc, c_zp = c_det.get("quantization", (0.0, 0))
    b_sc, b_zp = b_det.get("quantization", (0.0, 0))

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
            if sc >= conf_thresh:
                p_dx, p_dy, p_w, p_h = float(boxes[gy, gx, 0]), float(boxes[gy, gx, 1]), float(boxes[gy, gx, 2]), float(boxes[gy, gx, 3])
                cx = (gx + p_dx) / float(grid_size)
                cy = (gy + p_dy) / float(grid_size)
                a = alphas[bc]
                pw, ph = priors[str(bc)]["w"], priors[str(bc)]["h"]
                w = (1.0 - a) * pw + a * p_w
                h = (1.0 - a) * ph + a * p_h
                xmin = max(0.0, (cx - w / 2.0) * ORIG_WIDTH)
                ymin = max(0.0, (cy - h / 2.0) * ORIG_HEIGHT)
                xmax = min(ORIG_WIDTH, (cx + w / 2.0) * ORIG_WIDTH)
                ymax = min(ORIG_HEIGHT, (cy + h / 2.0) * ORIG_HEIGHT)
                boxes_list.append({
                    "class_id": bc,
                    "score": round(sc, 4),
                    "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]
                })
    return boxes_list

def run_all_stars_ensemble(model_keys=["128_full", "160_full_giou", "160_full_ciou"], conf_thresh=0.12, iou_thresh=0.35, output_path=OUTPUT_PATH):
    print(f"[*] Initializing All-Stars Ensemble with models: {model_keys}...")
    
    models = {}
    if "128_full" in model_keys and os.path.exists(M128_FULL_PATH):
        interp, in_d, c_d, b_d = get_interpreter(M128_FULL_PATH)
        models["128_full"] = (interp, in_d, c_d, b_d, 128, 16, ALPHAS_128)
        
    if "160_full_giou" in model_keys and os.path.exists(M160_FULL_GIOU_PATH):
        interp, in_d, c_d, b_d = get_interpreter(M160_FULL_GIOU_PATH)
        models["160_full_giou"] = (interp, in_d, c_d, b_d, 160, 20, ALPHAS_160_GIOU)
        
    if "160_full_ciou" in model_keys and os.path.exists(M160_FULL_CIOU_PATH):
        interp, in_d, c_d, b_d = get_interpreter(M160_FULL_CIOU_PATH)
        models["160_full_ciou"] = (interp, in_d, c_d, b_d, 160, 20, ALPHAS_160_CIOU)

    if "160_best_giou" in model_keys and os.path.exists(M160_BEST_GIOU_PATH):
        interp, in_d, c_d, b_d = get_interpreter(M160_BEST_GIOU_PATH)
        models["160_best_giou"] = (interp, in_d, c_d, b_d, 160, 20, ALPHAS_160_GIOU)

    if "160_best_ciou" in model_keys and os.path.exists(M160_BEST_CIOU_PATH):
        interp, in_d, c_d, b_d = get_interpreter(M160_BEST_CIOU_PATH)
        models["160_best_ciou"] = (interp, in_d, c_d, b_d, 160, 20, ALPHAS_160_CIOU)

    with open(PRIORS_PATH) as f:
        priors = json.load(f)

    test_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    print(f"[*] Processing {len(test_images)} test images with All-Stars Ensemble ({len(models)} active models)...")

    results = []
    total_boxes = 0
    for img_p in test_images:
        sample_id = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        if img is None:
            results.append(["DETECTION", sample_id, "[]"])
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        raw_boxes = []
        for name, (interp, in_d, c_d, b_d, s, g, a) in models.items():
            b = predict_single_model(interp, in_d, c_d, b_d, img_rgb, s, g, a, priors, conf_thresh)
            raw_boxes.extend(b)

        final_boxes = []
        for c in range(10):
            c_boxes = [box for box in raw_boxes if box["class_id"] == c]
            if not c_boxes:
                continue
            coords = [box["bbox"] for box in c_boxes]
            scores = [box["score"] for box in c_boxes]
            keep = nms(coords, scores, iou_thresh=iou_thresh)
            for k in keep:
                final_boxes.append(c_boxes[k])

        final_boxes.sort(key=lambda b: b["score"], reverse=True)
        total_boxes += len(final_boxes)
        preds_json = json.dumps(final_boxes[:100])
        results.append(["DETECTION", sample_id, preds_json])

    print(f"[✔] Total boxes: {total_boxes} (avg {total_boxes / len(test_images):.1f} boxes/image)")

    intent_rows = []
    if os.path.exists(INTENT_PREDS_PATH):
        with open(INTENT_PREDS_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row:
                    intent_rows.append(row)
        print(f"[✔] Loaded {len(intent_rows)} Intent predictions.")

    all_rows = intent_rows + results
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)

    print(f"[✔] Successfully created submission: {output_path} ({len(all_rows)} rows)")
    return output_path

if __name__ == "__main__":
    run_all_stars_ensemble()
