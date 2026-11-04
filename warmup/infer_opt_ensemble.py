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

ORIG_WIDTH = 1672.0
ORIG_HEIGHT = 941.0

M128_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
M160_GIOU_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
M160_CIOU_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
TEST_IMAGES_DIR = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
INTENT_PREDS_PATH = "/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv"

# Physically calibrated optimal blending alphas (Val mAP@0.5 = 68.55%)
OPT_ALPHAS = [0.5, 0.0, 0.6, 1.0, 0.3, 0.6, 0.4, 0.1, 0.0, 0.2]

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
                    "score": sc,
                    "bbox": [xmin, ymin, xmax, ymax]
                })
    return boxes_list

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def weighted_box_fusion_class(c_boxes, iou_thresh=0.28, boost_factor=0.04):
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
            "score": round(boosted_score, 4),
            "bbox": [round(avg_x1, 1), round(avg_y1, 1), round(avg_x2, 1), round(avg_y2, 1)]
        })

    return fused_boxes

def run_wbf_ensemble(conf_thresh=0.15, iou_thresh=0.28, boost_factor=0.04, output_path=None):
    if output_path is None:
        output_path = f"/home/bim/Projects/EAI-2026/warmup/submissions/sub_opt_alpha_c{int(conf_thresh*100)}.csv"

    print(f"[*] Initializing Triple Ensemble with Optimized Alphas (conf={conf_thresh}, iou={iou_thresh}, boost={boost_factor})...")
    interp128, in128, c128, b_det128 = get_interpreter(M128_PATH)
    interp160_giou, in160_g, c160_g, b_det160_g = get_interpreter(M160_GIOU_PATH)
    interp160_ciou, in160_c, c160_c, b_det160_c = get_interpreter(M160_CIOU_PATH)

    with open(PRIORS_PATH) as f:
        priors = json.load(f)

    test_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    print(f"[*] Processing {len(test_images)} test images...")

    results = []
    total_boxes = 0
    for img_p in test_images:
        sample_id = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        if img is None:
            results.append(["DETECTION", sample_id, "[]"])
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes_128 = predict_single_model(interp128, in128, c128, b_det128, img_rgb, 128, 16, OPT_ALPHAS, priors, conf_thresh)
        boxes_160_g = predict_single_model(interp160_giou, in160_g, c160_g, b_det160_g, img_rgb, 160, 20, OPT_ALPHAS, priors, conf_thresh)
        boxes_160_c = predict_single_model(interp160_ciou, in160_c, c160_c, b_det160_c, img_rgb, 160, 20, OPT_ALPHAS, priors, conf_thresh)

        raw_boxes = boxes_128 + boxes_160_g + boxes_160_c

        final_boxes = []
        for c in range(10):
            c_boxes = [b for b in raw_boxes if b["class_id"] == c]
            if not c_boxes:
                continue
            fused = weighted_box_fusion_class(c_boxes, iou_thresh=iou_thresh, boost_factor=boost_factor)
            final_boxes.extend(fused)

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)

    print(f"[✔] Successfully created submission: {output_path} ({len(all_rows)} rows)")
    return output_path

if __name__ == "__main__":
    run_wbf_ensemble(conf_thresh=0.15, iou_thresh=0.28, boost_factor=0.04)
