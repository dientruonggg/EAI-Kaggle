import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import cv2
import numpy as np
import tensorflow as tf
from src.micro_fomo.infer_bbox import nms
from src.evaluation import load_ground_truth, calculate_map50

CLASS_NAMES = [
    "person", "bicycle", "motorcycle", "car", "bus",
    "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"
]
VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W, ORIG_H = 1672.0, 941.0
TFLITE_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_fomo_128/mobilenet_fomo_128_int8.tflite"
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"

def evaluate_128():
    if not os.path.exists(TFLITE_PATH):
        print(f"File not found: {TFLITE_PATH}")
        return

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_shape = input_details["shape"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero = output_details.get("quantization", (0.0, 0))

    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    grid_size = output_details["shape"][1]
    print(f"[*] Model loaded. Grid size = {grid_size}x{grid_size}, in_shape = {in_shape}")

    # Cache outputs for all validation images
    cached_probs = {}
    for img_p in val_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (in_shape[1], in_shape[2]), interpolation=cv2.INTER_AREA)
        norm = (img_resized.astype(np.float32) / 255.0)
        in_tensor = np.round(norm / in_scale + in_zero).astype(np.int8)
        in_tensor = np.expand_dims(in_tensor, axis=0)
        interpreter.set_tensor(input_details["index"], in_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]
        probs = (output_data.astype(np.float32) - out_zero) * out_scale
        cached_probs[sid] = probs

    print("\n=== EVALUATING 128x128 SWEEP OVER CONF & IOU THRESHOLDS ===")
    best_mAP = 0.0
    best_cfg = None

    for conf in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        for iou in [0.40, 0.45, 0.50]:
            pred_dict = {}
            for sid, probs in cached_probs.items():
                raw_boxes = []
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        cell_probs = probs[gy, gx, 1:11]
                        best_cls = int(np.argmax(cell_probs))
                        score = float(cell_probs[best_cls])
                        if score >= conf:
                            cx = (gx + 0.5) / float(grid_size)
                            cy = (gy + 0.5) / float(grid_size)
                            cls_prior = priors.get(str(best_cls), {"w": 0.1, "h": 0.1})
                            pw, ph = cls_prior["w"], cls_prior["h"]
                            xmin = max(0.0, (cx - pw / 2.0) * ORIG_W)
                            ymin = max(0.0, (cy - ph / 2.0) * ORIG_H)
                            xmax = min(ORIG_W, (cx + pw / 2.0) * ORIG_W)
                            ymax = min(ORIG_H, (cy + ph / 2.0) * ORIG_H)
                            raw_boxes.append({"class_id": best_cls, "score": score, "bbox": [xmin, ymin, xmax, ymax]})
                final_boxes = []
                for c in range(10):
                    c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                    if not c_boxes: continue
                    coords = [b["bbox"] for b in c_boxes]
                    scores = [b["score"] for b in c_boxes]
                    keep = nms(coords, scores, iou_thresh=iou)
                    for k in keep:
                        final_boxes.append(c_boxes[k])
                final_boxes.sort(key=lambda b: b["score"], reverse=True)
                pred_dict[sid] = final_boxes[:100]

            mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
            if mAP50 > best_mAP:
                best_mAP = mAP50
                best_cfg = (conf, iou, class_aps)
            print(f"conf={conf:.2f}, iou={iou:.2f} -> mAP@0.5 = {mAP50 * 100:.2f}%")

    if best_cfg:
        conf, iou, aps = best_cfg
        print(f"\n[★ BEST CONFIG 128x128] conf={conf:.2f}, iou={iou:.2f} with mAP@0.5 = {best_mAP * 100:.2f}%")
        for c in range(10):
            print(f"  Class {c} ({CLASS_NAMES[c]}): AP50 = {aps.get(c, 0.0) * 100:.2f}%")

if __name__ == "__main__":
    evaluate_128()
