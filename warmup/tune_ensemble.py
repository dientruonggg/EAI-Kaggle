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
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"

ALPHAS_128 = [0.2, 0.8, 0.3, 0.5, 0.8, 0.5, 0.4, 0.1, 0.2, 0.0]
ALPHAS_160_G = [0.3, 0.7, 0.2, 1.0, 0.2, 0.7, 0.5, 0.0, 0.7, 0.5]
ALPHAS_160_C = [0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.3, 0.5]

def get_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    c_det = [d for d in out_dets if d["shape"][-1] == 11][0]
    b_det = [d for d in out_dets if d["shape"][-1] == 4][0]
    return interp, in_det, c_det, b_det

def cache_preds(path, val_imgs, grid_size, alphas, priors):
    interp, in_det, c_det, b_det = get_interpreter(path)
    in_shape = in_det["shape"]
    in_sc, in_zp = in_det.get("quantization", (0.0, 0))
    c_sc, c_zp = c_det.get("quantization", (0.0, 0))
    b_sc, b_zp = b_det.get("quantization", (0.0, 0))

    img_size = in_shape[1]
    predictions = {}
    for img_p in val_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        in_tensor = np.round(norm / in_sc + in_zp).astype(np.int8)[None, ...]
        
        interp.set_tensor(in_det["index"], in_tensor)
        interp.invoke()
        
        raw_cls = interp.get_tensor(c_det["index"])[0]
        raw_box = interp.get_tensor(b_det["index"])[0]
        
        probs = (raw_cls.astype(np.float32) - c_zp) * c_sc
        boxes = (raw_box.astype(np.float32) - b_zp) * b_sc
        
        predictions[sid] = (probs, boxes, grid_size, alphas)
    return predictions

def tune():
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"

    c128 = cache_preds(m128, val_imgs, 16, ALPHAS_128, priors)
    c160_g = cache_preds(m160_g, val_imgs, 20, ALPHAS_160_G, priors)
    c160_c = cache_preds(m160_c, val_imgs, 20, ALPHAS_160_C, priors)

    all_models = [c128, c160_g, c160_c]

    best_mAP = 0.0
    best_params = None

    for conf in [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]:
        for iou in [0.25, 0.30, 0.35, 0.40, 0.45]:
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
                                raw_boxes.append({
                                    "class_id": bc,
                                    "score": sc,
                                    "bbox": [xmin, ymin, xmax, ymax]
                                })

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
                best_params = (conf, iou, class_aps)
            print(f"conf={conf:.2f}, iou={iou:.2f} -> mAP@0.5 = {mAP50 * 100:.2f}%")

    conf, iou, aps = best_params
    print(f"\n[★ GLOBAL BEST ENSEMBLE CONFIG] conf={conf:.2f}, iou={iou:.2f} -> mAP@0.5 = {best_mAP * 100:.2f}%")
    for c in range(10):
        print(f"  Class {c} ({CLASS_NAMES[c]}): AP50 = {aps.get(c, 0.0) * 100:.2f}%")

if __name__ == "__main__":
    tune()
