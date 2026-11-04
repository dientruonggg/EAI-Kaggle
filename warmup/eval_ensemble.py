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
ALPHAS_160 = [0.3, 0.7, 0.2, 1.0, 0.2, 0.7, 0.5, 0.0, 0.7, 0.5]

def get_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    c_det = [d for d in out_dets if d["shape"][-1] == 11][0]
    b_det = [d for d in out_dets if d["shape"][-1] == 4][0]
    return interp, in_det, c_det, b_det

def cache_model_predictions(model_path, val_imgs, grid_size, alphas, priors):
    interp, in_det, c_det, b_det = get_interpreter(model_path)
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

def eval_multi_model(models_config, conf=0.12, iou_thresh=0.35):
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    all_cached = []
    for cfg in models_config:
        name = cfg["name"]
        path = cfg["path"]
        grid = cfg["grid"]
        alphas = cfg["alphas"]
        print(f"[*] Caching predictions for: {name} ({path})...")
        cached = cache_model_predictions(path, val_imgs, grid, alphas, priors)
        all_cached.append((name, cached))

    pred_dict = {}
    for img_p in val_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        raw_boxes = []
        for name, cached in all_cached:
            probs, boxes, grid_size, alphas = cached[sid]
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
            keep = nms(coords, scores, iou_thresh=iou_thresh)
            for k in keep:
                final_boxes.append(c_boxes[k])
                
        final_boxes.sort(key=lambda b: b["score"], reverse=True)
        pred_dict[sid] = final_boxes[:100]

    mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
    print(f"\n=======================================================")
    print(f"ENSEMBLE RESULTS (conf={conf:.2f}, iou={iou_thresh:.2f}) -> mAP@0.5 = {mAP50 * 100:.2f}%")
    print(f"=======================================================")
    for c in range(10):
        print(f"  Class {c} ({CLASS_NAMES[c]}): AP50 = {class_aps.get(c, 0.0) * 100:.2f}%")
    return mAP50, class_aps

if __name__ == "__main__":
    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"
    
    alphas_ciou = [0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.3, 0.5]

    cfg128 = {"name": "M128_Full", "path": m128, "grid": 16, "alphas": ALPHAS_128}
    cfg160_g = {"name": "M160_GIoU", "path": m160_g, "grid": 20, "alphas": ALPHAS_160}
    cfg160_c = {"name": "M160_CIoU", "path": m160_c, "grid": 20, "alphas": alphas_ciou}

    print("\n--- 1. Evaluating 2-Model Ensemble: M128_Full + M160_GIoU ---")
    eval_multi_model([cfg128, cfg160_g], conf=0.12, iou_thresh=0.35)

    print("\n--- 2. Evaluating 2-Model Ensemble: M128_Full + M160_CIoU ---")
    eval_multi_model([cfg128, cfg160_c], conf=0.12, iou_thresh=0.35)

    print("\n--- 3. Evaluating 3-Model Ensemble: M128_Full + M160_GIoU + M160_CIoU ---")
    eval_multi_model([cfg128, cfg160_g, cfg160_c], conf=0.12, iou_thresh=0.35)
    eval_multi_model([cfg128, cfg160_g, cfg160_c], conf=0.10, iou_thresh=0.35)
    eval_multi_model([cfg128, cfg160_g, cfg160_c], conf=0.15, iou_thresh=0.35)
