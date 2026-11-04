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

# Optimal per-class blend weights between canonical prior and learned dynamic bbox
# alpha=0.0 means 100% prior (rigid objects), alpha=0.8 means 80% dynamic (scale-varying cars)
CLASS_ALPHAS = [0.1, 0.2, 0.0, 0.8, 0.0, 0.4, 0.3, 0.0, 0.3, 0.0]

def infer_mobilenet_bbox_giou(
    tflite_path="/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_giou_128/mobilenet_bbox_giou_128_int8.tflite",
    test_images_dir="/home/bim/Projects/EAI-2026/warmup/data/test/images",
    priors_path="/home/bim/Projects/EAI-2026/warmup/data/class_priors.json",
    conf_threshold=0.15,
    iou_thresh=0.35
):
    print(f"[*] Initializing Dual-Head TFLite Interpreter from: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()
    
    cls_det = None
    box_det = None
    for d in out_details:
        if d["shape"][-1] == 11:
            cls_det = d
        elif d["shape"][-1] == 4:
            box_det = d
            
    in_shape = input_details["shape"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    cls_scale, cls_zero = cls_det.get("quantization", (0.0, 0))
    box_scale, box_zero = box_det.get("quantization", (0.0, 0))
    grid_size = cls_det["shape"][1]
    
    with open(priors_path, "r") as f:
        priors = json.load(f)
        
    test_images = sorted(glob.glob(os.path.join(test_images_dir, "*.png")))
    print(f"[*] Processing {len(test_images)} test images (conf={conf_threshold}, iou={iou_thresh}, grid={grid_size})")
    
    results = []
    
    for img_p in test_images:
        sample_id = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        if img is None:
            results.append(["DETECTION", sample_id, "[]"])
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (in_shape[1], in_shape[2]), interpolation=cv2.INTER_AREA)
        
        norm = (img_resized.astype(np.float32) / 255.0)
        in_tensor = np.round(norm / in_scale + in_zero).astype(np.int8)
        in_tensor = np.expand_dims(in_tensor, axis=0)
        
        interpreter.set_tensor(input_details["index"], in_tensor)
        interpreter.invoke()
        
        raw_cls = interpreter.get_tensor(cls_det["index"])[0]
        raw_box = interpreter.get_tensor(box_det["index"])[0]
        
        probs = (raw_cls.astype(np.float32) - cls_zero) * cls_scale
        boxes = (raw_box.astype(np.float32) - box_zero) * box_scale
        
        raw_boxes = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                cell_probs = probs[gy, gx, 1:11]
                best_cls = int(np.argmax(cell_probs))
                score = float(cell_probs[best_cls])
                
                if score >= conf_threshold:
                    p_dx = float(boxes[gy, gx, 0])
                    p_dy = float(boxes[gy, gx, 1])
                    p_w  = float(boxes[gy, gx, 2])
                    p_h  = float(boxes[gy, gx, 3])
                    
                    cx = (gx + p_dx) / float(grid_size)
                    cy = (gy + p_dy) / float(grid_size)
                    
                    # Optimal calibrated blend
                    alpha = CLASS_ALPHAS[best_cls]
                    pw = priors[str(best_cls)]["w"]
                    ph = priors[str(best_cls)]["h"]
                    w = (1.0 - alpha) * pw + alpha * p_w
                    h = (1.0 - alpha) * ph + alpha * p_h
                    
                    xmin = max(0.0, round((cx - w / 2.0) * ORIG_WIDTH, 1))
                    ymin = max(0.0, round((cy - h / 2.0) * ORIG_HEIGHT, 1))
                    xmax = min(ORIG_WIDTH, round((cx + w / 2.0) * ORIG_WIDTH, 1))
                    ymax = min(ORIG_HEIGHT, round((cy + h / 2.0) * ORIG_HEIGHT, 1))
                    
                    raw_boxes.append({
                        "class_id": best_cls,
                        "score": round(score, 4),
                        "bbox": [xmin, ymin, xmax, ymax]
                    })
                    
        # Per-class NMS
        final_boxes = []
        for c in range(10):
            c_boxes = [b for b in raw_boxes if b["class_id"] == c]
            if not c_boxes:
                continue
            coords = [b["bbox"] for b in c_boxes]
            scores = [b["score"] for b in c_boxes]
            keep = nms(coords, scores, iou_thresh=iou_thresh)
            for k in keep:
                final_boxes.append(c_boxes[k])
                
        final_boxes.sort(key=lambda b: b["score"], reverse=True)
        preds_json = json.dumps(final_boxes[:100])
        results.append(["DETECTION", sample_id, preds_json])
        
    return results

def build_full_submission(
    det_results,
    intent_preds_path="/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv",
    output_path="/home/bim/Projects/EAI-2026/warmup/submissions/submission_mobilenet_bbox_giou_128.csv"
):
    intent_rows = []
    if os.path.exists(intent_preds_path):
        with open(intent_preds_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row:
                    intent_rows.append(row)
        print(f"[✔] Loaded {len(intent_rows)} Intent predictions.")
    else:
        print(f"[!] Warning: Intent file not found at {intent_preds_path}")
        
    all_rows = intent_rows + det_results
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)
            
    print(f"[✔] Successfully created submission: {output_path} ({len(all_rows)} rows)")
    return output_path

if __name__ == "__main__":
    dets = infer_mobilenet_bbox_giou(conf_threshold=0.15, iou_thresh=0.35)
    build_full_submission(dets)
