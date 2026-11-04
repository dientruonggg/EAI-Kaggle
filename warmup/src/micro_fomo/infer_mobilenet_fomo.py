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

def infer_mobilenet_fomo(
    tflite_path="/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_fomo/mobilenet_fomo_int8.tflite",
    test_images_dir="/home/bim/Projects/EAI-2026/warmup/data/test/images",
    priors_path="/home/bim/Projects/EAI-2026/warmup/data/class_priors.json",
    conf_threshold=0.15,
    iou_thresh=0.50
):
    print(f"[*] Initializing TFLite Interpreter from: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    in_shape = input_details["shape"]
    in_dtype = input_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero = output_details.get("quantization", (0.0, 0))
    grid_size = output_details["shape"][1]
    
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
        
        output_data = interpreter.get_tensor(output_details["index"])[0]
        probs = (output_data.astype(np.float32) - out_zero) * out_scale
        
        raw_boxes = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                cell_probs = probs[gy, gx, 1:11]
                best_cls = int(np.argmax(cell_probs))
                score = float(cell_probs[best_cls])
                
                if score >= conf_threshold:
                    cx = (gx + 0.5) / float(grid_size)
                    cy = (gy + 0.5) / float(grid_size)
                    cls_prior = priors.get(str(best_cls), {"w": 0.1, "h": 0.1})
                    pw, ph = cls_prior["w"], cls_prior["h"]
                    
                    xmin = max(0.0, round((cx - pw / 2.0) * ORIG_WIDTH, 1))
                    ymin = max(0.0, round((cy - ph / 2.0) * ORIG_HEIGHT, 1))
                    xmax = min(ORIG_WIDTH, round((cx + pw / 2.0) * ORIG_WIDTH, 1))
                    ymax = min(ORIG_HEIGHT, round((cy + ph / 2.0) * ORIG_HEIGHT, 1))
                    
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
        final_boxes = final_boxes[:100]
        
        results.append(["DETECTION", sample_id, json.dumps(final_boxes, separators=(',', ':'))])
        
    return results

def save_and_build(out_detection_csv, final_submission_csv, conf=0.15):
    results = infer_mobilenet_fomo(conf_threshold=conf)
    os.makedirs(os.path.dirname(out_detection_csv), exist_ok=True)
    with open(out_detection_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerows(results)
    print(f"[✔] Detection predictions saved to: {out_detection_csv}")
    
    # Read intent predictions
    intent_csv = "/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv"
    intent_rows = []
    with open(intent_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if r:
                intent_rows.append(r)
                
    all_rows = results + intent_rows
    with open(final_submission_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerows(all_rows)
        
    print(f"[✔] Combined submission saved to: {final_submission_csv}")
    
    # Validate
    from submit import validate_submission
    validate_submission(final_submission_csv)

if __name__ == "__main__":
    det_out = "/home/bim/Projects/EAI-2026/warmup/submissions/mobilenet_fomo_preds.csv"
    sub_out = "/home/bim/Projects/EAI-2026/warmup/submissions/submission_mobilenet_fomo.csv"
    save_and_build(det_out, sub_out, conf=0.15)
