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

M128_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
M160_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
TEST_IMAGES_DIR = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
INTENT_PREDS_PATH = "/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv"
OUTPUT_PATH = "/home/bim/Projects/EAI-2026/warmup/submissions/submission_ensemble_128_160.csv"

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

def run_ensemble(conf_thresh=0.10, iou_thresh=0.35):
    print(f"[*] Initializing Ensemble (128x128 Full + 160x160 Best)...")
    interp128, in128, c128, b128 = get_interpreter(M128_PATH)
    interp160, in160, c160, b160 = get_interpreter(M160_PATH)
    
    with open(PRIORS_PATH) as f:
        priors = json.load(f)
        
    test_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    print(f"[*] Processing {len(test_images)} test images with Ensemble...")
    
    results = []
    
    for img_p in test_images:
        sample_id = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        if img is None:
            results.append(["DETECTION", sample_id, "[]"])
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_boxes = []
        
        # 1. Infer 128x128
        in_sc128, in_zp128 = in128.get("quantization", (0.0, 0))
        c_sc128, c_zp128 = c128.get("quantization", (0.0, 0))
        b_sc128, b_zp128 = b128.get("quantization", (0.0, 0))
        
        r128 = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_AREA)
        norm128 = r128.astype(np.float32) / 255.0
        in_t128 = np.round(norm128 / in_sc128 + in_zp128).astype(np.int8)[None, ...]
        interp128.set_tensor(in128["index"], in_t128)
        interp128.invoke()
        
        probs128 = (interp128.get_tensor(c128["index"])[0].astype(np.float32) - c_zp128) * c_sc128
        boxes128 = (interp128.get_tensor(b128["index"])[0].astype(np.float32) - b_zp128) * b_sc128
        
        for gy in range(16):
            for gx in range(16):
                cp = probs128[gy, gx, 1:11]
                bc = int(np.argmax(cp))
                sc = float(cp[bc])
                if sc >= conf_thresh:
                    p_dx, p_dy, p_w, p_h = float(boxes128[gy, gx, 0]), float(boxes128[gy, gx, 1]), float(boxes128[gy, gx, 2]), float(boxes128[gy, gx, 3])
                    cx = (gx + p_dx) / 16.0
                    cy = (gy + p_dy) / 16.0
                    a = ALPHAS_128[bc]
                    pw, ph = priors[str(bc)]["w"], priors[str(bc)]["h"]
                    w = (1.0 - a) * pw + a * p_w
                    h = (1.0 - a) * ph + a * p_h
                    xmin = max(0.0, (cx - w / 2.0) * ORIG_WIDTH)
                    ymin = max(0.0, (cy - h / 2.0) * ORIG_HEIGHT)
                    xmax = min(ORIG_WIDTH, (cx + w / 2.0) * ORIG_WIDTH)
                    ymax = min(ORIG_HEIGHT, (cy + h / 2.0) * ORIG_HEIGHT)
                    raw_boxes.append({
                        "class_id": bc,
                        "score": round(sc, 4),
                        "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]
                    })
                    
        # 2. Infer 160x160
        in_sc160, in_zp160 = in160.get("quantization", (0.0, 0))
        c_sc160, c_zp160 = c160.get("quantization", (0.0, 0))
        b_sc160, b_zp160 = b160.get("quantization", (0.0, 0))
        
        r160 = cv2.resize(img_rgb, (160, 160), interpolation=cv2.INTER_AREA)
        norm160 = r160.astype(np.float32) / 255.0
        in_t160 = np.round(norm160 / in_sc160 + in_zp160).astype(np.int8)[None, ...]
        interp160.set_tensor(in160["index"], in_t160)
        interp160.invoke()
        
        probs160 = (interp160.get_tensor(c160["index"])[0].astype(np.float32) - c_zp160) * c_sc160
        boxes160 = (interp160.get_tensor(b160["index"])[0].astype(np.float32) - b_zp160) * b_sc160
        
        for gy in range(20):
            for gx in range(20):
                cp = probs160[gy, gx, 1:11]
                bc = int(np.argmax(cp))
                sc = float(cp[bc])
                if sc >= conf_thresh:
                    p_dx, p_dy, p_w, p_h = float(boxes160[gy, gx, 0]), float(boxes160[gy, gx, 1]), float(boxes160[gy, gx, 2]), float(boxes160[gy, gx, 3])
                    cx = (gx + p_dx) / 20.0
                    cy = (gy + p_dy) / 20.0
                    a = ALPHAS_160[bc]
                    pw, ph = priors[str(bc)]["w"], priors[str(bc)]["h"]
                    w = (1.0 - a) * pw + a * p_w
                    h = (1.0 - a) * ph + a * p_h
                    xmin = max(0.0, (cx - w / 2.0) * ORIG_WIDTH)
                    ymin = max(0.0, (cy - h / 2.0) * ORIG_HEIGHT)
                    xmax = min(ORIG_WIDTH, (cx + w / 2.0) * ORIG_WIDTH)
                    ymax = min(ORIG_HEIGHT, (cy + h / 2.0) * ORIG_HEIGHT)
                    raw_boxes.append({
                        "class_id": bc,
                        "score": round(sc, 4),
                        "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]
                    })
                    
        # 3. Class-specific NMS across combined candidate boxes
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

def build_full_submission(det_results, output_path=OUTPUT_PATH):
    intent_rows = []
    if os.path.exists(INTENT_PREDS_PATH):
        with open(INTENT_PREDS_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row:
                    intent_rows.append(row)
        print(f"[✔] Loaded {len(intent_rows)} Intent predictions.")
    else:
        print(f"[!] Warning: Intent file not found at {INTENT_PREDS_PATH}")
        
    all_rows = intent_rows + det_results
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)
            
    print(f"[✔] Successfully created submission: {output_path} ({len(all_rows)} rows)")
    return output_path

if __name__ == "__main__":
    dets = run_ensemble(conf_thresh=0.10, iou_thresh=0.35)
    build_full_submission(dets)
