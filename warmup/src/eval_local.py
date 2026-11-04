import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import json
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from src.evaluation import load_ground_truth, calculate_map50

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W = 1672.0
ORIG_H = 941.0

def eval_yolo_model(weights_path, img_size=128, conf_thresh=0.20):
    print(f"\n==========================================")
    print(f"[*] Evaluating YOLO Model: {weights_path} (img_size={img_size}, conf={conf_thresh})")
    model = YOLO(weights_path)
    
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
        
    pred_dict = {}
    for img_p in val_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        preds = model.predict(source=img_p, imgsz=img_size, conf=conf_thresh, verbose=False)
        box_list = []
        if preds and len(preds) > 0:
            boxes = preds[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                for i in range(len(boxes)):
                    box_list.append({
                        "class_id": int(clss[i]),
                        "score": float(confs[i]),
                        "bbox": [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])]
                    })
        pred_dict[sid] = box_list
        
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)
    mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
    
    print(f"[✔] Local Validation mAP@0.5: {mAP50 * 100:.2f}%")
    for c, ap in sorted(class_aps.items()):
        print(f"    Class {c}: AP50 = {ap * 100:.2f}%")
    return mAP50, class_aps

def eval_micro_fomo(tflite_path, priors_path="/home/bim/Projects/EAI-2026/warmup/data/class_priors.json", conf_thresh=0.30):
    print(f"\n==========================================")
    print(f"[*] Evaluating Micro-FOMO TFLite: {tflite_path} (conf={conf_thresh})")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    in_shape = input_details["shape"]
    in_dtype = input_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero = output_details.get("quantization", (0.0, 0))
    
    with open(priors_path, "r") as f:
        priors = json.load(f)
        
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
        
    grid_size = output_details["shape"][1]
    pred_dict = {}
    
    for img_p in val_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (in_shape[1], in_shape[2]), interpolation=cv2.INTER_AREA)
        
        if in_dtype == np.int8:
            norm = (img_resized.astype(np.float32) / 255.0)
            in_tensor = np.round(norm / in_scale + in_zero).astype(np.int8)
        else:
            in_tensor = (img_resized.astype(np.float32) / 255.0).astype(np.float32)
            
        in_tensor = np.expand_dims(in_tensor, axis=0)
        interpreter.set_tensor(input_details["index"], in_tensor)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details["index"])[0]
        if output_details["dtype"] == np.int8:
            probs = (output_data.astype(np.float32) - out_zero) * out_scale
        else:
            probs = output_data.astype(np.float32)
            
        box_list = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                cell_probs = probs[gy, gx, 1:]
                best_cls = int(np.argmax(cell_probs))
                score = float(cell_probs[best_cls])
                
                if score >= conf_thresh:
                    cx = (gx + 0.5) / float(grid_size)
                    cy = (gy + 0.5) / float(grid_size)
                    cls_prior = priors.get(str(best_cls), {"w": 0.1, "h": 0.1})
                    pw = cls_prior["w"]
                    ph = cls_prior["h"]
                    
                    xmin = max(0.0, (cx - pw / 2.0) * ORIG_W)
                    ymin = max(0.0, (cy - ph / 2.0) * ORIG_H)
                    xmax = min(ORIG_W, (cx + pw / 2.0) * ORIG_W)
                    ymax = min(ORIG_H, (cy + ph / 2.0) * ORIG_H)
                    
                    box_list.append({
                        "class_id": best_cls,
                        "score": score,
                        "bbox": [xmin, ymin, xmax, ymax]
                    })
        box_list.sort(key=lambda b: b["score"], reverse=True)
        pred_dict[sid] = box_list[:100]
        
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)
    mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
    print(f"[✔] Local Validation mAP@0.5: {mAP50 * 100:.2f}%")
    for c, ap in sorted(class_aps.items()):
        print(f"    Class {c}: AP50 = {ap * 100:.2f}%")
    return mAP50, class_aps

if __name__ == "__main__":
    print("=== LOCAL BENCHMARK VALIDATION ===")
    
    # 0. FOMO-BBox INT8 (128x128 with Dynamic Box Regression)
    fomo_bbox_tflite = "/home/bim/Projects/EAI-2026/warmup/weights/micro_fomo/fomo_bbox_int8.tflite"
    if os.path.exists(fomo_bbox_tflite):
        from src.micro_fomo.infer_bbox import eval_fomo_bbox_on_val
        for conf in [0.15, 0.20, 0.25, 0.30]:
            eval_fomo_bbox_on_val(fomo_bbox_tflite, VAL_SPLIT_FILE, LABELS_DIR, conf_thresh=conf)

    # 1. Baseline YOLOv8n
    yolov8n_weights = "/home/bim/Projects/EAI-2026/warmup/runs/detect/yolov8n_warmup/weights/best.pt"
    if os.path.exists(yolov8n_weights):
        eval_yolo_model(yolov8n_weights, img_size=640, conf_thresh=0.25)
        
    # 2. Micro-YOLO 100ep
    micro_yolo_weights = "/home/bim/Projects/EAI-2026/warmup/runs/detect/micro_yolo_100ep/weights/best.pt"
    if os.path.exists(micro_yolo_weights):
        eval_yolo_model(micro_yolo_weights, img_size=128, conf_thresh=0.20)
        
    # 3. Micro-FOMO INT8 (96x96 with static priors)
    fomo_tflite = "/home/bim/Projects/EAI-2026/warmup/weights/micro_fomo/micro_fomo_int8.tflite"
    if os.path.exists(fomo_tflite):
        eval_micro_fomo(fomo_tflite, conf_thresh=0.30)
