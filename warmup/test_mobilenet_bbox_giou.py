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
TFLITE_PATH = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_giou_128/mobilenet_bbox_giou_128_int8.tflite"

def evaluate_bbox_giou(tflite_path=TFLITE_PATH):
    if not os.path.exists(tflite_path):
        print(f"File not found: {tflite_path}")
        return

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
            
    print(f"[*] Found cls_det: {cls_det['name']} ({cls_det['shape']}), box_det: {box_det['name']} ({box_det['shape']})")

    in_shape = input_details["shape"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    cls_scale, cls_zero = cls_det.get("quantization", (0.0, 0))
    box_scale, box_zero = box_det.get("quantization", (0.0, 0))

    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    grid_size = cls_det["shape"][1] # 16

    cached_outputs = {}
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
        
        raw_cls = interpreter.get_tensor(cls_det["index"])[0]
        raw_box = interpreter.get_tensor(box_det["index"])[0]
        
        probs = (raw_cls.astype(np.float32) - cls_zero) * cls_scale
        boxes = (raw_box.astype(np.float32) - box_zero) * box_scale
        
        cached_outputs[sid] = (probs, boxes)

    print("\n=== EVALUATING DUAL-HEAD DYNAMIC GIOU BBOX ON VAL SET ===")
    best_mAP = 0.0
    best_cfg = None

    for conf in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        for iou in [0.35, 0.40, 0.45, 0.50]:
            pred_dict = {}
            for sid, (probs, boxes) in cached_outputs.items():
                raw_boxes = []
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        cell_probs = probs[gy, gx, 1:11]
                        best_cls = int(np.argmax(cell_probs))
                        score = float(cell_probs[best_cls])
                        if score >= conf:
                            pred_b = boxes[gy, gx] # [dx, dy, w, h] in [0, 1]
                            p_dx, p_dy, p_w, p_h = float(pred_b[0]), float(pred_b[1]), float(pred_b[2]), float(pred_b[3])
                            
                            cx = (gx + p_dx) / float(grid_size)
                            cy = (gy + p_dy) / float(grid_size)
                            w = max(0.01, min(1.0, p_w))
                            h = max(0.01, min(1.0, p_h))
                            
                            xmin = max(0.0, (cx - w / 2.0) * ORIG_W)
                            ymin = max(0.0, (cy - h / 2.0) * ORIG_H)
                            xmax = min(ORIG_W, (cx + w / 2.0) * ORIG_W)
                            ymax = min(ORIG_H, (cy + h / 2.0) * ORIG_H)
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
        print(f"\n[★ BEST DUAL-HEAD GIOU CONFIG] conf={conf:.2f}, iou={iou:.2f} with mAP@0.5 = {best_mAP * 100:.2f}%")
        for c in range(10):
            print(f"  Class {c} ({CLASS_NAMES[c]}): AP50 = {aps.get(c, 0.0) * 100:.2f}%")

if __name__ == "__main__":
    evaluate_bbox_giou()
