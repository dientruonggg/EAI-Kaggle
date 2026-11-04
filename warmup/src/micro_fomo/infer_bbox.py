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

ORIG_W = 1672.0
ORIG_H = 941.0
GRID_SIZE = 16

def nms(boxes, scores, iou_thresh=0.45):
    """Simple standard NMS in pure numpy"""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        union = np.maximum(union, 1e-6)
        ovr = inter / union

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

def get_output_tensors(interpreter):
    out_details = interpreter.get_output_details()
    cls_det = None
    box_det = None
    for d in out_details:
        if d["shape"][-1] == 11:
            cls_det = d
        elif d["shape"][-1] == 4:
            box_det = d
    return cls_det, box_det

def infer_fomo_bbox_image(interpreter, img_path, conf_thresh=0.25):
    input_details = interpreter.get_input_details()[0]
    cls_det, box_det = get_output_tensors(interpreter)

    in_shape = input_details["shape"]
    in_dtype = input_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))

    img = cv2.imread(img_path)
    if img is None:
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (in_shape[1], in_shape[2]), interpolation=cv2.INTER_AREA)

    if in_dtype == np.int8:
        norm = img_resized.astype(np.float32) / 255.0
        in_tensor = np.round(norm / in_scale + in_zero).astype(np.int8)
    else:
        in_tensor = (img_resized.astype(np.float32) / 255.0).astype(np.float32)

    in_tensor = np.expand_dims(in_tensor, axis=0)
    interpreter.set_tensor(input_details["index"], in_tensor)
    interpreter.invoke()

    # Dequantize classification output (already Softmax probabilities from model)
    c_raw = interpreter.get_tensor(cls_det["index"])[0]
    c_scale, c_zero = cls_det.get("quantization", (0.0, 0))
    if cls_det["dtype"] == np.int8:
        probs = (c_raw.astype(np.float32) - c_zero) * c_scale
    else:
        probs = c_raw.astype(np.float32)

    # Dequantize box output (already Sigmoid from model)
    b_raw = interpreter.get_tensor(box_det["index"])[0]
    b_scale, b_zero = box_det.get("quantization", (0.0, 0))
    if box_det["dtype"] == np.int8:
        box_preds = (b_raw.astype(np.float32) - b_zero) * b_scale
    else:
        box_preds = b_raw.astype(np.float32)

    c_boxes = []
    c_scores = []
    c_classes = []

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            cell_probs = probs[gy, gx, 1:]  # 10 object classes
            best_cls = int(np.argmax(cell_probs))
            score = float(cell_probs[best_cls])

            if score >= conf_thresh:
                dx, dy, bw, bh = box_preds[gy, gx]
                cx = (gx + dx) / float(GRID_SIZE)
                cy = (gy + dy) / float(GRID_SIZE)

                # Ensure minimum physical box dimensions (at least 8px)
                bw = max(bw, 8.0 / ORIG_W)
                bh = max(bh, 8.0 / ORIG_H)

                xmin = max(0.0, (cx - bw / 2.0) * ORIG_W)
                ymin = max(0.0, (cy - bh / 2.0) * ORIG_H)
                xmax = min(ORIG_W, (cx + bw / 2.0) * ORIG_W)
                ymax = min(ORIG_H, (cy + bh / 2.0) * ORIG_H)

                c_boxes.append([xmin, ymin, xmax, ymax])
                c_scores.append(score)
                c_classes.append(best_cls)

    if len(c_boxes) == 0:
        return []

    # Run per-class NMS
    final_detections = []
    for cls_id in range(10):
        cls_indices = [i for i, c in enumerate(c_classes) if c == cls_id]
        if not cls_indices:
            continue
        sub_boxes = [c_boxes[i] for i in cls_indices]
        sub_scores = [c_scores[i] for i in cls_indices]
        keep = nms(sub_boxes, sub_scores, iou_thresh=0.45)
        for k in keep:
            idx = cls_indices[k]
            final_detections.append({
                "class_id": cls_id,
                "score": round(c_scores[idx], 4),
                "bbox": [round(coord, 1) for coord in c_boxes[idx]]
            })

    final_detections.sort(key=lambda b: b["score"], reverse=True)
    return final_detections[:100]

def eval_fomo_bbox_on_val(tflite_path, val_split_file, labels_dir, conf_thresh=0.25):
    from src.evaluation import load_ground_truth, calculate_map50
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    with open(val_split_file, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]

    pred_dict = {}
    for img_p in val_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        preds = infer_fomo_bbox_image(interpreter, img_p, conf_thresh=conf_thresh)
        pred_dict[sid] = preds

    gt_dict = load_ground_truth(val_split_file, labels_dir)
    mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
    print(f"\n[✔] FOMO-BBox Local Validation mAP@0.5: {mAP50 * 100:.2f}% (conf={conf_thresh})")
    for c, ap in sorted(class_aps.items()):
        print(f"    Class {c}: AP50 = {ap * 100:.2f}%")
    return mAP50, class_aps
