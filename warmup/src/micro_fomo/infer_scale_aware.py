import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import glob
import cv2
import numpy as np
import tensorflow as tf
from src.evaluation import load_ground_truth, calculate_map50

ORIG_W = 1672.0
ORIG_H = 941.0
GRID_SIZE = 16

def infer_scale_aware_image(interpreter, img_path, priors, conf_thresh=0.25):
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_shape = in_det["shape"]
    in_dtype = in_det["dtype"]
    in_scale, in_zero = in_det.get("quantization", (0.0, 0))
    out_scale, out_zero = out_det.get("quantization", (0.0, 0))

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

    interpreter.set_tensor(in_det["index"], np.expand_dims(in_tensor, axis=0))
    interpreter.invoke()

    out_raw = interpreter.get_tensor(out_det["index"])[0]  # (16, 16, 21)
    if out_det["dtype"] == np.int8:
        probs = (out_raw.astype(np.float32) - out_zero) * out_scale
    else:
        probs = out_raw.astype(np.float32)

    box_list = []
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            cell_probs = probs[gy, gx, 1:]  # categories 1..20
            best_cat = int(np.argmax(cell_probs)) + 1  # 1..20
            score = float(probs[gy, gx, best_cat])

            if score >= conf_thresh:
                if best_cat <= 10:
                    cls_id = best_cat - 1
                    pw = priors[str(cls_id)]["small"]["w"]
                    ph = priors[str(cls_id)]["small"]["h"]
                else:
                    cls_id = best_cat - 11
                    pw = priors[str(cls_id)]["large"]["w"]
                    ph = priors[str(cls_id)]["large"]["h"]

                cx = (gx + 0.5) / float(GRID_SIZE)
                cy = (gy + 0.5) / float(GRID_SIZE)

                xmin = max(0.0, (cx - pw / 2.0) * ORIG_W)
                ymin = max(0.0, (cy - ph / 2.0) * ORIG_H)
                xmax = min(ORIG_W, (cx + pw / 2.0) * ORIG_W)
                ymax = min(ORIG_H, (cy + ph / 2.0) * ORIG_H)

                box_list.append({
                    "class_id": cls_id,
                    "score": round(score, 4),
                    "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]
                })

    box_list.sort(key=lambda b: b["score"], reverse=True)
    return box_list[:100]

def eval_scale_aware_val(tflite_path, val_split_file, labels_dir, priors_path, conf_thresh=0.25):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    with open(priors_path, "r") as f:
        priors = json.load(f)

    with open(val_split_file, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]

    pred_dict = {}
    for p in val_imgs:
        sid = os.path.splitext(os.path.basename(p))[0]
        preds = infer_scale_aware_image(interpreter, p, priors, conf_thresh=conf_thresh)
        pred_dict[sid] = preds

    gt_dict = load_ground_truth(val_split_file, labels_dir)
    mAP50, class_aps = calculate_map50(gt_dict, pred_dict)
    print(f"\n[✔] Scale-Aware FOMO Local Val mAP@0.5: {mAP50 * 100:.2f}% (conf={conf_thresh})")
    for c, ap in sorted(class_aps.items()):
        print(f"    Class {c}: AP50 = {ap * 100:.2f}%")
    return mAP50, class_aps
