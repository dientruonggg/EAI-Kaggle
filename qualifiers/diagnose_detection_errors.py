import os
import sys
import glob
import json
import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.model_variants import build_micro_edge_ultra_v2, build_micro_edge_ultra_v3
from src.detection.infer import per_class_nms, NUM_CLASSES, ORIG_W, ORIG_H

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]

def compute_box_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def load_ground_truths(val_labels_dir):
    gt_by_img = {}
    label_files = glob.glob(os.path.join(val_labels_dir, "*.txt"))
    for lf in label_files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        boxes = []
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    xmin = max(0.0, (cx - bw / 2.0) * ORIG_W)
                    ymin = max(0.0, (cy - bh / 2.0) * ORIG_H)
                    xmax = min(ORIG_W, (cx + bw / 2.0) * ORIG_W)
                    ymax = min(ORIG_H, (cy + bh / 2.0) * ORIG_H)
                    boxes.append({"class_id": cid, "bbox": [xmin, ymin, xmax, ymax]})
        gt_by_img[stem] = boxes
    return gt_by_img

def run_diagnosis(model_path, model_type="v2", val_dir=None, conf_thresh=0.12, iou_thresh=0.45):
    if val_dir is None:
        val_dir = os.path.join(PROJECT_ROOT, "data/val")
    val_images_dir = os.path.join(val_dir, "images")
    val_labels_dir = os.path.join(val_dir, "labels")

    gt_by_img = load_ground_truths(val_labels_dir)
    image_paths = sorted(glob.glob(os.path.join(val_images_dir, "*.png")) +
                         glob.glob(os.path.join(val_images_dir, "*.jpg")))

    print(f"[*] Diagnosing {len(image_paths)} val images against {len(gt_by_img)} GT sets.")
    print(f"[*] Loading model: {model_path} (Type: {model_type})")

    if model_path.endswith(".tflite"):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()
        conf_det = next((d for d in output_details if "conf" in d["name"].lower() or d["shape"][-1] == 1), None)
        cls_det = next((d for d in output_details if "cls" in d["name"].lower() or d["shape"][-1] == 7), None)
        box_det = next((d for d in output_details if "box" in d["name"].lower() or d["shape"][-1] == 4), None)
        use_tflite = True
    else:
        if model_type == "v3":
            model = build_micro_edge_ultra_v3(num_classes=7, grid_size=20)
        else:
            model = build_micro_edge_ultra_v2(num_classes=7, grid_size=20)
        model.load_weights(model_path)
        use_tflite = False

    grid_size = 20

    # Diagnostics metrics counters
    total_predictions = 0
    total_gts = sum(len(gts) for gts in gt_by_img.values())

    tp_count = 0
    cls_err_count = 0
    loc_err_count = 0
    both_err_count = 0
    dup_err_count = 0
    bkg_err_count = 0
    missed_gt_count = 0

    per_class_tp = [0] * NUM_CLASSES
    per_class_cls_err = [0] * NUM_CLASSES
    per_class_loc_err = [0] * NUM_CLASSES
    per_class_missed = [0] * NUM_CLASSES
    per_class_gt = [0] * NUM_CLASSES
    per_class_ious = [[] for _ in range(NUM_CLASSES)]

    # Confusion matrix (7 classes + background as index 7)
    # confusion[gt_class][pred_class]
    confusion_matrix = np.zeros((8, 8), dtype=int)

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gts = gt_by_img.get(stem, [])
        for g in gts:
            per_class_gt[g["class_id"]] += 1

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (160, 160), interpolation=cv2.INTER_AREA)

        if use_tflite:
            if input_details["dtype"] == np.int8:
                scale, zp = input_details["quantization"]
                in_tensor = (resized.astype(np.float32) / (scale * 255.0) + zp).astype(np.int8)[None, ...]
            else:
                in_tensor = (resized.astype(np.float32) / 255.0)[None, ...]
            interpreter.set_tensor(input_details["index"], in_tensor)
            interpreter.invoke()

            def dequant(d):
                raw = interpreter.get_tensor(d["index"])[0]
                s, z = d.get("quantization", (0.0, 0))
                if d["dtype"] == np.int8 and s > 0:
                    return (raw.astype(np.float32) - z) * s
                return raw.astype(np.float32)

            pred_conf = dequant(conf_det)
            pred_cls = dequant(cls_det)
            pred_box = dequant(box_det)
        else:
            in_tensor = (resized.astype(np.float32) / 255.0)[None, ...]
            preds = model(in_tensor, training=False)
            pred_conf = preds["conf_output"].numpy()[0]
            pred_cls = preds["cls_output"].numpy()[0]
            pred_box = preds["box_output"].numpy()[0]

        img_boxes, img_scores, img_classes = [], [], []
        for gy in range(grid_size):
            for gx in range(grid_size):
                conf_val = float(pred_conf[gy, gx, 0])
                cls_probs = pred_cls[gy, gx]
                max_c = int(np.argmax(cls_probs))
                cls_p = float(cls_probs[max_c])
                final_score = conf_val * cls_p

                if final_score < conf_thresh:
                    continue

                off_x, off_y, bw, bh = pred_box[gy, gx]
                cx_norm = (gx + off_x) / float(grid_size)
                cy_norm = (gy + off_y) / float(grid_size)

                xmin = max(0.0, (cx_norm - bw / 2.0) * ORIG_W)
                ymin = max(0.0, (cy_norm - bh / 2.0) * ORIG_H)
                xmax = min(ORIG_W, (cx_norm + bw / 2.0) * ORIG_W)
                ymax = min(ORIG_H, (cy_norm + bh / 2.0) * ORIG_H)

                if (xmax - xmin) >= 4.0 and (ymax - ymin) >= 4.0:
                    img_boxes.append([xmin, ymin, xmax, ymax])
                    img_scores.append(final_score)
                    img_classes.append(max_c)

        # Apply per-class NMS
        kept_preds = per_class_nms(
            img_boxes, img_scores, img_classes,
            num_classes=NUM_CLASSES,
            iou_thresh=iou_thresh,
            max_boxes_per_image=100
        )

        total_predictions += len(kept_preds)

        # Match predictions to GTs
        gt_matched = [False] * len(gts)

        for pred in kept_preds:
            p_box = pred["bbox"]
            p_cid = pred["class_id"]

            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gts):
                iou = compute_box_iou(p_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_gt_idx >= 0 and best_iou >= 0.5:
                gt_cid = gts[best_gt_idx]["class_id"]
                if p_cid == gt_cid:
                    if not gt_matched[best_gt_idx]:
                        # True Positive
                        tp_count += 1
                        gt_matched[best_gt_idx] = True
                        per_class_tp[p_cid] += 1
                        per_class_ious[p_cid].append(best_iou)
                        confusion_matrix[gt_cid][p_cid] += 1
                    else:
                        # Duplicate Detection
                        dup_err_count += 1
                else:
                    # Classification Error (IoU >= 0.5 but wrong class)
                    cls_err_count += 1
                    per_class_cls_err[gt_cid] += 1
                    confusion_matrix[gt_cid][p_cid] += 1
            elif best_gt_idx >= 0 and 0.1 <= best_iou < 0.5:
                gt_cid = gts[best_gt_idx]["class_id"]
                if p_cid == gt_cid:
                    # Localization Error (Right class, bad coordinates)
                    loc_err_count += 1
                    per_class_loc_err[gt_cid] += 1
                else:
                    # Both Cls and Loc Error
                    both_err_count += 1
            else:
                # Background Error (IoU < 0.1 with any GT)
                bkg_err_count += 1
                confusion_matrix[7][p_cid] += 1  # 7 = background

        # Count missed GTs
        for g_idx, matched in enumerate(gt_matched):
            if not matched:
                missed_gt_count += 1
                per_class_missed[gts[g_idx]["class_id"]] += 1
                confusion_matrix[gts[g_idx]["class_id"]][7] += 1  # 7 = missed (pred as bg)

    # Print Report
    print("\n" + "=" * 70)
    print("        DETECTION ERROR BREAKDOWN & DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Total Ground Truth Objects: {total_gts}")
    print(f"Total Model Predictions:    {total_predictions}")
    print("-" * 70)
    print(f"  [1] True Positives (TP, IoU >= 0.5):      {tp_count:>5} ({tp_count / total_gts * 100:5.2f}% of GT)")
    print(f"  [2] Classification Error (IoU>=0.5, Wrong Cls):{cls_err_count:>5} ({cls_err_count / total_predictions * 100:5.2f}% of Preds)")
    print(f"  [3] Localization Error (IoU 0.1-0.5, Right Cls):{loc_err_count:>5} ({loc_err_count / total_predictions * 100:5.2f}% of Preds)")
    print(f"  [4] Both Cls & Loc Error (IoU 0.1-0.5, Wrong): {both_err_count:>5} ({both_err_count / total_predictions * 100:5.2f}% of Preds)")
    print(f"  [5] Duplicate Detections (NMS Leakage):    {dup_err_count:>5} ({dup_err_count / total_predictions * 100:5.2f}% of Preds)")
    print(f"  [6] Background False Positives (IoU < 0.1):{bkg_err_count:>5} ({bkg_err_count / total_predictions * 100:5.2f}% of Preds)")
    print(f"  [7] Missed Ground Truth (False Negatives): {missed_gt_count:>5} ({missed_gt_count / total_gts * 100:5.2f}% of GT)")
    print("=" * 70)

    print("\n[★ PER-CLASS DETAILED BREAKDOWN]")
    print(f"{'Class':<10} | {'GT':<5} | {'TP':<5} | {'Recall':<7} | {'Cls Err':<7} | {'Loc Err':<7} | {'Missed':<7} | {'Mean IoU':<8}")
    print("-" * 70)
    for c in range(NUM_CLASSES):
        gt_c = per_class_gt[c]
        tp_c = per_class_tp[c]
        rec = (tp_c / gt_c * 100) if gt_c > 0 else 0.0
        cls_e = per_class_cls_err[c]
        loc_e = per_class_loc_err[c]
        mis = per_class_missed[c]
        m_iou = np.mean(per_class_ious[c]) if per_class_ious[c] else 0.0
        print(f"{CLASS_NAMES[c]:<10} | {gt_c:<5} | {tp_c:<5} | {rec:6.2f}% | {cls_e:<7} | {loc_e:<7} | {mis:<7} | {m_iou:6.3f}")

    print("\n[★ CONFUSION MATRIX (Rows: Ground Truth, Columns: Predictions)]")
    cols_header = "          " + " ".join([f"{c[:5]:>6}" for c in CLASS_NAMES]) + "   BKG"
    print(cols_header)
    for r in range(NUM_CLASSES):
        row_str = f"{CLASS_NAMES[r]:<10}"
        for col in range(8):
            row_str += f"{confusion_matrix[r][col]:>6} "
        print(row_str)
    row_bg = f"{'BG Pred':<10}"
    for col in range(8):
        row_bg += f"{confusion_matrix[7][col]:>6} "
    print(row_bg)
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qualifiers/weights/champion_ultrav2_best.keras")
    parser.add_argument("--type", default="v2", choices=["v2", "v3"])
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--nms_iou", type=float, default=0.45)
    args = parser.parse_args()

    run_diagnosis(args.model, model_type=args.type, conf_thresh=args.conf, iou_thresh=args.nms_iou)
