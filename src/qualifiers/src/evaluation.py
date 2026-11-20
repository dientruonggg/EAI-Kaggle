import os
import json
import csv
import numpy as np

ORIG_W = 320.0
ORIG_H = 240.0
NUM_CLASSES = 7

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def calculate_map50(gt_dict, pred_dict, num_classes=NUM_CLASSES):
    """
    Calculates mAP@0.5 (PASCAL VOC 2010+ style continuous integration) for 7 classes.
    gt_dict: sample_id -> list of {'class_id': int, 'bbox': [xmin, ymin, xmax, ymax]}
    pred_dict: sample_id -> list of {'class_id': int, 'score': float, 'bbox': [xmin, ymin, xmax, ymax]}
    """
    class_aps = {}

    for c in range(num_classes):
        total_gt = 0
        img_gt_boxes = {}
        for sample_id, gts in gt_dict.items():
            c_gts = [g["bbox"] for g in gts if g["class_id"] == c]
            img_gt_boxes[sample_id] = {
                "boxes": c_gts,
                "detected": [False] * len(c_gts)
            }
            total_gt += len(c_gts)

        if total_gt == 0:
            continue

        c_preds = []
        for sample_id, preds in pred_dict.items():
            for p in preds:
                if p["class_id"] == c:
                    c_preds.append((p["score"], sample_id, p["bbox"]))

        c_preds.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(c_preds))
        fp = np.zeros(len(c_preds))

        for idx, (score, s_id, pbox) in enumerate(c_preds):
            if s_id not in img_gt_boxes or len(img_gt_boxes[s_id]["boxes"]) == 0:
                fp[idx] = 1
                continue

            gt_data = img_gt_boxes[s_id]
            best_iou = 0.0
            best_gt_idx = -1

            for gt_i, gbox in enumerate(gt_data["boxes"]):
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_i

            if best_iou >= 0.5:
                if not gt_data["detected"][best_gt_idx]:
                    tp[idx] = 1
                    gt_data["detected"][best_gt_idx] = True
                else:
                    fp[idx] = 1  # Duplicate detection
            else:
                fp[idx] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / total_gt
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        i_indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_indices + 1] - mrec[i_indices]) * mpre[i_indices + 1])
        class_aps[c] = float(ap)

    if not class_aps:
        return 0.0, {}
    mean_ap = float(np.mean(list(class_aps.values())))
    return mean_ap, class_aps

def evaluate_submission(sub_csv, gt_detection_dir, gt_intent_csv):
    """
    Computes overall score S = 0.70 * (mAP50 * 100) + 0.30 * (Accuracy * 100).
    """
    print(f"[*] Evaluating submission: {sub_csv}")
    det_preds = {}
    intent_preds = {}

    with open(sub_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            task, sid, pred = row[0], row[1], row[2]
            if task == "DETECTION":
                try:
                    det_preds[sid] = json.loads(pred)
                except Exception:
                    det_preds[sid] = []
            elif task == "INTENT":
                intent_preds[sid] = pred.strip()

    # Calculate Detection mAP50
    gt_det = {}
    label_files = [f for f in os.listdir(gt_detection_dir) if f.endswith(".txt")]
    for lf in label_files:
        sid = os.path.splitext(lf)[0]
        boxes = []
        with open(os.path.join(gt_detection_dir, lf), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    xmin = (cx - w / 2.0) * ORIG_W
                    ymin = (cy - h / 2.0) * ORIG_H
                    xmax = (cx + w / 2.0) * ORIG_W
                    ymax = (cy + h / 2.0) * ORIG_H
                    boxes.append({"class_id": cid, "bbox": [xmin, ymin, xmax, ymax]})
        gt_det[sid] = boxes

    map50, class_aps = calculate_map50(gt_det, det_preds, num_classes=NUM_CLASSES)

    # Calculate Intent Accuracy
    gt_intent = {}
    with open(gt_intent_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_intent[r["question_id"]] = r["intent"].strip()

    correct_intent = 0
    total_intent = len(gt_intent)
    for qid, gt_lbl in gt_intent.items():
        if intent_preds.get(qid) == gt_lbl:
            correct_intent += 1
    accuracy = (correct_intent / total_intent) if total_intent > 0 else 0.0

    score = 0.70 * (map50 * 100.0) + 0.30 * (accuracy * 100.0)

    print("\n" + "=" * 60)
    print("           COMPETITION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Detection mAP@0.5:   {map50 * 100.0:.2f}% (Weight: 70%)")
    print(f"Intent Accuracy:     {accuracy * 100.0:.2f}% (Weight: 30%)")
    print(f"Overall Score S:     {score:.2f} / 100.00")
    print("=" * 60)

    return score, map50, accuracy
