import os
import glob
import json
import cv2
import numpy as np

ORIG_W = 1672.0
ORIG_H = 941.0

def load_ground_truth(val_split_file, labels_dir):
    """
    Loads ground truth bounding boxes for validation images.
    Returns: dict of img_id -> list of {'class_id': int, 'bbox': [xmin, ymin, xmax, ymax]}
    """
    with open(val_split_file, "r") as f:
        img_paths = [line.strip() for line in f if line.strip()]

    gt_dict = {}
    for img_p in img_paths:
        sample_id = os.path.splitext(os.path.basename(img_p))[0]
        label_file = os.path.join(labels_dir, f"{sample_id}.txt")
        boxes = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
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
        gt_dict[sample_id] = boxes
    return gt_dict

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

def calculate_map50(gt_dict, pred_dict, num_classes=10):
    """
    Calculates mAP@0.5 (PASCAL VOC 2010+ style continuous integration)
    gt_dict: sample_id -> list of {'class_id': int, 'bbox': [xmin, ymin, xmax, ymax]}
    pred_dict: sample_id -> list of {'class_id': int, 'score': float, 'bbox': [xmin, ymin, xmax, ymax]}
    """
    class_aps = {}

    for c in range(num_classes):
        # 1. Gather all GTs for class c
        total_gt = 0
        img_gt_boxes = {}
        for sample_id, gts in gt_dict.items():
            c_gts = [g["bbox"] for g in gts if g["class_id"] == c]
            img_gt_boxes[sample_id] = {
                "boxes": c_gts,
                "matched": [False] * len(c_gts)
            }
            total_gt += len(c_gts)

        if total_gt == 0:
            continue

        # 2. Gather all predictions for class c
        c_preds = []
        for sample_id, preds in pred_dict.items():
            for p in preds:
                if p["class_id"] == c:
                    c_preds.append({
                        "sample_id": sample_id,
                        "score": p["score"],
                        "bbox": p["bbox"]
                    })

        if len(c_preds) == 0:
            class_aps[c] = 0.0
            continue

        # Sort predictions by confidence score descending
        c_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(c_preds))
        fp = np.zeros(len(c_preds))

        for idx, pred in enumerate(c_preds):
            sid = pred["sample_id"]
            gt_info = img_gt_boxes.get(sid, {"boxes": [], "matched": []})
            boxes = gt_info["boxes"]
            matched = gt_info["matched"]

            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g_box in enumerate(boxes):
                iou = compute_iou(pred["bbox"], g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= 0.50:
                if not matched[best_gt_idx]:
                    tp[idx] = 1.0
                    matched[best_gt_idx] = True
                else:
                    fp[idx] = 1.0  # duplicate detection
            else:
                fp[idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / total_gt
        precisions = cum_tp / (cum_tp + cum_fp)

        # VOC 2010+ continuous area integration
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        i_indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_indices + 1] - mrec[i_indices]) * mpre[i_indices + 1])
        class_aps[c] = float(ap)

    if len(class_aps) == 0:
        return 0.0, {}
    mAP50 = float(np.mean(list(class_aps.values())))
    return mAP50, class_aps
