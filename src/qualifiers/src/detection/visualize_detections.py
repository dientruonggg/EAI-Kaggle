#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - SCORE & NMS BOUNDING BOX INSPECTOR & VISUALIZER
Inspects how Score = Conf * Cls_Prob and NMS filter bounding boxes against Ground Truth.
"""
import os
import sys
import glob
import json
import cv2
import numpy as np

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.infer import infer_test_images, nms, ORIG_W, ORIG_H
from src.evaluation import compute_iou

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
CLASS_VI = ["Chai", "Cốc", "Điện thoại", "Sách", "Chuột máy tính", "Hộp", "Ô tô đồ chơi"]

def load_ground_truth(labels_dir):
    gt_dict = {}
    for lf in glob.glob(os.path.join(labels_dir, "*.txt")):
        sid = os.path.splitext(os.path.basename(lf))[0]
        boxes = []
        with open(lf, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    cid = int(p[0])
                    cx, cy, w, h = map(float, p[1:5])
                    xmin = max(0.0, (cx - w / 2.0) * ORIG_W)
                    ymin = max(0.0, (cy - h / 2.0) * ORIG_H)
                    xmax = min(ORIG_W, (cx + w / 2.0) * ORIG_W)
                    ymax = min(ORIG_H, (cy + h / 2.0) * ORIG_H)
                    boxes.append({"class_id": cid, "bbox": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]})
        gt_dict[sid] = boxes
    return gt_dict

def inspect_score_and_nms(
    model_path=os.path.join(PROJECT_ROOT, "weights/mobilenet_bbox_160_int8.tflite"),
    images_dir=os.path.join(PROJECT_ROOT, "data/val/images"),
    labels_dir=os.path.join(PROJECT_ROOT, "data/val/labels"),
    conf_thresh=0.25,
    iou_thresh=0.45,
    max_images=10
):
    print("=" * 85)
    print("       🔍 EDGE AI CHALLENGE 2026 - SCORE & NMS BOUNDING BOX INSPECTION")
    print(f"       Model: {os.path.basename(model_path)} | Conf Thresh: {conf_thresh} | NMS IoU Thresh: {iou_thresh}")
    print("=" * 85)

    gt_dict = load_ground_truth(labels_dir)
    preds = infer_test_images(model_path, images_dir, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    pred_dict = {p["sample_id"]: json.loads(p["prediction"]) for p in preds}

    total_tp = 0
    total_fp = 0
    total_gt = 0

    image_ids = sorted(list(gt_dict.keys()))[:max_images]

    for sid in image_ids:
        gts = gt_dict.get(sid, [])
        det_boxes = pred_dict.get(sid, [])
        total_gt += len(gts)

        print("\n" + "-" * 85)
        print(f" 📸 ẢNH: {sid} | Số vật thể thật: {len(gts)} | Bounding box phát hiện sau NMS: {len(det_boxes)}")
        print("-" * 85)

        # Ground Truth
        print("  🎯 [GROUND TRUTH]:")
        if not gts:
            print("     (Không có vật thể - Cảnh rỗng)")
        for g_i, g in enumerate(gts):
            c_name = CLASS_NAMES[g["class_id"]]
            c_vi = CLASS_VI[g["class_id"]]
            print(f"     GT #{g_i+1}: Lớp {g['class_id']} ({c_name} - {c_vi}) | BBox: {g['bbox']}")

        # Predictions
        print("  📦 [PREDICTED BBOXES SAU SCORE & NMS]:")
        if not det_boxes:
            print("     (Không phát hiện box nào vượt ngưỡng)")
        
        gt_matched = [False] * len(gts)

        for p_i, p in enumerate(det_boxes):
            cid = p["class_id"]
            c_name = CLASS_NAMES[cid]
            c_vi = CLASS_VI[cid]
            score = p["score"]
            pbox = p["bbox"]

            # Match against GT
            best_iou = 0.0
            best_gt_idx = -1
            for g_i, g in enumerate(gts):
                if g["class_id"] == cid:
                    iou = compute_iou(pbox, g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_i

            is_tp = False
            if best_iou >= 0.50 and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                is_tp = True
                gt_matched[best_gt_idx] = True
                total_tp += 1
                status = f"✅ TRUE POSITIVE (IoU={best_iou:.3f} >= 0.50 với GT #{best_gt_idx+1})"
            elif best_iou > 0.0:
                total_fp += 1
                status = f"⚠️ FALSE POSITIVE / Low IoU (IoU={best_iou:.3f} < 0.50)"
            else:
                total_fp += 1
                status = "❌ FALSE POSITIVE (Box thừa / Sai lớp)"

            print(f"     Pred #{p_i+1}: Lớp {cid} ({c_name} - {c_vi:<10}) | Score: {score:.4f} | BBox: {pbox}")
            print(f"             └──> {status}")

    print("\n" + "=" * 85)
    print(f" 📊 TỔNG KẾT TRÊN {len(image_ids)} ẢNH KIỂM THỬ ĐẦU TIÊN:")
    print(f"    - Tổng BBox thật (Ground Truth): {total_gt}")
    print(f"    - True Positives (Khớp chuẩn IoU >= 0.50): {total_tp}")
    print(f"    - False Positives (Box lệch / thừa): {total_fp}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(PROJECT_ROOT, "weights/mobilenet_bbox_160_int8.tflite"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max_images", type=int, default=8)
    args = parser.parse_args()

    inspect_score_and_nms(model_path=args.model, conf_thresh=args.conf, iou_thresh=args.iou, max_images=args.max_images)
