import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
from src.evaluation import load_ground_truth, calculate_map50, compute_iou
from tune_ensemble import cache_preds

VAL_SPLIT_FILE = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
LABELS_DIR = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
ORIG_W, ORIG_H = 1672.0, 941.0
PRIORS_PATH = "/home/bim/Projects/EAI-2026/warmup/data/class_priors.json"
CLASS_NAMES = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"]

OPT_ALPHAS = [0.5, 0.0, 0.6, 1.0, 0.3, 0.6, 0.4, 0.1, 0.0, 0.2]

def wbf_consensus(c_boxes, iou_thresh=0.28, boost_factor=0.03, min_cluster_size=1, min_single_score=0.25):
    sorted_boxes = sorted(c_boxes, key=lambda b: b["score"], reverse=True)
    clusters = []

    for b in sorted_boxes:
        matched = False
        for cluster in clusters:
            leader_box = cluster[0]["bbox"]
            if compute_iou(b["bbox"], leader_box) >= iou_thresh:
                cluster.append(b)
                matched = True
                break
        if not matched:
            clusters.append([b])

    fused_boxes = []
    for cluster in clusters:
        # Consensus Filtering:
        # If cluster has only 1 model prediction, require higher confidence
        if len(cluster) < min_cluster_size and cluster[0]["score"] < min_single_score:
            continue

        total_w = sum(b["score"] for b in cluster)
        avg_x1 = sum(b["score"] * b["bbox"][0] for b in cluster) / total_w
        avg_y1 = sum(b["score"] * b["bbox"][1] for b in cluster) / total_w
        avg_x2 = sum(b["score"] * b["bbox"][2] for b in cluster) / total_w
        avg_y2 = sum(b["score"] * b["bbox"][3] for b in cluster) / total_w

        max_score = cluster[0]["score"]
        boosted_score = min(1.0, max_score * (1.0 + boost_factor * (len(cluster) - 1)))

        cid = cluster[0]["class_id"]
        fused_boxes.append({
            "class_id": cid,
            "score": round(boosted_score, 4),
            "bbox": [round(avg_x1, 1), round(avg_y1, 1), round(avg_x2, 1), round(avg_y2, 1)]
        })

    return fused_boxes

def test():
    with open(PRIORS_PATH, "r") as f:
        priors = json.load(f)
    with open(VAL_SPLIT_FILE, "r") as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    gt_dict = load_ground_truth(VAL_SPLIT_FILE, LABELS_DIR)

    m128 = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_full_128/mobilenet_bbox_augmented_full_128_int8.tflite"
    m160_g = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160/mobilenet_bbox_augmented_160_int8.tflite"
    m160_c = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_ciou_160/mobilenet_bbox_ciou_160_int8.tflite"

    c128 = cache_preds(m128, val_imgs, 16, [0.0]*10, priors)
    c160_g = cache_preds(m160_g, val_imgs, 20, [0.0]*10, priors)
    c160_c = cache_preds(m160_c, val_imgs, 20, [0.0]*10, priors)
    models = [c128, c160_g, c160_c]

    conf = 0.12 # Can afford lower raw conf since filter catches solitary noise!
    
    for min_cluster in [1, 2]:
        for single_sc in [0.14, 0.20, 0.25, 0.30, 0.35]:
            pred_dict = {}
            total_fps = 0
            for img_p in val_imgs:
                sid = os.path.splitext(os.path.basename(img_p))[0]
                raw_boxes = []
                for mod in models:
                    probs, boxes, grid_size, _ = mod[sid]
                    for gy in range(grid_size):
                        for gx in range(grid_size):
                            cp = probs[gy, gx, 1:11]
                            bc = int(np.argmax(cp))
                            sc = float(cp[bc])
                            if sc >= conf:
                                p_dx, p_dy, p_w, p_h = float(boxes[gy, gx, 0]), float(boxes[gy, gx, 1]), float(boxes[gy, gx, 2]), float(boxes[gy, gx, 3])
                                cx = (gx + p_dx) / float(grid_size)
                                cy = (gy + p_dy) / float(grid_size)
                                a = OPT_ALPHAS[bc]
                                pw, ph = priors[str(bc)]["w"], priors[str(bc)]["h"]
                                w = (1.0 - a) * pw + a * p_w
                                h = (1.0 - a) * ph + a * p_h
                                xmin = max(0.0, (cx - w / 2.0) * ORIG_W)
                                ymin = max(0.0, (cy - h / 2.0) * ORIG_H)
                                xmax = min(ORIG_W, (cx + w / 2.0) * ORIG_W)
                                ymax = min(ORIG_H, (cy + h / 2.0) * ORIG_H)
                                raw_boxes.append({"class_id": bc, "score": sc, "bbox": [xmin, ymin, xmax, ymax]})
                final_boxes = []
                for c in range(10):
                    c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                    if not c_boxes: continue
                    fused = wbf_consensus(c_boxes, iou_thresh=0.28, boost_factor=0.03, min_cluster_size=min_cluster, min_single_score=single_sc)
                    final_boxes.extend(fused)
                final_boxes.sort(key=lambda b: b["score"], reverse=True)
                pred_dict[sid] = final_boxes[:100]

            mAP50, _ = calculate_map50(gt_dict, pred_dict)
            print(f"min_cluster={min_cluster}, single_score_thresh={single_sc:.2f} -> mAP@0.5 = {mAP50*100:.2f}%")

if __name__ == "__main__":
    test()
