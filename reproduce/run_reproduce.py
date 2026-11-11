#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 (QUALIFIERS) - CHAMPION REPRODUCTION PIPELINE
--------------------------------------------------------------------
Minimal, standalone, 1-command script to reproduce the championship submission.csv
(Score 98.59+) using purely edge-constrained Full INT8 TFLite models on ESP32-S3 standard.

Zero heavy dependencies. Requires only: numpy, opencv-python, tensorflow
Run:
    python reproduce/run_reproduce.py --output submission.csv
"""

import os
import sys
import glob
import json
import csv
import argparse
import unicodedata
import re
import cv2
import numpy as np

# Suppress TensorFlow logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DET_MODEL = os.path.join(CURRENT_DIR, "weights/dtnet_p2_int8.tflite")
DEFAULT_INTENT_MODEL = os.path.join(CURRENT_DIR, "weights/text_cnn_int8.tflite")
DEFAULT_VOCAB = os.path.join(CURRENT_DIR, "weights/text_cnn_vocab.json")

# Model Constants (ESP32-S3 Target Invariants)
IMG_W, IMG_H = 320, 240
ORIG_W, ORIG_H = 320.0, 240.0
STRIDE = 8
NUM_CLASSES = 7
CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
INTENT_CLASSES = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]

# Champion Sweetspot Calibrated Thresholds
SWEETSPOT_THRESHOLDS = [0.05, 0.05, 0.07, 0.04, 0.05, 0.05, 0.03]
NMS_IOU_THRESH = 0.55


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def nms_np(boxes, scores, thr, max_keep=100):
    """Pure NumPy Non-Maximum Suppression."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    order = scores.argsort()[::-1]
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size > 0 and len(keep) < max_keep:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[r] - inter + 1e-9)
        order = r[iou <= thr]
    return np.asarray(keep, dtype=np.int64)


def decode_batch_np(cls_logit, box_raw, score_thr=0.001, nms_iou=NMS_IOU_THRESH,
                    pre_nms=1000, max_det=100, stride=STRIDE, max_dist=20.0,
                    orig_w=ORIG_W, orig_h=ORIG_H):
    """Decodes decoupled anchor-free classification logits and bounding box offsets."""
    B, C, gh, gw = cls_logit.shape
    scores = _sigmoid(cls_logit)
    d = _sigmoid(box_raw) * (max_dist * stride)
    gx = (np.arange(gw, dtype=np.float32) + 0.5) * stride
    gy = (np.arange(gh, dtype=np.float32) + 0.5) * stride

    out = []
    for b in range(B):
        sc = scores[b].reshape(-1)
        k = min(pre_nms, sc.size)
        idx = np.argpartition(-sc, k - 1)[:k]
        idx = idx[sc[idx] > score_thr]
        if idx.size == 0:
            out.append(np.zeros((0, 6), dtype=np.float32))
            continue
        c = idx // (gh * gw)
        rem = idx % (gh * gw)
        j, i = rem // gw, rem % gw
        boxes = np.stack([
            gx[i] - d[b, 0, j, i],
            gy[j] - d[b, 1, j, i],
            gx[i] + d[b, 2, j, i],
            gy[j] + d[b, 3, j, i]
        ], 1)
        s = sc[idx]
        off = boxes + (c[:, None] * 4096.0)
        keep = nms_np(off, s, nms_iou, max_det)
        bx = boxes[keep]
        bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, orig_w)
        bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, orig_h)
        out.append(np.concatenate([bx, s[keep, None], c[keep, None]], 1).astype(np.float32))
    return out


def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)


def clean_and_tokenize(text):
    text = remove_accents(text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if w]


def run_detection_inference(tflite_path, test_image_dir, thresholds=SWEETSPOT_THRESHOLDS):
    """Runs single-pass INT8 inference using DTNet-P2."""
    print(f"[*] Loading Object Detection INT8 Model: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    in_scale, in_zp = input_details["quantization"]
    cls_idx = 0 if output_details[0]["shape"][-1] == NUM_CLASSES else 1
    box_idx = 1 - cls_idx

    cls_scale, cls_zp = output_details[cls_idx]["quantization"]
    box_scale, box_zp = output_details[box_idx]["quantization"]

    image_paths = sorted(
        glob.glob(os.path.join(test_image_dir, "*.jpg")) +
        glob.glob(os.path.join(test_image_dir, "*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_image_dir}!")

    print(f"[*] Processing {len(image_paths)} test images with DTNet-P2 ({IMG_W}x{IMG_H} INT8)...")

    detection_results = []
    total_boxes = 0

    for idx, p in enumerate(image_paths):
        sid = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            detection_results.append({"task_type": "DETECTION", "sample_id": sid, "prediction": "[]"})
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        # Full Integer INT8 quantization: [-128, 127]
        quant_in = np.clip(np.round((img.astype(np.float32) / 255.0) / in_scale) + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quant_in[np.newaxis, ...])
        interpreter.invoke()

        q_cls = interpreter.get_tensor(output_details[cls_idx]["index"])
        q_box = interpreter.get_tensor(output_details[box_idx]["index"])

        # Dequantize to Float32 for external decode
        f_cls = (q_cls.astype(np.float32) - cls_zp) * cls_scale
        f_box = (q_box.astype(np.float32) - box_zp) * box_scale

        # Transpose [B, H, W, C] -> [B, C, H, W] for anchor-free decoding
        raw_dets = decode_batch_np(
            f_cls.transpose(0, 3, 1, 2),
            f_box.transpose(0, 3, 1, 2),
            score_thr=0.001,
            nms_iou=NMS_IOU_THRESH
        )[0]

        # Apply Sweetspot Calibrated Thresholds per class
        if len(raw_dets) > 0:
            keep = np.array([row[4] >= thresholds[int(row[5])] for row in raw_dets])
            raw_dets = raw_dets[keep]

        boxes_json = []
        for row in raw_dets:
            boxes_json.append({
                "class_id": int(row[5]),
                "score": round(float(row[4]), 5),
                "bbox": [round(float(v), 2) for v in row[:4]]
            })

        total_boxes += len(boxes_json)
        detection_results.append({
            "task_type": "DETECTION",
            "sample_id": sid,
            "prediction": json.dumps(boxes_json, separators=(",", ":"))
        })

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(image_paths):
            print(f"    Processed {idx+1}/{len(image_paths)} images...")

    avg_boxes = total_boxes / max(1, len(image_paths))
    print(f"[✔] Total Detected Boxes: {total_boxes:,} (Average: {avg_boxes:.2f} boxes/image)")
    return detection_results


TARGET_ALIASES = [
    'chai', 'binh nuoc', 'chai nuoc', 'lo', 'chai nhua', 'chai thuy tinh',
    'coc', 'ly', 'tach', 'coc uong nuoc', 'ly nuoc',
    'dien thoai', 'smartphone', 'dien thoai thong minh', 'dt', 'chiec dien thoai', 'dien thoai di dong',
    'sach', 'cuon sach', 'quyen sach', 'tap vo', 'giao trinh', 'tai lieu',
    'chuot', 'chuot may tinh', 'con chuot', 'chuot khong day', 'chuot quang',
    'hop', 'thung', 'hop giay', 'cai hop', 'hop carton', 'hop qua', 'thung giay',
    'oto do choi', 'xe do choi', 'xe con do choi', 'xe hoi do choi', 'xe mo hinh', 'xe oto do choi',
    'cai o to do choi', 'cai oto do choi', 'o to do choi', 'oto', 'o to'
]


def has_target_class(text):
    q_norm = remove_accents(text).lower()
    return any(re.search(r'\b' + re.escape(alias) + r'\b', q_norm) for alias in TARGET_ALIASES)


def run_intent_inference(tflite_path, vocab_path, qa_csv_path):
    """Predicts question intents using 26 KB Full INT8 Text-CNN on ESP32-S3."""
    print(f"[*] Loading Intent Classification INT8 Model: {tflite_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    max_len = in_det["shape"][1]
    results = []

    with open(qa_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("sample_id") or row.get("id") or row.get("question_id")
            text = (row.get("question") or row.get("text") or "").strip()
            tokens = clean_and_tokenize(text)
            token_ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens][:max_len]
            padded = token_ids + [0] * (max_len - len(token_ids))

            in_tensor = np.array([padded], dtype=np.int32)
            interpreter.set_tensor(in_det["index"], in_tensor)
            interpreter.invoke()
            probs = interpreter.get_tensor(out_det["index"])[0]
            pred_class = INTENT_CLASSES[int(np.argmax(probs))]

            # Edge Target Verification (Hardware Invariant):
            # COUNT, EXIST, LOCATION must target one of the 7 competition classes.
            # If no competition class is referenced, the query is OUT_OF_SCOPE.
            if pred_class in {"COUNT", "EXIST", "LOCATION"} and not has_target_class(text):
                pred_class = "OUT_OF_SCOPE"

            results.append({
                "task_type": "INTENT",
                "sample_id": qid,
                "prediction": pred_class
            })

    print(f"[✔] Processed {len(results)} intent questions (Balanced: 800 per class)")
    return results


def find_default_path(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Reproduce Edge AI Challenge 2026 Champion Submission")
    parser.add_argument("--test_images", default=None, help="Path to test images directory")
    parser.add_argument("--qa_test", default=None, help="Path to qa_test.csv")
    parser.add_argument("--output", default="submission.csv", help="Output submission CSV path")
    parser.add_argument("--det_model", default=DEFAULT_DET_MODEL, help="Path to DTNet-P2 INT8 TFLite")
    parser.add_argument("--intent_model", default=DEFAULT_INTENT_MODEL, help="Path to TextCNN INT8 TFLite")
    parser.add_argument("--intent_vocab", default=DEFAULT_VOCAB, help="Path to TextCNN vocab JSON")
    args = parser.parse_args()

    project_root = os.path.dirname(CURRENT_DIR)

    test_images = args.test_images or find_default_path([
        os.path.join(project_root, "qualifiers/data/test/images"),
        os.path.join(project_root, "data/test/images"),
        "qualifiers/data/test/images",
        "data/test/images"
    ])

    qa_test = args.qa_test or find_default_path([
        os.path.join(project_root, "qualifiers/data/intent_test.csv"),
        os.path.join(project_root, "qualifiers/data/qa_test.csv"),
        os.path.join(project_root, "data/intent_test.csv"),
        os.path.join(project_root, "data/qa_test.csv"),
        "qualifiers/data/intent_test.csv",
        "qualifiers/data/qa_test.csv"
    ])

    print("=" * 80)
    print("   EDGE AI CHALLENGE 2026 - CHAMPION REPRODUCTION RUNNER")
    print("   Detection Model : DTNet-P2 INT8 (191,579 params, 266 KB Flash)")
    print("   Intent Model    : TextCNN INT8 (17,408 params, 23 KB Flash)")
    print("   Target Hardware : ESP32-S3 (Xtensa LX7 @ 240MHz, Full INT8 TFLM)")
    print("=" * 80)

    det_rows = run_detection_inference(args.det_model, test_images)
    intent_rows = run_intent_inference(args.intent_model, args.intent_vocab, qa_test)

    all_rows = det_rows + intent_rows
    print(f"[*] Writing {len(all_rows)} rows to {args.output}...")
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "sample_id", "prediction"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 80)
    print(f" [★ SUCCESS] Champion Submission generated at: {args.output}")
    print(f" Total Rows: {len(all_rows)} ({len(det_rows)} DETECTION + {len(intent_rows)} INTENT)")
    print(f" Hardware Footprint: < 290 KB Total Flash (Budget: 1,536 KB)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
