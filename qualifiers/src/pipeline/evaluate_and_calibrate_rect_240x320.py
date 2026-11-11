"""
Evaluates and Calibrates the Converged Rectangular (240x320) Champion Model.
- Restores weights from best_rect_weights.weights.h5 (lowest val_loss: 2.5886).
- Evaluates FP32 validation mAP50 on 600 split images.
- Quantizes to Full Integer INT8 TFLite.
- Calibrates detection sweetspot and generates candidate submission.
"""

import os
import sys
import glob
import time
import json
import argparse
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
import tensorflow as tf
tf.config.optimizer.set_jit(False)
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

from src.detection.model_variants import (
    build_micro_edge_ultra_v2,
    NUM_CLASSES,
    MAX_ALLOWED_PARAMS,
    MAX_ALLOWED_MODEL_SIZE_KB
)
from src.detection.dataset import ORIG_W, ORIG_H
from src.detection.infer import per_class_nms
from src.evaluation import calculate_map50
from tune_sweetspot_calibration import run_calibration

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


def load_ground_truths_from_paths(label_paths):
    gt_dict = {}
    for lp in label_paths:
        sample_id = os.path.splitext(os.path.basename(lp))[0]
        boxes = []
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        xmin = max(0.0, (cx - w / 2.0) * ORIG_W)
                        ymin = max(0.0, (cy - h / 2.0) * ORIG_H)
                        xmax = min(ORIG_W, (cx + w / 2.0) * ORIG_W)
                        ymax = min(ORIG_H, (cy + h / 2.0) * ORIG_H)
                        boxes.append({"class_id": cid, "bbox": [xmin, ymin, xmax, ymax]})
        gt_dict[sample_id] = boxes
    return gt_dict


def evaluate_rect_model_map50(model, img_paths, label_dir, conf_thresh=0.08, iou_thresh=0.45):
    lbl_paths = [os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0] + ".txt") for p in img_paths]
    gt_dict = load_ground_truths_from_paths(lbl_paths)

    pred_dict = {}
    for p in img_paths:
        sample_id = os.path.splitext(os.path.basename(p))[0]
        im = cv2.imread(p)
        if im is None:
            pred_dict[sample_id] = []
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im.shape[:2] != (240, 320):
            im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
        norm = (im.astype(np.float32) / 255.0)[None, ...]

        out = model(norm, training=False)
        pred_conf = out["conf_output"][0].numpy()
        pred_cls = out["cls_output"][0].numpy()
        pred_box = out["box_output"][0].numpy()

        grid_h, grid_w = pred_conf.shape[:2]
        bxs, scs, cls_list = [], [], []

        for gy in range(grid_h):
            for gx in range(grid_w):
                conf = float(pred_conf[gy, gx, 0])
                if conf < conf_thresh:
                    continue
                cls_probs = pred_cls[gy, gx]
                max_c = int(np.argmax(cls_probs))
                score = conf * float(cls_probs[max_c])
                if score < conf_thresh:
                    continue
                off_x, off_y, bw, bh = pred_box[gy, gx]
                cx = (gx + off_x) / float(grid_w)
                cy = (gy + off_y) / float(grid_h)
                xmin = max(0.0, (cx - bw / 2.0) * ORIG_W)
                ymin = max(0.0, (cy - bh / 2.0) * ORIG_H)
                xmax = min(ORIG_W, (cx + bw / 2.0) * ORIG_W)
                ymax = min(ORIG_H, (cy + bh / 2.0) * ORIG_H)
                if (xmax - xmin) >= 4.0 and (ymax - ymin) >= 4.0:
                    bxs.append([xmin, ymin, xmax, ymax])
                    scs.append(score)
                    cls_list.append(max_c)

        det_objs = per_class_nms(bxs, scs, cls_list, num_classes=NUM_CLASSES, iou_thresh=iou_thresh)
        pred_dict[sample_id] = det_objs

    map50, class_aps = calculate_map50(gt_dict, pred_dict, num_classes=NUM_CLASSES)
    return map50, class_aps


def export_rect_int8_tflite(model, rep_img_paths, output_path, img_h=240, img_w=320):
    print(f"[*] Quantizing model with {len(rep_img_paths)} calibration images to Full Integer INT8 ({img_h}x{img_w})...")

    def rep_gen():
        for p in rep_img_paths:
            im = cv2.imread(p)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if im.shape[:2] != (img_h, img_w):
                im = cv2.resize(im, (img_w, img_h), interpolation=cv2.INTER_AREA)
            norm = (im.astype(np.float32) / 255.0)[None, ...]
            yield [norm]

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, img_h, img_w, 3], dtype=tf.float32)])
    def serve_fn(inp):
        return model(inp)

    cf = serve_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([cf])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_quant)

    size_kb = len(tflite_quant) / 1024.0
    print(f"[✔] INT8 Model generated: {output_path} ({size_kb:.1f} KB / Budget: {MAX_ALLOWED_MODEL_SIZE_KB} KB)")
    if size_kb > MAX_ALLOWED_MODEL_SIZE_KB:
        raise ValueError(f"FATAL: INT8 model size ({size_kb:.1f} KB) exceeds flash budget ({MAX_ALLOWED_MODEL_SIZE_KB} KB)!")
    return size_kb


def main():
    parser = argparse.ArgumentParser(description="Rectangular 240x320 Evaluation and Calibration Runner")
    parser.add_argument("--weights", default=os.path.join(PROJECT_ROOT, "weights/rect_240x320_champion/best_rect_weights.weights.h5"))
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "weights/rect_240x320_champion"))
    args = parser.parse_args()

    print("=" * 80)
    print("   EVALUATION & CALIBRATION: RECTANGULAR 240x320 NATIVE MODEL")
    print("=" * 80)

    model = build_micro_edge_ultra_v2(input_shape=(240, 320, 3), num_classes=NUM_CLASSES)
    total_params = model.count_params()
    print(f"[✔] Initialized MicroEdgeCNN_UltraV2 with {total_params:,} parameters (< {MAX_ALLOWED_PARAMS:,})")

    assert os.path.exists(args.weights), f"Weights file not found: {args.weights}"
    model.load_weights(args.weights)
    print(f"[✔] Successfully loaded optimal weights: {args.weights}")

    # Build dataset splits (same seed 42, 9:1 split)
    combined_img_dir = os.path.join(PROJECT_ROOT, "data/combined/images")
    combined_lbl_dir = os.path.join(PROJECT_ROOT, "data/combined/labels")
    all_imgs = sorted(glob.glob(os.path.join(combined_img_dir, "*.jpg")))

    import random
    rng = random.Random(42)
    shuffled_imgs = list(all_imgs)
    rng.shuffle(shuffled_imgs)

    n_total = len(shuffled_imgs)
    n_val = int(0.10 * n_total)
    train_imgs = shuffled_imgs[n_val:]
    val_imgs = shuffled_imgs[:n_val]
    print(f"[*] Dataset split: {len(train_imgs)} train / {len(val_imgs)} val images")

    # Evaluate FP32 mAP50
    print("\n[*] Evaluating FP32 mAP@50 on 600 validation images...")
    post_map, post_aps = evaluate_rect_model_map50(model, val_imgs, combined_lbl_dir, conf_thresh=0.08)
    print(f"[✔] FP32 Rectangular mAP@50: {post_map * 100:.2f}%")
    for c, ap in post_aps.items():
        print(f"    Class {c} ({CLASS_NAMES[c]}): {ap * 100:.2f}%")

    # Export INT8
    int8_tflite_path = os.path.join(args.output_dir, "model_int8.tflite")
    rep_sample_imgs = train_imgs[:150]
    export_rect_int8_tflite(model, rep_sample_imgs, int8_tflite_path, img_h=240, img_w=320)

    # Sweetspot density calibration
    print("\n" + "=" * 80)
    print("   AUTOMATIC SWEETSPOT DENSITY CALIBRATION (240x320 NATIVE INT8)")
    print("=" * 80)

    val_img_dir = os.path.join(PROJECT_ROOT, "data/val/images")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "data/val/labels")
    test_img_dir = os.path.join(PROJECT_ROOT, "data/test/images")
    intent_preds_path = os.path.join(PROJECT_ROOT, "submissions/intent_preds.csv")
    calibrated_csv = os.path.join(PROJECT_ROOT, "candidate_submissions/calibrated_rect_240x320_sweetspot.csv")
    calibrated_json = os.path.join(PROJECT_ROOT, "candidate_submissions/calibrated_rect_240x320_thresholds.json")

    run_calibration(
        model_path=int8_tflite_path,
        val_images_dir=val_img_dir,
        val_labels_dir=val_lbl_dir,
        test_images_dir=test_img_dir,
        intent_preds_csv=intent_preds_path,
        output_csv=calibrated_csv,
        output_json=calibrated_json,
        target_min=2600,
        target_max=2800,
        iou_thresh=0.45,
        max_boxes=100
    )


if __name__ == "__main__":
    main()
