#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - RECTANGULAR 240x320 (NATIVE 4:3 OV7670) DEEP CONVERGENCE
---------------------------------------------------------------------------------
Architecture: MicroEdgeCNN_UltraV2 (179,068 parameters < 200,000 limit)
Input Shape : (240, 320, 3) - Native 4:3 Aspect Ratio (Zero Image Distortion)
Dataset     : 6,000 annotated images split 9:1 (5,400 Train / 600 Val)
Horizon     : Up to 200 Epochs with EarlyStopping Patience = 25
Quantization: Full Integer INT8 (int8 in/out, <= 1.5MB budget)
Calibration : Coordinate Descent + Sweetspot Density Calibration (43.3 - 46.7 boxes/img)
"""

import os
import sys
import time
import glob
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
from tensorflow.keras import optimizers, callbacks

from src.detection.model_variants import (
    build_micro_edge_ultra_v2,
    NUM_CLASSES,
    MAX_ALLOWED_PARAMS,
    MAX_ALLOWED_MODEL_SIZE_KB
)
from src.detection.dataset import QualifierDetectionDataset, ORIG_W, ORIG_H
from src.detection.losses import decoupled_conf_loss, decoupled_cls_loss, decoupled_ciou_loss
from src.detection.infer import per_class_nms
from src.evaluation import calculate_map50
from tune_sweetspot_calibration import run_calibration

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


class PatienceCountdownCallback(callbacks.Callback):
    """
    Visual EarlyStopping Progress & Patience Countdown Monitor.
    Monitors val_loss and displays countdown from patience down to 0.
    """
    def __init__(self, patience=25, min_delta=1e-4, total_epochs=200):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.total_epochs = total_epochs
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.wait_count = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.wait_count = 0
        print(f"\n[*] PatienceMonitor: Max Epochs={self.total_epochs} | EarlyStopping Patience={self.patience}")
        print("    Monitoring: 'val_loss' (min_delta = 1e-4)\n", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss", float("inf"))
        tr_loss = logs.get("loss", float("inf"))

        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.wait_count = 0
            remaining = self.patience
            status_tag = "[★ NEW BEST]"
        else:
            self.wait_count += 1
            remaining = self.patience - self.wait_count
            status_tag = f"[Patience: {remaining}/{self.patience}]"

        print(
            f"--> Epoch {epoch+1:03d}/{self.total_epochs:03d} | "
            f"train_loss: {tr_loss:.4f} | val_loss: {val_loss:.4f} "
            f"(best: {self.best_val_loss:.4f} @ Ep {self.best_epoch}) {status_tag}",
            flush=True
        )

        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch + 1
            self.model.stop_training = True
            print(f"\n[!] EarlyStopping triggered at Epoch {self.stopped_epoch}! Best val_loss: {self.best_val_loss:.4f} at Epoch {self.best_epoch}\n", flush=True)


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


def evaluate_rect_model_map50(model, val_imgs, labels_dir, conf_thresh=0.12, iou_thresh=0.45):
    """Evaluates mAP@50 for (240, 320) rectangular model."""
    val_lbl_paths = [os.path.join(labels_dir, f"{os.path.splitext(os.path.basename(p))[0]}.txt") for p in val_imgs]
    gt_dict = load_ground_truths_from_paths(val_lbl_paths)

    pred_dict = {}
    for p in val_imgs:
        sample_id = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (240, 320):
            resized = cv2.resize(img_rgb, (320, 240), interpolation=cv2.INTER_AREA)
        else:
            resized = img_rgb
        norm_in = (resized.astype(np.float32) / 255.0)[None, ...]
        preds = model(norm_in, training=False)

        pred_conf = preds["conf_output"].numpy()[0]
        pred_cls = preds["cls_output"].numpy()[0]
        pred_box = preds["box_output"].numpy()[0]

        grid_h, grid_w = pred_box.shape[0], pred_box.shape[1]
        bxs, scs, cls_list = [], [], []

        for gy in range(grid_h):
            for gx in range(grid_w):
                conf = float(pred_conf[gy, gx, 0])
                cls_p = pred_cls[gy, gx]
                max_c = int(np.argmax(cls_p))
                score = conf * float(cls_p[max_c])
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
    """Converts Keras model to Full Integer INT8 TFLite with (240, 320) input."""
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
    parser = argparse.ArgumentParser(description="Rectangular 240x320 MicroEdgeCNN Deep Convergence Runner")
    parser.add_argument("--warmstart", default=os.path.join(PROJECT_ROOT, "weights/continuous_loop/trial_150/best_weights.keras"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--min_lr", type=float, default=3e-6)
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "weights/rect_240x320_champion"))
    args = parser.parse_args()

    print("=" * 80)
    print("   EDGE AI CHALLENGE 2026 - RECTANGULAR 240x320 NATIVE 4:3 DEEP CONVERGENCE")
    print(f"   Architecture    : MicroEdgeCNN_UltraV2 (Input: 240x320x3, Grid: 30x40)")
    print(f"   Base Champion   : {args.warmstart}")
    print(f"   Dataset Split   : 9:1 (5,400 Train / 600 Val across 6,000 images)")
    print(f"   Epochs & Patience: Max {args.epochs} Epochs | EarlyStopping Patience = {args.patience}")
    print(f"   Batch Size & LR : Batch {args.batch_size} | LR: {args.lr:.2e} -> {args.min_lr:.2e}")
    print(f"   Target Invariant: < 200,000 params | Full INT8 | Single Model | Sweet Spot N")
    print("=" * 80)

    # 1. Prepare Full 6,000 Image Pool & Create 9:1 Split
    all_imgs = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data/train/images/*.jpg")) +
                      glob.glob(os.path.join(PROJECT_ROOT, "data/train/images/*.png")) +
                      glob.glob(os.path.join(PROJECT_ROOT, "data/val/images/*.jpg")) +
                      glob.glob(os.path.join(PROJECT_ROOT, "data/val/images/*.png")))
    total_imgs = len(all_imgs)
    print(f"[*] Total dataset pool loaded: {total_imgs} annotated images")

    # Pool labels directory
    combined_lbl_dir = os.path.join(PROJECT_ROOT, "data/kfold_pool_labels")
    os.makedirs(combined_lbl_dir, exist_ok=True)
    all_label_files = glob.glob(os.path.join(PROJECT_ROOT, "data/train/labels/*.txt")) + \
                     glob.glob(os.path.join(PROJECT_ROOT, "data/val/labels/*.txt"))
    for lf in all_label_files:
        dst = os.path.join(combined_lbl_dir, os.path.basename(lf))
        if not os.path.exists(dst):
            try:
                os.symlink(lf, dst)
            except Exception:
                import shutil
                shutil.copy2(lf, dst)

    # Reproducible 9:1 Split (seed 42)
    np.random.seed(42)
    shuffled_idx = np.random.permutation(total_imgs)
    val_size = int(round(total_imgs * 0.10))
    val_indices = shuffled_idx[:val_size]
    train_indices = shuffled_idx[val_size:]

    train_imgs = [all_imgs[i] for i in train_indices]
    val_imgs = [all_imgs[i] for i in val_indices]
    print(f"[✔] 9:1 Partition Created: {len(train_imgs)} Train images (90%) | {len(val_imgs)} Val images (10%)")

    # 2. Build Rectangular 240x320 Architecture
    img_h, img_w = 240, 320
    grid_h, grid_w = img_h // 8, img_w // 8
    model = build_micro_edge_ultra_v2(input_shape=(img_h, img_w, 3), num_classes=NUM_CLASSES)
    total_params = model.count_params()
    print(f"[✔] Model initialized: MicroEdgeCNN_UltraV2 with {total_params:,} parameters (< {MAX_ALLOWED_PARAMS:,})")
    print(f"    Input Shape: ({img_h}, {img_w}, 3) | Detection Grid: ({grid_h}, {grid_w}) = {grid_h * grid_w} cells")

    # 3. Load Champion Pretrained Weights (Transfer learning from 160x160 champion)
    warmstart_path = args.warmstart
    if not os.path.exists(warmstart_path):
        alt_paths = [
            os.path.join(PROJECT_ROOT, "weights/champion_ultrav2_best.keras"),
            os.path.join(PROJECT_ROOT, "weights/champion_ultra_best.keras")
        ]
        for ap in alt_paths:
            if os.path.exists(ap):
                warmstart_path = ap
                break

    if warmstart_path and os.path.exists(warmstart_path):
        print(f"[*] Loading pretrained champion weights: {warmstart_path}")
        try:
            custom_objects = {
                "decoupled_conf_loss": decoupled_conf_loss,
                "decoupled_cls_loss": decoupled_cls_loss,
                "decoupled_ciou_loss": decoupled_ciou_loss,
                "decoupled_box_loss": decoupled_ciou_loss
            }
            src_m = tf.keras.models.load_model(warmstart_path, custom_objects=custom_objects, compile=False)
            model.set_weights(src_m.get_weights())
            print(f"[✔] Successfully transferred 100% of weights from Champion to Rectangular model!")
        except Exception as e:
            print(f"[*] Fallback loading via load_weights: {e}")
            model.load_weights(warmstart_path)
            print("[✔] Weights loaded via load_weights!")

    # Baseline evaluation on validation split
    print("\n[*] Evaluating Pre-Tuning Baseline on 600 validation images (Native 240x320)...")
    base_map, base_aps = evaluate_rect_model_map50(model, val_imgs, combined_lbl_dir, conf_thresh=0.12)
    print(f"[✔] Pre-Tuning Rectangular Baseline mAP@50: {base_map * 100:.2f}%")
    for c, ap in base_aps.items():
        print(f"    Class {c} ({CLASS_NAMES[c]}): {ap * 100:.2f}%")

    # 4. Dataset Pipeline Setup
    train_ds = QualifierDetectionDataset(
        train_imgs, combined_lbl_dir,
        img_size=(img_h, img_w), grid_size=(grid_h, grid_w), batch_size=args.batch_size,
        augment=True, augment_perspective=True, augment_photometric=True, augment_occlusion=True
    )
    val_ds = QualifierDetectionDataset(
        val_imgs, combined_lbl_dir,
        img_size=(img_h, img_w), grid_size=(grid_h, grid_w), batch_size=args.batch_size,
        augment=False, augment_perspective=False, augment_photometric=False, augment_occlusion=False
    )

    # 5. Optimizer & Cosine Decay Schedule
    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch * args.epochs
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=args.min_lr / args.lr
    )
    opt = optimizers.Adam(learning_rate=lr_schedule)

    losses = {
        "conf_output": decoupled_conf_loss,
        "cls_output": decoupled_cls_loss,
        "box_output": decoupled_ciou_loss
    }
    loss_weights = {"conf_output": 1.0, "cls_output": 2.5, "box_output": 4.5}

    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, jit_compile=False)

    # 6. Callbacks & Checkpoints
    os.makedirs(args.output_dir, exist_ok=True)
    best_weights_path = os.path.join(args.output_dir, "best_rect_weights.weights.h5")
    final_model_path = os.path.join(args.output_dir, "final_rect_model.keras")

    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=best_weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )
    patience_cb = PatienceCountdownCallback(patience=args.patience, total_epochs=args.epochs)
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.6,
        patience=8,
        min_lr=args.min_lr,
        verbose=1
    )

    # 7. Execute Deep Training Loop (200 Epoch Horizon)
    print(f"\n[*] Starting Deep Convergence Training (Up to {args.epochs} Epochs, Patience={args.patience})...\n")
    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt_cb, patience_cb, reduce_lr_cb],
        verbose=0
    )
    duration = time.time() - t0
    print(f"\n[✔] Training loop completed in {duration / 60.0:.1f} minutes!")

    # Restore optimal weights from lowest val_loss
    print("\n[*] Restoring optimal weights from lowest validation loss...")
    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
        print(f"[✔] Best weights successfully restored from {best_weights_path}")
    model.save(final_model_path)
    print(f"[✔] Full Keras model saved to {final_model_path}")

    # Post-tuning evaluation
    print("\n[*] Evaluating Post-Tuning mAP@50 on 600 validation images...")
    post_map, post_aps = evaluate_rect_model_map50(model, val_imgs, combined_lbl_dir, conf_thresh=0.12)
    print(f"[✔] Post-Tuning Rectangular mAP@50: {post_map * 100:.2f}% (Baseline: {base_map * 100:.2f}%)")
    for c, ap in post_aps.items():
        delta = (ap - base_aps[c]) * 100.0
        print(f"    Class {c} ({CLASS_NAMES[c]}): {ap * 100:.2f}% (delta: {delta:+.2f}%)")

    # 8. Export Full Integer INT8 TFLite Model
    int8_tflite_path = os.path.join(args.output_dir, "model_int8.tflite")
    rep_sample_imgs = train_imgs[:120]
    export_rect_int8_tflite(model, rep_sample_imgs, int8_tflite_path, img_h=img_h, img_w=img_w)

    # 9. Automatic Sweetspot Density Calibration & Candidate Generation
    print("\n" + "=" * 80)
    print("   AUTOMATIC SWEETSPOT DENSITY CALIBRATION (240x320 NATIVE)")
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
