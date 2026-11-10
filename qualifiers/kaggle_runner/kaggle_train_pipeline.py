#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - KAGGLE DUAL T4x2 DISTRIBUTED TRAINER
=============================================================
High-Speed Distributed Training for Champion MicroEdgeCNN_Max (Trial 20 Architecture):
- Model: MicroEdgeCNN_Max (128,940 params, strictly < 200,000 ESP32-S3 bound)
- Dual T4 GPUs with MirroredStrategy & Batch Size 64 (32/GPU)
- Loss: CIoU conf=1.8, cls=1.8, box=2.8
- Tabletop Perspective Augmentation: pitch range (25.0, 55.0)
- Optimizer: Adam + CosineDecay (1.8e-3 -> 1.8e-5) across 45 epochs (~2-3 mins total)
- Full Integer INT8 PTQ Quantization exported to /kaggle/working/
"""

import os
import sys
import glob
import json
import time
import shutil
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

print(f"[*] TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"[*] Available Physical GPUs: {gpus}")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"[✔] MirroredStrategy initialized across {strategy.num_replicas_in_sync} GPUs!")
else:
    strategy = tf.distribute.get_strategy()
    print(f"[✔] Running on single device: {gpus[0] if gpus else 'CPU'}")

ORIG_W = 320.0
ORIG_H = 240.0
NUM_CLASSES = 7  # 0: bottle, 1: cup, 2: phone, 3: book, 4: mouse, 5: box, 6: toy_car
IMG_SIZE = 160
GRID_SIZE = 20
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. MODEL ARCHITECTURE: MicroEdgeCNN_Max (128,940 params)
# =============================================================================
def se_block(x, channels, reduction=4, name="se"):
    squeeze = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
    squeeze = layers.Dense(max(8, channels // reduction), activation="relu", use_bias=False, name=f"{name}_sq")(squeeze)
    excite = layers.Dense(channels, activation="sigmoid", use_bias=False, name=f"{name}_ex")(squeeze)
    return layers.Multiply(name=f"{name}_mul")([x, excite])

def build_micro_edge_max(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Stem (160 -> 80)
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="m_stem")(inputs)
    x = layers.BatchNormalization(name="m_stem_bn")(x)
    x = layers.ReLU(6.0, name="m_stem_relu")(x)

    # Stage 1 (80 -> 40)
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_s1_dw")(x)
    x = layers.BatchNormalization(name="m_s1_dw_bn")(x)
    x = layers.ReLU(6.0, name="m_s1_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="m_s1_pw")(x)
    x = layers.BatchNormalization(name="m_s1_pw_bn")(x)

    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="m_s1_down_dw")(x)
    x = layers.BatchNormalization(name="m_s1_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="m_s1_down_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="m_s1_down_pw")(x)
    x = layers.BatchNormalization(name="m_s1_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="m_s1_down_pw_relu")(x)

    # Stage 2: Geometric Shapes (40 -> 20) with Residual & SE
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_s2_dw1")(x)
    x = layers.BatchNormalization(name="m_s2_dw1_bn")(x)
    x = layers.ReLU(6.0, name="m_s2_dw1_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="m_s2_pw1")(x)
    x = layers.BatchNormalization(name="m_s2_pw1_bn")(x)
    x = layers.Add(name="m_s2_add1")([res, x])

    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="m_s2_down_dw")(x)
    x = layers.BatchNormalization(name="m_s2_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="m_s2_down_dw_relu")(x)
    x = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_s2_down_pw")(x)
    x = layers.BatchNormalization(name="m_s2_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="m_s2_down_pw_relu")(x)

    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_s2_dw2")(x)
    x = layers.BatchNormalization(name="m_s2_dw2_bn")(x)
    x = layers.ReLU(6.0, name="m_s2_dw2_relu")(x)
    x = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_s2_pw2")(x)
    x = layers.BatchNormalization(name="m_s2_pw2_bn")(x)
    x = se_block(x, channels=96, reduction=4, name="m_s2_se")
    shape_feat = layers.Add(name="m_s2_add2")([res, x])  # 20x20x96

    # Stage 3: High-level Semantics (20 -> 10)
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="m_s3_down_dw")(shape_feat)
    x = layers.BatchNormalization(name="m_s3_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="m_s3_down_dw_relu")(x)
    x = layers.Conv2D(144, 1, padding="same", use_bias=False, name="m_s3_down_pw")(x)
    x = layers.BatchNormalization(name="m_s3_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="m_s3_down_pw_relu")(x)

    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_s3_dw1")(x)
    x = layers.BatchNormalization(name="m_s3_dw1_bn")(x)
    x = layers.ReLU(6.0, name="m_s3_dw1_relu")(x)
    x = layers.Conv2D(144, 1, padding="same", use_bias=False, name="m_s3_pw1")(x)
    x = layers.BatchNormalization(name="m_s3_pw1_bn")(x)
    x = se_block(x, channels=144, reduction=4, name="m_s3_se")
    sem_feat = layers.Add(name="m_s3_add1")([res, x])  # 10x10x144

    # Lateral projection & upsampling (FPN-lite)
    sem_proj = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_s3_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="m_s3_up")(sem_proj)

    # Feature fusion at 20x20
    fused = layers.Concatenate(name="m_fused_cat")([shape_feat, sem_up])  # 20x20x192
    fused = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_fused_pw")(fused)
    fused = layers.BatchNormalization(name="m_fused_bn")(fused)
    fused = layers.ReLU(6.0, name="m_fused_relu")(fused)

    # Refinement block with DWConv
    res_f = fused
    fused = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_refine_dw")(fused)
    fused = layers.BatchNormalization(name="m_refine_dw_bn")(fused)
    fused = layers.ReLU(6.0, name="m_refine_dw_relu")(fused)
    fused = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_refine_pw")(fused)
    fused = layers.BatchNormalization(name="m_refine_pw_bn")(fused)
    fused = layers.Add(name="m_refine_add")([res_f, fused])

    # Decoupled heads
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(fused)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(fused)

    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_Max"
    )
    return model

# =============================================================================
# 2. DECOUPLED LOSS FUNCTIONS (TRIAL 20 SPECIFICATION)
# =============================================================================
def decoupled_conf_loss(y_true, y_pred):
    conf_mask = y_true[..., 0:1]
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt = tf.where(tf.equal(conf_mask, 1.0), y_pred, 1.0 - y_pred)
    focal_weight = 0.25 * tf.pow(1.0 - pt, 2.0)
    bce = -tf.math.log(pt)
    loss = focal_weight * bce
    num_pos = tf.maximum(1.0, tf.reduce_sum(conf_mask))
    return tf.reduce_sum(loss) / num_pos

def decoupled_cls_loss(y_true, y_pred):
    labels = y_true[..., :NUM_CLASSES]
    conf_mask = y_true[..., NUM_CLASSES:NUM_CLASSES+1]
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    ce = -tf.reduce_sum(labels * tf.math.log(y_pred), axis=-1, keepdims=True)
    loss = ce * conf_mask
    num_pos = tf.maximum(1.0, tf.reduce_sum(conf_mask))
    return tf.reduce_sum(loss) / num_pos

def decoupled_ciou_loss(y_true, y_pred):
    conf_mask = y_true[..., 4:5]
    grid_h = tf.cast(tf.shape(y_pred)[1], y_pred.dtype)
    grid_w = tf.cast(tf.shape(y_pred)[2], y_pred.dtype)

    gy, gx = tf.meshgrid(tf.range(grid_h, dtype=y_pred.dtype), tf.range(grid_w, dtype=y_pred.dtype), indexing="ij")
    gx = tf.expand_dims(tf.expand_dims(gx, 0), -1)
    gy = tf.expand_dims(tf.expand_dims(gy, 0), -1)

    pred_cx = (gx + y_pred[..., 0:1]) / grid_w
    pred_cy = (gy + y_pred[..., 1:2]) / grid_h
    pred_w = tf.clip_by_value(y_pred[..., 2:3], 1e-4, 1.0)
    pred_h = tf.clip_by_value(y_pred[..., 3:4], 1e-4, 1.0)

    true_cx = (gx + y_true[..., 0:1]) / grid_w
    true_cy = (gy + y_true[..., 1:2]) / grid_h
    true_w = tf.clip_by_value(y_true[..., 2:3], 1e-4, 1.0)
    true_h = tf.clip_by_value(y_true[..., 3:4], 1e-4, 1.0)

    b1_x1, b1_y1 = pred_cx - pred_w / 2.0, pred_cy - pred_h / 2.0
    b1_x2, b1_y2 = pred_cx + pred_w / 2.0, pred_cy + pred_h / 2.0
    b2_x1, b2_y1 = true_cx - true_w / 2.0, true_cy - true_h / 2.0
    b2_x2, b2_y2 = true_cx + true_w / 2.0, true_cy + true_h / 2.0

    inter_x1 = tf.maximum(b1_x1, b2_x1)
    inter_y1 = tf.maximum(b1_y1, b2_y1)
    inter_x2 = tf.minimum(b1_x2, b2_x2)
    inter_y2 = tf.minimum(b1_y2, b2_y2)
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = pred_w * pred_h
    b2_area = true_w * true_h
    union_area = b1_area + b2_area - inter_area + 1e-7
    iou = tf.clip_by_value(inter_area / union_area, 0.0, 1.0)

    c_x1 = tf.minimum(b1_x1, b2_x1)
    c_y1 = tf.minimum(b1_y1, b2_y1)
    c_x2 = tf.maximum(b1_x2, b2_x2)
    c_y2 = tf.maximum(b1_y2, b2_y2)
    c_w = tf.maximum(0.0, c_x2 - c_x1)
    c_h = tf.maximum(0.0, c_y2 - c_y1)
    c2 = tf.square(c_w) + tf.square(c_h) + 1e-7

    rho2 = tf.square(pred_cx - true_cx) + tf.square(pred_cy - true_cy)
    pi = tf.constant(np.pi, dtype=y_pred.dtype)
    v = (4.0 / (np.pi ** 2)) * tf.square(tf.atan(true_w / true_h) - tf.atan(pred_w / pred_h))
    alpha = v / (1.0 - iou + v + 1e-7)
    alpha = tf.stop_gradient(alpha)

    ciou = iou - (rho2 / c2) - (alpha * v)
    loss = (1.0 - ciou) * conf_mask
    num_pos = tf.maximum(1.0, tf.reduce_sum(conf_mask))
    return tf.reduce_sum(loss) / num_pos

# =============================================================================
# 3. TABLETOP AUGMENTATIONS & DATASET LOADER
# =============================================================================
def apply_tabletop_perspective(image, boxes, pitch_range=(25.0, 55.0), prob=0.5):
    if image is None or image.size == 0 or len(boxes) == 0 or np.random.rand() > prob:
        return image, boxes

    img_h, img_w = image.shape[:2]
    pitch = np.random.uniform(pitch_range[0], pitch_range[1])
    pitch_norm = (pitch - pitch_range[0]) / max(1e-5, (pitch_range[1] - pitch_range[0]))

    top_squeeze = 0.15 + 0.20 * pitch_norm
    top_y_shift = 0.05 + 0.15 * pitch_norm
    roll_skew = np.random.uniform(-0.03, 0.03)

    src = np.array([[0.0, 0.0], [float(img_w), 0.0], [float(img_w), float(img_h)], [0.0, float(img_h)]], dtype=np.float32)
    dst = np.array([
        [(top_squeeze + roll_skew) * float(img_w), top_y_shift * float(img_h)],
        [(1.0 - top_squeeze + roll_skew) * float(img_w), top_y_shift * float(img_h)],
        [float(img_w), float(img_h)],
        [0.0, float(img_h)]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(image, M, (img_w, img_h), borderMode=cv2.BORDER_REFLECT101)

    warped_boxes = []
    for cid, cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2.0) * float(img_w)
        y1 = (cy - bh / 2.0) * float(img_h)
        x2 = (cx + bw / 2.0) * float(img_w)
        y2 = (cy + bh / 2.0) * float(img_h)

        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(4, 1, 2)
        p_warped = cv2.perspectiveTransform(corners, M).reshape(4, 2)
        if not np.all(np.isfinite(p_warped)):
            continue

        wx1, wy1 = float(np.min(p_warped[:, 0])), float(np.min(p_warped[:, 1]))
        wx2, wy2 = float(np.max(p_warped[:, 0])), float(np.max(p_warped[:, 1]))
        if wx2 < wx1: wx1, wx2 = wx2, wx1
        if wy2 < wy1: wy1, wy2 = wy2, wy1

        wx1, wy1 = max(0.0, min(wx1, float(img_w))), max(0.0, min(wy1, float(img_h)))
        wx2, wy2 = max(0.0, min(wx2, float(img_w))), max(0.0, min(wy2, float(img_h)))
        wbw, wbh = wx2 - wx1, wy2 - wy1

        if wbw >= 4.0 and wbh >= 4.0:
            wcx = max(0.0, min(1.0, (wx1 + wx2) / (2.0 * float(img_w))))
            wcy = max(0.0, min(1.0, (wy1 + wy2) / (2.0 * float(img_h))))
            new_bw = max(0.0, min(1.0, wbw / float(img_w)))
            new_bh = max(0.0, min(1.0, wbh / float(img_h)))
            warped_boxes.append((cid, wcx, wcy, new_bw, new_bh))

    return warped_img, warped_boxes

class QualifierDetectionDataset(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels_dir, img_size=160, grid_size=20,
                 batch_size=64, augment=True, num_classes=NUM_CLASSES):
        self.image_paths = image_paths
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.augment = augment
        self.num_classes = num_classes
        self.indices = np.arange(len(self.image_paths))
        if self.augment:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_imgs, batch_conf, batch_cls, batch_box = [], [], [], []

        for b_i in batch_idx:
            img_path = self.image_paths[b_i]
            sample_id = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_dir, f"{sample_id}.txt")

            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((int(ORIG_H), int(ORIG_W), 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            if 0 <= cid < self.num_classes:
                                cx, cy, w, h = map(float, parts[1:5])
                                boxes.append((cid, cx, cy, w, h))

            if self.augment and len(boxes) > 0:
                img, boxes = apply_tabletop_perspective(img, boxes, pitch_range=(25.0, 55.0), prob=0.5)
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    boxes = [(cid, 1.0 - cx, cy, w, h) for (cid, cx, cy, w, h) in boxes]

            resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            norm_img = resized.astype(np.float32) / 255.0

            conf_grid = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
            cls_grid = np.zeros((self.grid_size, self.grid_size, self.num_classes), dtype=np.float32)
            box_grid = np.zeros((self.grid_size, self.grid_size, 5), dtype=np.float32)

            for cid, cx, cy, w, h in boxes:
                gx = int(cx * self.grid_size)
                gy = int(cy * self.grid_size)
                gx = min(max(gx, 0), self.grid_size - 1)
                gy = min(max(gy, 0), self.grid_size - 1)

                offset_x = (cx * self.grid_size) - gx
                offset_y = (cy * self.grid_size) - gy

                conf_grid[gy, gx, 0] = 1.0
                cls_grid[gy, gx, cid] = 1.0
                box_grid[gy, gx, 0] = offset_x
                box_grid[gy, gx, 1] = offset_y
                box_grid[gy, gx, 2] = w
                box_grid[gy, gx, 3] = h
                box_grid[gy, gx, 4] = 1.0

            cls_target = np.concatenate([cls_grid, conf_grid], axis=-1)
            batch_imgs.append(norm_img)
            batch_conf.append(conf_grid)
            batch_cls.append(cls_target)
            batch_box.append(box_grid)

        return (
            np.array(batch_imgs, dtype=np.float32),
            {
                "conf_output": np.array(batch_conf, dtype=np.float32),
                "cls_output": np.array(batch_cls, dtype=np.float32),
                "box_output": np.array(batch_box, dtype=np.float32),
            }
        )

# =============================================================================
# 4. DATASET DISCOVERY
# =============================================================================
def locate_dataset():
    primary = "/kaggle/input/qualifiers-dataset"
    if os.path.exists(os.path.join(primary, "train", "images")):
        print(f"[*] Found attached qualifiers-dataset at: {primary}")
        train_imgs = sorted(glob.glob(os.path.join(primary, "train", "images", "*.jpg")) + glob.glob(os.path.join(primary, "train", "images", "*.png")))
        train_lbls = os.path.join(primary, "train", "labels")
        val_imgs = sorted(glob.glob(os.path.join(primary, "val", "images", "*.jpg")) + glob.glob(os.path.join(primary, "val", "images", "*.png")))
        val_lbls = os.path.join(primary, "val", "labels")
        return train_imgs, train_lbls, val_imgs, val_lbls

    candidates = glob.glob("/kaggle/input/**/train/images", recursive=True)
    if candidates:
        train_img_dir = candidates[0]
        root_dir = os.path.dirname(os.path.dirname(train_img_dir))
        print(f"[*] Found attached dataset at: {root_dir}")
        train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")) + glob.glob(os.path.join(train_img_dir, "*.png")))
        train_lbls = os.path.join(root_dir, "train", "labels")
        val_imgs = sorted(glob.glob(os.path.join(root_dir, "val", "images", "*.jpg")) + glob.glob(os.path.join(root_dir, "val", "images", "*.png")))
        val_lbls = os.path.join(root_dir, "val", "labels")
        return train_imgs, train_lbls, val_imgs, val_lbls

    raise FileNotFoundError("Could not find qualifiers dataset in /kaggle/input!")

# =============================================================================
# 5. FULL INT8 QUANTIZATION (ESP32-S3 COMPLIANT)
# =============================================================================
def quantize_and_export_int8(keras_model, train_imgs, out_tflite_path):
    print("=" * 70)
    print("   QUANTIZING TO FULL INTEGER INT8 FOR ESP32-S3 COMPLIANCE")
    print("=" * 70)
    params = keras_model.count_params()
    print(f"[*] Verifying params: {params:,} (Limit: 200,000)")

    def rep_gen():
        for p in train_imgs[:100]:
            im = cv2.imread(p)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
            yield [(im.astype(np.float32) / 255.0)[None, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant = converter.convert()
    size_kb = len(tflite_quant) / 1024.0
    print(f"[✔] INT8 Model Size: {size_kb:.2f} KB (ESP32-S3 Limit: 1500 KB)")

    with open(out_tflite_path, "wb") as f:
        f.write(tflite_quant)
    print(f"[✔] Successfully exported: {out_tflite_path}")

# =============================================================================
# 6. MAIN TRAINING ENTRYPOINT
# =============================================================================
def main():
    train_imgs, train_lbl_dir, val_imgs, val_lbl_dir = locate_dataset()
    print(f"[*] Training on {len(train_imgs)} images, validating on {len(val_imgs)} images.")

    batch_size = 64 if len(gpus) > 1 else 32
    train_ds = QualifierDetectionDataset(train_imgs, train_lbl_dir, img_size=IMG_SIZE,
                                         grid_size=GRID_SIZE, batch_size=batch_size, augment=True)
    val_ds = QualifierDetectionDataset(val_imgs, val_lbl_dir, img_size=IMG_SIZE,
                                       grid_size=GRID_SIZE, batch_size=batch_size, augment=False) if val_imgs else None

    epochs = 45
    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch * epochs

    with strategy.scope():
        model = build_micro_edge_max(input_shape=(IMG_SIZE, IMG_SIZE, 3))
        params = model.count_params()
        print(f"[✔] Initialized MicroEdgeCNN_Max: {params:,} parameters (Budget: 200,000)")

        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=1.8e-3,
            decay_steps=total_steps,
            alpha=0.01  # Minimum LR: 1.8e-5
        )
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=optimizer,
            loss={
                "conf_output": decoupled_conf_loss,
                "cls_output": decoupled_cls_loss,
                "box_output": decoupled_ciou_loss
            },
            loss_weights={
                "conf_output": 1.8,
                "cls_output": 1.8,
                "box_output": 2.8
            }
        )

    best_keras = os.path.join(OUTPUT_DIR, "best_weights.keras")
    cbs = [
        callbacks.ModelCheckpoint(best_keras, monitor="val_loss" if val_ds else "loss",
                                  save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss" if val_ds else "loss", patience=15,
                                min_delta=1e-3, restore_best_weights=True, verbose=1)
    ]

    print("[*] Starting High-Speed Dual T4 Distributed Training...")
    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs,
        verbose=1
    )
    print(f"[✔] Dual T4 Training finished in {time.time() - t0:.1f}s!")

    # Reload best weights
    model = tf.keras.models.load_model(best_keras, custom_objects={
        "decoupled_conf_loss": decoupled_conf_loss,
        "decoupled_cls_loss": decoupled_cls_loss,
        "decoupled_ciou_loss": decoupled_ciou_loss
    }, compile=False)

    # Export INT8 model
    out_tflite = os.path.join(OUTPUT_DIR, "micro_edge_max_int8.tflite")
    quantize_and_export_int8(model, train_imgs, out_tflite)
    print("\n[✔] ALL KAGGLE TRAINING AND QUANTIZATION FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
