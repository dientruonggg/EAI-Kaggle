#!/usr/bin/env python3
"""
TRAIN MOBILENETV2-BBOX 160x160 WITH NEGATIVE SAMPLING & EIOU LOSS
Target: Microcontroller ESP32-S3 (< 200,000 parameters, Full Integer INT8, <= 1.5 MB)
Dataset: 120 train images + 24 negative background crops (zero overlap with GT boxes)
Validation: clean 20-image val_split.txt
"""

import os
import sys
import glob
import json
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, models

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

ORIG_W = 1672.0
ORIG_H = 941.0
NUM_CLASSES = 10
GRID_SIZE = 20
IMG_SIZE = 160

grid_x = np.tile(np.arange(GRID_SIZE, dtype=np.float32), (GRID_SIZE, 1))
grid_y = np.tile(np.arange(GRID_SIZE, dtype=np.float32)[:, None], (1, GRID_SIZE))
tf_gx = tf.constant(grid_x[None, :, :, None], dtype=tf.float32)
tf_gy = tf.constant(grid_y[None, :, :, None], dtype=tf.float32)

def build_model(pretrained_path=None, img_size=160, alpha=0.50):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet"
    )
    
    feat_stride8 = base.get_layer("block_5_add").output
    feat_stride16 = base.get_layer("block_12_add").output
    
    up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="fpn_upsample")(feat_stride16)
    fused = layers.Concatenate(name="fpn_fused")([feat_stride8, up])
    
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="fomo_refine")(fused)
    x = layers.BatchNormalization(name="fomo_refine_bn")(x)
    x = layers.ReLU(6.0, name="fomo_refine_relu")(x)
    
    cls_out = layers.Conv2D(NUM_CLASSES + 1, 1, padding="same", activation="softmax", name="cls_output")(x)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(x)
    
    model = models.Model(inputs=base.inputs, outputs={"cls_output": cls_out, "box_output": box_out}, name="MobileNetV2_EIoU_Neg_160")
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[*] Loading pretrained weights from: {pretrained_path}")
        src = tf.keras.models.load_model(pretrained_path, compile=False)
        matched = 0
        for l in model.layers:
            try:
                l.set_weights(src.get_layer(l.name).get_weights())
                matched += 1
            except Exception:
                pass
        print(f"[✔] Successfully transferred weights for {matched} layers!")
        
    return model

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        focal_weight = alpha_t * tf.pow(1.0 - pt, gamma)
        bce = -tf.math.log(pt)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * bce, axis=-1))
    return loss

def eiou_loss():
    def loss(y_true, y_pred):
        obj_mask = y_true[..., 4:5]
        
        pred_dx = y_pred[..., 0:1]
        pred_dy = y_pred[..., 1:2]
        pred_w = tf.clip_by_value(y_pred[..., 2:3], 1e-4, 1.0)
        pred_h = tf.clip_by_value(y_pred[..., 3:4], 1e-4, 1.0)
        
        pred_cx = (tf_gx + pred_dx) / float(GRID_SIZE)
        pred_cy = (tf_gy + pred_dy) / float(GRID_SIZE)
        
        true_dx = y_true[..., 0:1]
        true_dy = y_true[..., 1:2]
        true_w = tf.clip_by_value(y_true[..., 2:3], 1e-4, 1.0)
        true_h = tf.clip_by_value(y_true[..., 3:4], 1e-4, 1.0)
        
        true_cx = (tf_gx + true_dx) / float(GRID_SIZE)
        true_cy = (tf_gy + true_dy) / float(GRID_SIZE)
        
        b1_x1 = pred_cx - pred_w / 2.0
        b1_y1 = pred_cy - pred_h / 2.0
        b1_x2 = pred_cx + pred_w / 2.0
        b1_y2 = pred_cy + pred_h / 2.0
        
        b2_x1 = true_cx - true_w / 2.0
        b2_y1 = true_cy - true_h / 2.0
        b2_x2 = true_cx + true_w / 2.0
        b2_y2 = true_cy + true_h / 2.0
        
        inter_x1 = tf.maximum(b1_x1, b2_x1)
        inter_y1 = tf.maximum(b1_y1, b2_y1)
        inter_x2 = tf.minimum(b1_x2, b2_x2)
        inter_y2 = tf.minimum(b1_y2, b2_y2)
        inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
        inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        b1_area = tf.maximum(0.0, b1_x2 - b1_x1) * tf.maximum(0.0, b1_y2 - b1_y1)
        b2_area = tf.maximum(0.0, b2_x2 - b2_x1) * tf.maximum(0.0, b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + 1e-7
        iou = inter_area / union_area
        
        c_x1 = tf.minimum(b1_x1, b2_x1)
        c_y1 = tf.minimum(b1_y1, b2_y1)
        c_x2 = tf.maximum(b1_x2, b2_x2)
        c_y2 = tf.maximum(b1_y2, b2_y2)
        c_w = tf.maximum(0.0, c_x2 - c_x1)
        c_h = tf.maximum(0.0, c_y2 - c_y1)
        c2 = tf.square(c_w) + tf.square(c_h) + 1e-7
        
        rho2_center = tf.square(pred_cx - true_cx) + tf.square(pred_cy - true_cy)
        rho2_w = tf.square(pred_w - true_w)
        rho2_h = tf.square(pred_h - true_h)
        
        eiou = iou - (rho2_center / c2) - (rho2_w / tf.square(c_w)) - (rho2_h / tf.square(c_h))
        loss_val = 1.0 - eiou
        
        num_objs = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
        return tf.reduce_sum(loss_val * obj_mask) / num_objs
    return loss

def apply_augmentation(img, boxes):
    aug_img = img.copy()
    aug_boxes = []
    
    # 1. Color Jitter
    if random.random() < 0.5:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.2), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.8, 1.2), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
    # 2. Random horizontal flip
    if random.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
        for cid, xmin, ymin, xmax, ymax in boxes:
            new_xmin = 1.0 - xmax
            new_xmax = 1.0 - xmin
            aug_boxes.append((cid, new_xmin, ymin, new_xmax, ymax))
    else:
        for b in boxes:
            aug_boxes.append(b)
            
    return aug_img, aug_boxes

def encode_targets(boxes, grid_size=GRID_SIZE):
    cls_target = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
    cls_target[..., 0] = 1.0 # default background
    
    box_target = np.zeros((grid_size, grid_size, 5), dtype=np.float32)
    sorted_boxes = sorted(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]), reverse=True)
    
    for cid, xmin, ymin, xmax, ymax in sorted_boxes:
        bw = xmax - xmin
        bh = ymax - ymin
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        
        gx_f = cx * grid_size
        gy_f = cy * grid_size
        gx = int(np.clip(gx_f, 0, grid_size - 1))
        gy = int(np.clip(gy_f, 0, grid_size - 1))
        
        dx = float(gx_f - gx)
        dy = float(gy_f - gy)
        
        cls_target[gy, gx, 0] = 0.0
        cls_target[gy, gx, cid + 1] = 1.0
        
        box_target[gy, gx, 0] = dx
        box_target[gy, gx, 1] = dy
        box_target[gy, gx, 2] = bw
        box_target[gy, gx, 3] = bh
        box_target[gy, gx, 4] = 1.0
        
    return cls_target, box_target

class AugmentedSequence(tf.keras.utils.Sequence):
    def __init__(self, raw_samples, batch_size=16, img_size=160, grid_size=20, augment=True):
        self.samples = raw_samples
        self.batch_size = batch_size
        self.img_size = img_size
        self.grid_size = grid_size
        self.augment = augment
        self.indices = np.arange(len(self.samples))
        
    def __len__(self):
        return int(np.ceil(len(self.samples) / float(self.batch_size)))
        
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        Y_cls = []
        Y_box = []
        
        for i in batch_idx:
            img, boxes = self.samples[i]
            if self.augment:
                img_aug, boxes_aug = apply_augmentation(img, boxes)
            else:
                img_aug, boxes_aug = img, boxes
                
            resized = cv2.resize(img_aug, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            norm = resized.astype(np.float32) / 255.0
            
            c_tar, b_tar = encode_targets(boxes_aug, self.grid_size)
            
            X.append(norm)
            Y_cls.append(c_tar)
            Y_box.append(b_tar)
            
        return np.array(X, dtype=np.float32), {"cls_output": np.array(Y_cls, dtype=np.float32), "box_output": np.array(Y_box, dtype=np.float32)}

def export_dual_head_tflite(model, rep_images, output_path):
    print(f"[*] Quantizing to Full Integer INT8 -> {output_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def rep_gen():
        for img in rep_images:
            inp = np.expand_dims(img, axis=0).astype(np.float32)
            yield [inp]
            
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[✔] INT8 Model successfully saved to: {output_path} ({len(tflite_model) / 1024:.2f} KB)")

def load_data_with_negatives(val_split_file, data_dir, num_negatives=24):
    with open(val_split_file, "r") as f:
        val_img_paths = set(line.strip() for line in f if line.strip())
        
    img_files = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
    train_samples = []
    val_samples = []
    labels_dir = os.path.join(data_dir, "labels")
    
    harvestable_crops = []
    
    for img_p in img_files:
        img = cv2.imread(img_p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stem = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(labels_dir, f"{stem}.txt")
        boxes = []
        raw_boxes_pixel = []
        
        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        xmin = cx - w / 2.0
                        ymin = cy - h / 2.0
                        xmax = cx + w / 2.0
                        ymax = cy + h / 2.0
                        boxes.append((cid, xmin, ymin, xmax, ymax))
                        
                        # pixel coords for negative crop detection
                        raw_boxes_pixel.append((xmin * ORIG_W, ymin * ORIG_H, xmax * ORIG_W, ymax * ORIG_H))
                        
        if img_p in val_img_paths:
            val_samples.append((img_rgb, boxes))
        else:
            train_samples.append((img_rgb, boxes))
            
            # Find non-overlapping background crop from this train image
            ih, iw = img_rgb.shape[:2]
            crop_w, crop_h = 440, 440
            for _ in range(20):
                rx1 = random.randint(0, max(1, iw - crop_w))
                ry1 = random.randint(0, max(1, ih - crop_h))
                rx2 = rx1 + crop_w
                ry2 = ry1 + crop_h
                
                # Check overlap with any GT box
                overlap = False
                for bx1, by1, bx2, by2 in raw_boxes_pixel:
                    if max(rx1, bx1) < min(rx2, bx2) and max(ry1, by1) < min(ry2, by2):
                        overlap = True
                        break
                if not overlap:
                    neg_patch = img_rgb[ry1:ry2, rx1:rx2].copy()
                    harvestable_crops.append(neg_patch)
                    break

    # Add negative samples to train
    random.shuffle(harvestable_crops)
    selected_negs = harvestable_crops[:num_negatives]
    print(f"[*] Harvested {len(harvestable_crops)} pure negative background patches; Selected {len(selected_negs)} for training.")
    
    for neg_img in selected_negs:
        # Negative sample has ZERO boxes: []
        train_samples.append((neg_img, []))
        
    return train_samples, val_samples

def train():
    data_dir = "/home/bim/Projects/EAI-2026/warmup/data/train"
    val_split_file = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
    save_dir = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_eiou_neg_160"
    os.makedirs(save_dir, exist_ok=True)
    
    print("[*] Loading Raw Samples with Negative Sampling (160x160 EIoU)...")
    train_samples, val_samples = load_data_with_negatives(val_split_file, data_dir, num_negatives=24)
    print(f"[✔] Total Train Samples: {len(train_samples)} (including 24 negative backgrounds), Val: {len(val_samples)}.")
    
    train_gen = AugmentedSequence(train_samples, batch_size=16, img_size=IMG_SIZE, grid_size=GRID_SIZE, augment=True)
    val_gen = AugmentedSequence(val_samples, batch_size=16, img_size=IMG_SIZE, grid_size=GRID_SIZE, augment=False)
    
    # Resume from baseline EIoU weights
    baseline_keras = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_eiou_160/model_best.keras"
    model = build_model(pretrained_path=baseline_keras, img_size=IMG_SIZE)
    total_p = model.count_params()
    print(f"[✔] Model Parameter Count: {total_p:,} (Strictly < 200,000 constraint: {'PASS' if total_p < 200000 else 'FAIL'})")
    assert total_p < 200000, "Violated microcontroller constraint!"
    
    focal = focal_loss(gamma=2.0, alpha=0.25)
    eiou_l = eiou_loss()
    
    best_keras_path = os.path.join(save_dir, "model_best.keras")
    tflite_path = os.path.join(save_dir, "mobilenet_bbox_eiou_neg_160_int8.tflite")
    
    ckpt = callbacks.ModelCheckpoint(
        filepath=best_keras_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    # Cosine decay fine-tuning schedule starting from 1e-4
    epochs = 35
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=epochs * len(train_gen),
        alpha=0.05
    )
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss={"cls_output": focal, "box_output": eiou_l},
        loss_weights={"cls_output": 1.2, "box_output": 3.0} # Slightly increased cls weight to punish false positives
    )
    
    print(f"\n--- TRAINING WITH NEGATIVE SAMPLING AT 160x160 ({epochs} epochs) ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[ckpt],
        verbose=1
    )
    
    print("\n[✔] Training completed! Exporting best checkpoint to INT8 TFLite...")
    best_model = tf.keras.models.load_model(best_keras_path, compile=False)
    
    rep_X = [cv2.resize(s[0], (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0 for s in val_samples]
    export_dual_head_tflite(best_model, rep_X, tflite_path)

if __name__ == "__main__":
    train()
