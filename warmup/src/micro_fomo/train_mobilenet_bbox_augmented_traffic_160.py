#!/usr/bin/env python3
"""
TRAIN MOBILENETV2-BBOX 160x160 ON AUGMENTED TRAFFIC DATASET
Trained on:
  - 265 traffic images (120 orig + 71 COCO traffic + 74 synthesized rare classes)
  - Evaluated on: clean 20-image val_split.txt
  - Loss: Focal Loss (cls) + CIoU Loss (bbox)
  - Target Hardware: ESP32-S3 (< 200,000 parameters, Full Integer INT8)
"""

import os
import sys
import argparse
import glob
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, models

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    
    model = models.Model(inputs=base.inputs, outputs={"cls_output": cls_out, "box_output": box_out}, name="MobileNetV2_Augmented_Traffic_160")
    
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

def ciou_loss():
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
        
        rho2 = tf.square(pred_cx - true_cx) + tf.square(pred_cy - true_cy)
        
        pi = 3.141592653589793
        v = (4.0 / (pi ** 2)) * tf.square(tf.atan(true_w / true_h) - tf.atan(pred_w / pred_h))
        alpha = v / (1.0 - iou + v + 1e-7)
        
        ciou = iou - (rho2 / c2) - (alpha * v)
        ciou_loss_val = (1.0 - ciou) * obj_mask
        
        num_pos = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
        return tf.reduce_sum(ciou_loss_val) / num_pos
    return loss

def apply_augmentation(img, boxes):
    aug_img = img.copy()
    aug_boxes = [b for b in boxes]
    
    # Random horizontal flip (p=0.5)
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
        new_boxes = []
        for cid, cx, cy, w, h in aug_boxes:
            new_boxes.append((cid, 1.0 - cx, cy, w, h))
        aug_boxes = new_boxes
        
    # Random brightness / contrast jitter
    if random.random() > 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12, 12)
        aug_img = np.clip(alpha * aug_img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        
    return aug_img, aug_boxes

def encode_targets(boxes, grid_size=GRID_SIZE):
    cls_target = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
    cls_target[..., 0] = 1.0 # default background
    box_target = np.zeros((grid_size, grid_size, 5), dtype=np.float32)
    
    # Sort large boxes first so smaller objects override center cells
    sorted_boxes = sorted(boxes, key=lambda b: b[3] * b[4], reverse=True)
    
    for cid, cx, cy, bw, bh in sorted_boxes:
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

class TrafficSequence(tf.keras.utils.Sequence):
    def __init__(self, samples, batch_size=16, img_size=160, grid_size=20, augment=True):
        self.samples = samples
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
        X, Y_cls, Y_box = [], [], []
        
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
    print(f"[*] Starting Full Integer INT8 PTQ Quantization -> {output_path}...")
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
    sz_kb = len(tflite_model) / 1024.0
    print(f"[✔] INT8 Model successfully saved to: {output_path} ({sz_kb:.2f} KB)")
    assert sz_kb <= 1536.0, "Model exceeded 1.5MB flash budget!"

def load_dataset(split_file):
    samples = []
    with open(split_file, "r") as f:
        paths = [l.strip() for l in f if l.strip()]
        
    for img_p in paths:
        img = cv2.imread(img_p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Determine label path
        # If in train_augmented_traffic/images/xxx.png -> train_augmented_traffic/labels/xxx.txt
        # If in train/images/xxx.png -> train/labels/xxx.txt
        dir_p = os.path.dirname(img_p)
        parent_p = os.path.dirname(dir_p)
        sid = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(parent_p, "labels", f"{sid}.txt")
        
        boxes = []
        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append((cid, cx, cy, w, h))
        samples.append((img_rgb, boxes))
        
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default="/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_eiou_160/model_best.keras")
    args = parser.parse_args()

    train_split = "/home/bim/Projects/EAI-2026/warmup/data/train_augmented_traffic_split.txt"
    val_split = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
    save_dir = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_traffic_160"
    os.makedirs(save_dir, exist_ok=True)

    print(f"[*] Loading training data from: {train_split}")
    train_samples = load_dataset(train_split)
    print(f"[✔] Loaded {len(train_samples)} training samples.")

    print(f"[*] Loading validation data from: {val_split}")
    val_samples = load_dataset(val_split)
    print(f"[✔] Loaded {len(val_samples)} validation samples.")

    train_seq = TrafficSequence(train_samples, batch_size=args.batch_size, img_size=IMG_SIZE, augment=True)
    val_seq = TrafficSequence(val_samples, batch_size=args.batch_size, img_size=IMG_SIZE, augment=False)

    print("[*] Building MobileNetV2-BBox (160x160, alpha=0.50)...")
    model = build_model(pretrained_path=args.resume if os.path.exists(args.resume) else None, img_size=IMG_SIZE, alpha=0.50)
    
    total_p = model.count_params()
    print(f"[✔] Model Parameter Count: {total_p:,} (< 200,000 threshold: {'PASS' if total_p < 200000 else 'FAIL'})")
    assert total_p < 200000, "Violated microcontroller constraint!"

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss={
            "cls_output": focal_loss(gamma=2.0, alpha=0.25),
            "box_output": ciou_loss()
        },
        loss_weights={
            "cls_output": 1.0,
            "box_output": 2.5
        }
    )

    best_keras_path = os.path.join(save_dir, "model_best.keras")
    ckpt = callbacks.ModelCheckpoint(best_keras_path, monitor="val_loss", save_best_only=True, verbose=1)
    lr_decay = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

    print(f"\n[*] Commencing Training for {args.epochs} epochs...")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=[ckpt, lr_decay, early_stop],
        verbose=1
    )

    print("\n[✔] Training finished! Preparing representative images for INT8 quantization...")
    rep_images = []
    for img_rgb, _ in train_samples[:120]:
        resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        rep_images.append(norm)

    tflite_out = os.path.join(save_dir, "mobilenet_bbox_augmented_traffic_160_int8.tflite")
    export_dual_head_tflite(model, rep_images, tflite_out)
    print(f"\n[🎉] COMPLETED! Trained & Quantized model saved to:\n     {tflite_out}")

if __name__ == "__main__":
    main()
