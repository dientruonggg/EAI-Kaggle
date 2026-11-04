import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, models

from src.micro_fomo.model import export_to_tflite_int8

ORIG_W = 1672.0
ORIG_H = 941.0
NUM_CLASSES = 10
CLASS_NAMES = [
    "person", "bicycle", "motorcycle", "car", "bus",
    "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"
]

def load_split_data(split_file, labels_dir, img_size=(128, 128)):
    grid_size = img_size[0] // 8  # 16
    with open(split_file, "r") as f:
        img_paths = [l.strip() for l in f if l.strip()]
        
    X = []
    Y = []
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        label_file = os.path.join(labels_dir, f"{base}.txt")
        
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        
        target = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
        target[:, :, 0] = 1.0  # Background
        
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy = float(parts[1]), float(parts[2])
                        gx = min(grid_size - 1, max(0, int(cx * grid_size)))
                        gy = min(grid_size - 1, max(0, int(cy * grid_size)))
                        target[gy, gx, 0] = 0.0
                        target[gy, gx, cls_id + 1] = 1.0
                        
        X.append(img_norm)
        Y.append(target)
        
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def build_fomo_mobilenetv2_128(img_size=128, alpha=0.50):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet"
    )
    
    # Stride 8: 128 -> 16x16x16
    feat_stride8 = base.get_layer("block_5_add").output
    # Stride 16: 128 -> 8x8x32
    feat_stride16 = base.get_layer("block_12_add").output
    
    up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="fpn_upsample")(feat_stride16)
    fused = layers.Concatenate(name="fpn_fused")([feat_stride8, up])
    
    # Refinement Conv
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="fomo_refine")(fused)
    x = layers.BatchNormalization(name="fomo_refine_bn")(x)
    x = layers.ReLU(6.0, name="fomo_refine_relu")(x)
    
    # Final classification head: 16x16x11
    out = layers.Conv2D(NUM_CLASSES + 1, 1, padding="same", activation="softmax", name="fomo_output")(x)
    
    model = models.Model(inputs=base.inputs, outputs=out, name=f"MobileNetV2_{alpha}_FOMO_128")
    return model

def focal_loss(gamma=2.0, bg_weight=0.2, fg_weight=3.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        is_bg = y_true[:, :, :, 0:1]
        is_fg = 1.0 - is_bg
        weights = is_bg * bg_weight + is_fg * fg_weight
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal = tf.pow(1.0 - pt, gamma)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1, keepdims=True)
        return tf.reduce_mean(weights * focal * ce)
    return loss

def train_and_export():
    train_split = "/home/bim/Projects/EAI-2026/warmup/data/train_split.txt"
    val_split = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
    labels_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
    save_dir = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_fomo_128"
    os.makedirs(save_dir, exist_ok=True)
    
    print("[*] Loading train & val data for 128x128...")
    X_train, Y_train = load_split_data(train_split, labels_dir, img_size=(128, 128))
    X_val, Y_val = load_split_data(val_split, labels_dir, img_size=(128, 128))
    print(f"[*] X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"[*] X_val:   {X_val.shape}, Y_val:   {Y_val.shape}")
    
    model = build_fomo_mobilenetv2_128(img_size=128, alpha=0.50)
    print(f"[*] Total model parameters: {model.count_params():,}")
    assert model.count_params() <= 200000, "Must be <= 200,000 parameters!"
    
    loss_fn = focal_loss(gamma=2.0, bg_weight=0.2, fg_weight=3.0)
    
    # Phase 1: Warmup head (freeze base)
    print("\n--- PHASE 1: WARMUP FOMO HEAD (15 epochs) ---")
    for layer in model.layers:
        if "fomo" not in layer.name and "fpn" not in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
            
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=loss_fn, metrics=["accuracy"])
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=15,
        batch_size=16,
        verbose=1
    )
    
    # Phase 2: Fine-tune backbone (unfreeze from block 3 onwards)
    print("\n--- PHASE 2: FINE-TUNING FROM BLOCK 3 (35 epochs) ---")
    for layer in model.layers:
        if "block_0" in layer.name or "block_1" in layer.name or "block_2" in layer.name or "Conv1" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
            
    best_weights_path = os.path.join(save_dir, "model_best.keras")
    tflite_path = os.path.join(save_dir, "mobilenet_fomo_128_int8.tflite")
    
    cb_list = [
        callbacks.ModelCheckpoint(best_weights_path, save_best_only=True, monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    model.compile(optimizer=optimizers.Adam(learning_rate=2e-4), loss=loss_fn, metrics=["accuracy"])
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=35,
        batch_size=16,
        callbacks=cb_list,
        verbose=1
    )
    
    print("\n[✔] Training completed! Exporting INT8 quantized TFLite...")
    best_model = tf.keras.models.load_model(best_weights_path, custom_objects={"loss": loss_fn})
    export_to_tflite_int8(best_model, X_train[:40], tflite_path)
    print(f"[✔] INT8 Model exported to: {tflite_path}")

if __name__ == "__main__":
    train_and_export()
