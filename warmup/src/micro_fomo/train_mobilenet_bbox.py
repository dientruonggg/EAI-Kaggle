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

ORIG_W = 1672.0
ORIG_H = 941.0
NUM_CLASSES = 10
GRID_SIZE = 16

def load_split_bbox_data(split_file, labels_dir, img_size=(128, 128)):
    grid_size = img_size[0] // 8  # 16
    with open(split_file, "r") as f:
        img_paths = [l.strip() for l in f if l.strip()]
        
    X = []
    Y_cls = []
    Y_box = []
    
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        label_file = os.path.join(labels_dir, f"{base}.txt")
        
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        
        target_cls = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
        target_cls[:, :, 0] = 1.0  # Background
        
        target_box = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        gx = min(grid_size - 1, max(0, int(cx * grid_size)))
                        gy = min(grid_size - 1, max(0, int(cy * grid_size)))
                        
                        target_cls[gy, gx, 0] = 0.0
                        target_cls[gy, gx, cls_id + 1] = 1.0
                        target_box[gy, gx] = [cx, cy, w, h]
                        
        X.append(img_norm)
        Y_cls.append(target_cls)
        Y_box.append(target_box)
        
    return np.array(X, dtype=np.float32), np.array(Y_cls, dtype=np.float32), np.array(Y_box, dtype=np.float32)

def build_mobilenet_bbox(img_size=128, alpha=0.50):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet"
    )
    
    feat_stride8 = base.get_layer("block_5_add").output  # 16x16x16
    feat_stride16 = base.get_layer("block_12_add").output # 8x8x32
    
    up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="fpn_upsample")(feat_stride16)
    fused = layers.Concatenate(name="fpn_fused")([feat_stride8, up])
    
    shared = layers.Conv2D(32, 3, padding="same", activation="relu", name="shared_refine")(fused)
    
    # Dual heads
    cls_out = layers.Conv2D(NUM_CLASSES + 1, 1, padding="same", activation="softmax", name="cls_output")(shared)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(shared)
    
    model = models.Model(inputs=base.inputs, outputs={"cls_output": cls_out, "box_output": box_out}, name="MobileNetV2_BBox")
    return model

def make_losses():
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        is_bg = y_true[:, :, :, 0:1]
        is_fg = 1.0 - is_bg
        weights = is_bg * 0.2 + is_fg * 3.0
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal = tf.pow(1.0 - pt, 2.0)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1, keepdims=True)
        return tf.reduce_mean(weights * focal * ce)
        
    def bbox_loss(y_true, y_pred):
        # Mask on cells where ground truth box is non-zero
        is_obj = tf.reduce_max(y_true, axis=-1, keepdims=True) > 0.0
        is_obj = tf.cast(is_obj, tf.float32)
        diff = tf.abs(y_true - y_pred)
        smooth_l1 = tf.where(diff < 0.05, 0.5 * tf.square(diff) / 0.05, diff - 0.025)
        loss = tf.reduce_sum(smooth_l1 * is_obj) / (tf.reduce_sum(is_obj) + 1e-6)
        return loss
        
    return focal_loss, bbox_loss

def export_dual_head_tflite(model, rep_data, out_path):
    def rep_gen():
        for sample in rep_data:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant = converter.convert()
    
    size_kb = len(tflite_quant) / 1024.0
    print(f"[✔] Dual-head INT8 model exported: {size_kb:.2f} KB (Limit: 1500 KB)")
    with open(out_path, "wb") as f:
        f.write(tflite_quant)
    print(f"[✔] Saved to: {out_path}")
    return tflite_quant

def train_and_export():
    train_split = "/home/bim/Projects/EAI-2026/warmup/data/train_split.txt"
    val_split = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
    labels_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
    save_dir = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_128"
    os.makedirs(save_dir, exist_ok=True)
    
    print("[*] Loading Dual-Head Data (Classification + Box coordinates)...")
    X_train, Y_cls_train, Y_box_train = load_split_bbox_data(train_split, labels_dir, img_size=(128, 128))
    X_val, Y_cls_val, Y_box_val = load_split_bbox_data(val_split, labels_dir, img_size=(128, 128))
    
    model = build_mobilenet_bbox(img_size=128, alpha=0.50)
    print(f"[*] Total Model Parameters: {model.count_params():,}")
    
    focal, bbox_l = make_losses()
    
    # Phase 1: Warmup heads (15 epochs)
    print("\n--- PHASE 1: WARMUP DUAL HEADS (15 epochs) ---")
    for layer in model.layers:
        if "output" not in layer.name and "shared" not in layer.name and "fpn" not in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
            
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={"cls_output": focal, "box_output": bbox_l},
        loss_weights={"cls_output": 1.0, "box_output": 5.0}
    )
    
    model.fit(
        X_train, {"cls_output": Y_cls_train, "box_output": Y_box_train},
        validation_data=(X_val, {"cls_output": Y_cls_val, "box_output": Y_box_val}),
        epochs=15,
        batch_size=16,
        verbose=1
    )
    
    # Phase 2: Fine-tuning backbone (35 epochs)
    print("\n--- PHASE 2: FINE-TUNING BACKBONE (35 epochs) ---")
    for layer in model.layers:
        if "block_0" in layer.name or "block_1" in layer.name or "block_2" in layer.name or "Conv1" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
            
    best_path = os.path.join(save_dir, "model_best.keras")
    tflite_path = os.path.join(save_dir, "mobilenet_bbox_128_int8.tflite")
    
    cb = [
        callbacks.ModelCheckpoint(best_path, save_best_only=True, monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=2e-4),
        loss={"cls_output": focal, "box_output": bbox_l},
        loss_weights={"cls_output": 1.0, "box_output": 5.0}
    )
    
    model.fit(
        X_train, {"cls_output": Y_cls_train, "box_output": Y_box_train},
        validation_data=(X_val, {"cls_output": Y_cls_val, "box_output": Y_box_val}),
        epochs=35,
        batch_size=16,
        callbacks=cb,
        verbose=1
    )
    
    print("\n[✔] Training completed! Exporting INT8 quantized TFLite...")
    best_model = tf.keras.models.load_model(best_path, custom_objects={"focal_loss": focal, "bbox_loss": bbox_l})
    export_dual_head_tflite(best_model, X_train[:40], tflite_path)

if __name__ == "__main__":
    train_and_export()
