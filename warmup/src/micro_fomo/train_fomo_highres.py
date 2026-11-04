import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import glob
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

from src.micro_fomo.model import create_pretrained_micro_fomo, export_to_tflite_int8
from src.micro_fomo.train import weighted_fomo_loss

def train_fomo_resolution(img_size=128, epochs=50, lr=0.001):
    grid_size = img_size // 8
    print(f"\n==========================================")
    print(f"[*] Training Pretrained Micro-FOMO at {img_size}x{img_size} (Grid: {grid_size}x{grid_size})")

    # Load Train and Val splits
    train_split_file = "data/train_split.txt"
    val_split_file = "data/val_split.txt"
    labels_dir = "data/train/labels"
    save_dir = f"weights/micro_fomo_{img_size}"
    os.makedirs(save_dir, exist_ok=True)

    with open(train_split_file) as f:
        train_paths = [l.strip() for l in f if l.strip()]
    with open(val_split_file) as f:
        val_paths = [l.strip() for l in f if l.strip()]

    def load_split(paths):
        X = []
        Y = []
        for p in paths:
            sid = os.path.splitext(os.path.basename(p))[0]
            lbl = os.path.join(labels_dir, f"{sid}.txt")
            img = cv2.imread(p)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
            norm = img_resized.astype(np.float32) / 255.0

            target = np.zeros((grid_size, grid_size, 11), dtype=np.float32)
            target[:, :, 0] = 1.0

            if os.path.exists(lbl):
                with open(lbl) as fp:
                    for line in fp:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            cx, cy = float(parts[1]), float(parts[2])
                            gx = min(grid_size - 1, max(0, int(cx * grid_size)))
                            gy = min(grid_size - 1, max(0, int(cy * grid_size)))
                            target[gy, gx, 0] = 0.0
                            target[gy, gx, cid + 1] = 1.0

            X.append(norm)
            Y.append(target)
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    X_train, Y_train = load_split(train_paths)
    X_val, Y_val = load_split(val_paths)
    print(f"[*] Train set: {len(X_train)} images, Val set: {len(X_val)} images")

    # Data Augmentation on train set: add horizontally flipped copies
    X_aug = []
    Y_aug = []
    for i in range(len(X_train)):
        img = X_train[i]
        tgt = Y_train[i]
        # Original
        X_aug.append(img)
        Y_aug.append(tgt)
        # Flip horizontal
        img_flip = cv2.flip(img, 1)
        tgt_flip = np.flip(tgt, axis=1)  # flip grid columns
        X_aug.append(img_flip)
        Y_aug.append(tgt_flip)

    X_train = np.array(X_aug, dtype=np.float32)
    Y_train = np.array(Y_aug, dtype=np.float32)
    print(f"[✔] Augmented Train set size: {len(X_train)} images")

    model = create_pretrained_micro_fomo(input_shape=(img_size, img_size, 3), num_classes=10)
    loss_fn = weighted_fomo_loss(fg_weight=15.0)
    optimizer = optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    keras_path = os.path.join(save_dir, "model_best.keras")
    tflite_path = os.path.join(save_dir, f"micro_fomo_{img_size}_int8.tflite")

    cb_list = [
        callbacks.ModelCheckpoint(keras_path, save_best_only=True, monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    ]

    print(f"[*] Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=cb_list,
        verbose=1
    )

    # Load best model for INT8 export
    best_model = tf.keras.models.load_model(keras_path, custom_objects={"loss": loss_fn})
    export_to_tflite_int8(best_model, X_train[:50], tflite_path)

    size_kb = os.path.getsize(tflite_path) / 1024.0
    print(f"[✔] Successfully exported {tflite_path}: {size_kb:.2f} KB (Limit: 1,500 KB)")
    return tflite_path

if __name__ == "__main__":
    train_fomo_resolution(img_size=128, epochs=40, lr=0.001)
