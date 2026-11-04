import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from src.micro_fomo.model import create_pretrained_micro_fomo, export_to_tflite_int8
from src.micro_fomo.dataset import load_fomo_dataset

def weighted_fomo_loss(alpha=0.25, gamma=2.0, fg_weight=15.0):
    """
    Weighted focal loss for FOMO to overcome background imbalance (27:1 ratio).
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Background mask (channel 0)
        is_bg = y_true[:, :, :, 0:1]
        is_fg = 1.0 - is_bg
        
        # Focal weights
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        # Cross-entropy
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1, keepdims=True)
        
        # Apply class weights: 1.0 for bg, fg_weight for objects
        class_weight = is_bg * 1.0 + is_fg * fg_weight
        
        total_loss = ce * focal_weight * class_weight
        return tf.reduce_mean(total_loss)
    return loss

def train_micro_fomo(
    images_dir="/home/bim/Projects/EAI-2026/warmup/data/train/images",
    labels_dir="/home/bim/Projects/EAI-2026/warmup/data/train/labels",
    save_dir="/home/bim/Projects/EAI-2026/warmup/weights/micro_fomo",
    epochs=50,
    batch_size=16,
    lr=0.002
):
    os.makedirs(save_dir, exist_ok=True)
    print("[*] Loading dataset for Pretrained Micro-FOMO...")
    X, Y = load_fomo_dataset(images_dir, labels_dir, img_size=(96, 96))
    
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    print(f"[*] Train set: {len(X_train)} images, Val set: {len(X_val)} images")

    loss_fn = weighted_fomo_loss(fg_weight=15.0)
    model = create_pretrained_micro_fomo(input_shape=(96, 96, 3), num_classes=10)
    optimizer = optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    keras_weights_path = os.path.join(save_dir, "micro_fomo.keras")
    tflite_path = os.path.join(save_dir, "micro_fomo_int8.tflite")
    
    cb_list = [
        callbacks.ModelCheckpoint(keras_weights_path, save_best_only=True, monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1)
    ]
    
    print(f"[*] Training Pretrained Micro-FOMO for {epochs} epochs...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )
    
    print("[✔] Training complete! Exporting INT8 TFLite model...")
    best_model = tf.keras.models.load_model(keras_weights_path, custom_objects={"loss": loss_fn})
    export_to_tflite_int8(best_model, X_train[:50], tflite_path)
    
    return best_model, tflite_path

if __name__ == "__main__":
    train_micro_fomo()
