import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.model import build_mobilenet_bbox_model, MAX_ALLOWED_PARAMS
from src.detection.dataset import QualifierDetectionDataset
from src.detection.losses import focal_loss

def train_qualifier_detector(
    train_images_dir=os.path.join(PROJECT_ROOT, "data/train/images"),
    train_labels_dir=os.path.join(PROJECT_ROOT, "data/train/labels"),
    val_images_dir=os.path.join(PROJECT_ROOT, "data/val/images"),
    val_labels_dir=os.path.join(PROJECT_ROOT, "data/val/labels"),
    weights_save_dir=os.path.join(PROJECT_ROOT, "weights/mobilenet_bbox_160"),
    epochs=50,
    batch_size=16,
    img_size=160,
    alpha=0.35,
    lr=1e-3
):
    print("=" * 70)
    print("   EDGE AI CHALLENGE 2026 - QUALIFIERS DETECTOR TRAINING")
    print("   Constraint: Single Model, < 200k params, TFLite INT8 ready")
    print("=" * 70)

    train_imgs = sorted(glob.glob(os.path.join(train_images_dir, "*.png")) +
                        glob.glob(os.path.join(train_images_dir, "*.jpg")))
    val_imgs = sorted(glob.glob(os.path.join(val_images_dir, "*.png")) +
                      glob.glob(os.path.join(val_images_dir, "*.jpg")))

    print(f"[*] Found {len(train_imgs)} train images and {len(val_imgs)} val images.")
    if len(train_imgs) == 0:
        print("[!] Warning: No train images found in data directory yet.")
        print("[*] Building model architecture and verifying parameter invariants...")
        model = build_mobilenet_bbox_model(input_shape=(img_size, img_size, 3), alpha=alpha)
        print(f"[✔] Model is valid: {model.count_params():,} params (< {MAX_ALLOWED_PARAMS:,})")
        return model

    grid_size = img_size // 8
    train_ds = QualifierDetectionDataset(train_imgs, train_labels_dir, img_size=img_size,
                                         grid_size=grid_size, batch_size=batch_size, augment=True)
    val_ds = QualifierDetectionDataset(val_imgs, val_labels_dir, img_size=img_size,
                                       grid_size=grid_size, batch_size=batch_size, augment=False) if val_imgs else None

    model = build_mobilenet_bbox_model(input_shape=(img_size, img_size, 3), alpha=alpha)
    
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss={
            "cls_output": lambda y_t, y_p: focal_loss(y_t, y_p, gamma=2.0, alpha=0.25),
            "box_output": "mse"
        },
        loss_weights={
            "cls_output": 1.0,
            "box_output": 5.0
        }
    )

    os.makedirs(weights_save_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_save_dir, "best_weights.keras")
    
    cbs = [
        callbacks.ModelCheckpoint(best_weights_path, monitor="val_loss" if val_ds else "loss",
                                  save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss" if val_ds else "loss", factor=0.5,
                                    patience=5, min_lr=1e-5, verbose=1)
    ]

    print(f"[*] Training detector for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs,
        verbose=1
    )

    print(f"[✔] Training finished. Best weights saved to: {best_weights_path}")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Qualifier Detector")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_qualifier_detector(epochs=args.epochs, batch_size=args.batch_size,
                             img_size=args.img_size, lr=args.lr)

