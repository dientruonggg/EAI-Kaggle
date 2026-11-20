import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.model import build_hierarchical_12layer_detector, build_mobilenet_bbox_model, MAX_ALLOWED_PARAMS
from src.detection.dataset import QualifierDetectionDataset
from src.detection.losses import decoupled_conf_loss, decoupled_cls_loss, decoupled_box_loss

def train_qualifier_detector(
    train_images_dir=os.path.join(PROJECT_ROOT, "data/train/images"),
    train_labels_dir=os.path.join(PROJECT_ROOT, "data/train/labels"),
    val_images_dir=os.path.join(PROJECT_ROOT, "data/val/images"),
    val_labels_dir=os.path.join(PROJECT_ROOT, "data/val/labels"),
    weights_save_dir=os.path.join(PROJECT_ROOT, "weights/mobilenet_bbox_160"),
    epochs=40,
    batch_size=16,
    img_size=160,
    lr=1.5e-3,
    patience=12,
    augment_occlusion=False
):
    print("=" * 70)
    print("   EDGE AI CHALLENGE 2026 - MICRO-EDGE-CNN 12-LAYER TRAINING")
    print("   Decoupled Conf/Cls/Box Heads & Hard-Negative Error Weighting")
    print(f"   EarlyStopping Patience: {patience} epochs | Max Epochs: {epochs}")
    print(f"   Occlusion Curriculum: {'ENABLED' if augment_occlusion else 'DISABLED'}")
    print("=" * 70)

    train_imgs = sorted(glob.glob(os.path.join(train_images_dir, "*.png")) +
                        glob.glob(os.path.join(train_images_dir, "*.jpg")))
    val_imgs = sorted(glob.glob(os.path.join(val_images_dir, "*.png")) +
                      glob.glob(os.path.join(val_images_dir, "*.jpg")))

    print(f"[*] Found {len(train_imgs)} train images and {len(val_imgs)} val images.")
    if len(train_imgs) == 0:
        print("[!] Warning: No train images found in data directory.")
        return None

    grid_size = img_size // 8
    train_ds = QualifierDetectionDataset(train_imgs, train_labels_dir, img_size=img_size,
                                         grid_size=grid_size, batch_size=batch_size,
                                         augment=True, augment_occlusion=augment_occlusion)
    val_ds = QualifierDetectionDataset(val_imgs, val_labels_dir, img_size=img_size,
                                       grid_size=grid_size, batch_size=batch_size,
                                       augment=False, augment_occlusion=False) if val_imgs else None

    model = build_hierarchical_12layer_detector(input_shape=(img_size, img_size, 3), num_classes=7)
    
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss={
            "conf_output": decoupled_conf_loss,
            "cls_output": decoupled_cls_loss,
            "box_output": decoupled_box_loss
        },
        loss_weights={
            "conf_output": 1.5,
            "cls_output": 1.5,
            "box_output": 2.5
        }
    )

    os.makedirs(weights_save_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_save_dir, "best_weights.keras")
    
    cbs = [
        callbacks.ModelCheckpoint(best_weights_path, monitor="val_loss" if val_ds else "loss",
                                  save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss" if val_ds else "loss", factor=0.5,
                                    patience=max(3, patience // 2), min_lr=1e-5, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss" if val_ds else "loss", patience=patience,
                                min_delta=1e-3, restore_best_weights=True, verbose=1)
    ]

    print(f"[*] Training detector for up to {epochs} epochs (will auto-terminate on convergence)...")
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
    parser = argparse.ArgumentParser(description="Train Qualifier Detector with EarlyStopping")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--augment_occlusion", action="store_true", help="Enable tabletop occlusion curriculum")
    args = parser.parse_args()

    train_qualifier_detector(epochs=args.epochs, batch_size=args.batch_size,
                             img_size=args.img_size, lr=args.lr, patience=args.patience,
                             augment_occlusion=args.augment_occlusion)
