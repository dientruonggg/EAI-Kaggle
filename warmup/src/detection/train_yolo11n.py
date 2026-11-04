#!/usr/bin/env python3
"""
TRAIN YOLO11N ON WARMUP TRAFFIC DATASET
State-of-the-art YOLO11 architecture fine-tuned on the 10 Warmup traffic classes.
"""

import os
import sys
from ultralytics import YOLO

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"

def train():
    data_yaml = os.path.join(PROJECT_ROOT, "data", "dataset.yaml")
    project = os.path.join(PROJECT_ROOT, "runs", "detect")
    name = "yolo11n_warmup"

    print("[*] Initializing YOLO11n...")
    model = YOLO("yolo11n.pt")

    print(f"[*] Training YOLO11n for 25 epochs on CPU (batch=16, imgsz=640)...")
    results = model.train(
        data=data_yaml,
        epochs=25,
        imgsz=640,
        batch=16,
        device="cpu",
        project=project,
        name=name,
        workers=4,
        cos_lr=True,
        close_mosaic=5,
        exist_ok=True,
        verbose=True
    )
    best_weights = os.path.join(project, name, "weights", "best.pt")
    print(f"[✔] Training complete! Best weights saved to: {best_weights}")
    return best_weights

if __name__ == "__main__":
    train()
