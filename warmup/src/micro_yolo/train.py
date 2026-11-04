import os
from ultralytics import YOLO

MAX_ALLOWED_PARAMS = 200000

def train_micro_yolo(
    yaml_config="/home/bim/Projects/EAI-2026/warmup/src/micro_yolo/micro_yolo.yaml",
    data_yaml="/home/bim/Projects/EAI-2026/warmup/data/dataset.yaml",
    project="/home/bim/Projects/EAI-2026/warmup/runs/detect",
    name="micro_yolo",
    epochs=25,
    imgsz=128,
    batch=16,
    workers=4
):
    print(f"[*] Initializing Micro-YOLO from config: {yaml_config}")
    model = YOLO(yaml_config)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"[*] Micro-YOLO total parameters: {total_params:,}")
    if total_params > MAX_ALLOWED_PARAMS:
        raise ValueError(f"CRITICAL: Param count {total_params} exceeds limit of {MAX_ALLOWED_PARAMS}!")

    print(f"[*] Training Micro-YOLO for {epochs} epochs at imgsz={imgsz} on CPU...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        project=project,
        name=name,
        workers=workers,
        exist_ok=True,
        verbose=True
    )
    
    best_pt = os.path.join(project, name, "weights", "best.pt")
    print(f"[✔] Micro-YOLO training complete! Saved best model to: {best_pt}")
    return best_pt

if __name__ == "__main__":
    train_micro_yolo()
