import os
from ultralytics import YOLO

def train_detector(
    data_yaml="/home/bim/Projects/EAI-2026/warmup/data/dataset.yaml",
    model_name="yolov8n.pt",
    epochs=12,
    imgsz=640,
    batch=16,
    project="/home/bim/Projects/EAI-2026/warmup/runs/detect",
    name="yolov8n_warmup",
    workers=4
):
    print(f"[*] Initializing YOLO model: {model_name}")
    model = YOLO(model_name)

    print(f"[*] Training for {epochs} epochs on CPU...")
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
    best_weights = os.path.join(project, name, "weights", "best.pt")
    print(f"[✔] Training complete! Best weights saved to: {best_weights}")
    return best_weights

if __name__ == "__main__":
    train_detector()
