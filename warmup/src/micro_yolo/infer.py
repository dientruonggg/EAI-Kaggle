import os
import glob
import json
import csv
from ultralytics import YOLO

ORIG_WIDTH = 1672.0
ORIG_HEIGHT = 941.0
MAX_ALLOWED_MODEL_SIZE_KB = 1500  # 1.5 MB

def infer_and_export_micro_yolo(
    weights_path="/home/bim/Projects/EAI-2026/warmup/runs/detect/micro_yolo/weights/best.pt",
    test_dir="/home/bim/Projects/EAI-2026/warmup/data/test/images",
    output_csv="/home/bim/Projects/EAI-2026/warmup/submissions/micro_yolo_preds.csv",
    conf_thresh=0.20,
    export_tflite=True
):
    print(f"[*] Loading Micro-YOLO model from: {weights_path}")
    model = YOLO(weights_path)
    
    # 1. Export to TFLite INT8 if requested
    if export_tflite:
        try:
            print("[*] Exporting Micro-YOLO to TFLite (int8=True, imgsz=128)...")
            export_res = model.export(
                format="tflite",
                int8=True,
                imgsz=128,
                data="/home/bim/Projects/EAI-2026/warmup/data/dataset.yaml"
            )
            print(f"[✔] Export result: {export_res}")
            if os.path.exists(export_res):
                size_kb = os.path.getsize(export_res) / 1024.0
                print(f"[✔] Micro-YOLO TFLite INT8 size: {size_kb:.2f} KB (Limit: {MAX_ALLOWED_MODEL_SIZE_KB} KB)")
                if size_kb > MAX_ALLOWED_MODEL_SIZE_KB:
                    raise ValueError(f"CRITICAL: Size {size_kb} KB exceeds {MAX_ALLOWED_MODEL_SIZE_KB} KB limit!")
        except Exception as e:
            print(f"[!] Note on TFLite export: {e}")

    # 2. Predict on 60 test images
    image_paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    print(f"[*] Inferring {len(image_paths)} test images...")
    
    results_data = []
    for p in image_paths:
        sample_id = os.path.splitext(os.path.basename(p))[0]
        preds = model.predict(source=p, imgsz=128, conf=conf_thresh, verbose=False)
        box_list = []
        
        if preds and len(preds) > 0:
            boxes = preds[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                
                for i in range(min(len(boxes), 100)):
                    c_id = int(clss[i])
                    score = round(float(confs[i]), 4)
                    coords = [round(float(c), 1) for c in xyxy[i]]
                    box_list.append({
                        "class_id": c_id,
                        "score": score,
                        "bbox": coords
                    })
                    
        results_data.append({
            "task_type": "DETECTION",
            "sample_id": sample_id,
            "prediction": json.dumps(box_list, separators=(',', ':'))
        })
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "sample_id", "prediction"])
        writer.writeheader()
        writer.writerows(results_data)
        
    print(f"[✔] Saved {len(results_data)} predictions to: {output_csv}")
    return results_data

if __name__ == "__main__":
    infer_and_export_micro_yolo()
