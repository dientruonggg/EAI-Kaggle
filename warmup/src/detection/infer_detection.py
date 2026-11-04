import os
import glob
import json
import pandas as pd
from ultralytics import YOLO

def infer_test_images(model_weights_path, test_images_dir, conf_threshold=0.25, max_boxes=100):
    print(f"[*] Loading YOLO model from: {model_weights_path}")
    model = YOLO(model_weights_path)

    image_paths = sorted(glob.glob(os.path.join(test_images_dir, "*.png")))
    print(f"[*] Found {len(image_paths)} test images in {test_images_dir}")

    results_data = []

    for img_path in image_paths:
        sample_id = os.path.splitext(os.path.basename(img_path))[0]
        
        preds = model.predict(source=img_path, conf=conf_threshold, verbose=False)
        box_list = []

        if preds and len(preds) > 0:
            boxes = preds[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()

                for i in range(min(len(boxes), max_boxes)):
                    c_id = int(clss[i])
                    score = round(float(confs[i]), 4)
                    coords = [round(float(coord), 1) for coord in xyxy[i]]
                    box_list.append({
                        "class_id": c_id,
                        "score": score,
                        "bbox": coords
                    })

        pred_json_str = json.dumps(box_list, separators=(',', ':'))
        results_data.append({
            "task_type": "DETECTION",
            "sample_id": sample_id,
            "prediction": pred_json_str
        })

    df = pd.DataFrame(results_data)
    print(f"[✔] Inferred {len(df)} images.")
    return df

if __name__ == "__main__":
    weights = "/home/bim/Projects/EAI-2026/warmup/runs/detect/yolov8n_warmup/weights/best.pt"
    test_dir = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
    out_csv = "/home/bim/Projects/EAI-2026/warmup/submissions/detection_preds.csv"

    if os.path.exists(weights):
        df = infer_test_images(weights, test_dir)
        df.to_csv(out_csv, index=False)
        print(f"[✔] Saved to {out_csv}")
    else:
        print(f"Weights not found yet at {weights}")
