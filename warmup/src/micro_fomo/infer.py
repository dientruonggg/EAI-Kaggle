import os
import glob
import json
import csv
import cv2
import numpy as np
import tensorflow as tf

ORIG_WIDTH = 1672.0
ORIG_HEIGHT = 941.0
GRID_SIZE = 12

def nms(boxes, scores, iou_thresh=0.50):
    """Pure numpy per-class Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        union = np.maximum(union, 1e-6)
        ovr = inter / union

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

def infer_micro_fomo_tflite(
    tflite_path,
    test_images_dir,
    priors_path="/home/bim/Projects/EAI-2026/warmup/data/class_priors.json",
    conf_threshold=0.18,
    iou_thresh=0.50,
    use_nms=True
):
    print(f"[*] Initializing TFLite Interpreter from: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    in_shape = input_details["shape"]  # e.g. [1, 96, 96, 3]
    in_dtype = input_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero = output_details.get("quantization", (0.0, 0))
    
    grid_size = output_details["shape"][1]
    
    # Load class priors
    with open(priors_path, "r") as f:
        priors = json.load(f)
        
    test_images = sorted(glob.glob(os.path.join(test_images_dir, "*.png")))
    print(f"[*] Found {len(test_images)} test images in {test_images_dir} (grid_size: {grid_size})")
    print(f"[*] Inference settings: conf_threshold={conf_threshold}, iou_thresh={iou_thresh}, use_nms={use_nms}")
    
    results = []
    
    for img_p in test_images:
        sample_id = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        if img is None:
            results.append(["DETECTION", sample_id, "[]"])
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (in_shape[1], in_shape[2]), interpolation=cv2.INTER_AREA)
        
        # Preprocess according to input dtype
        if in_dtype == np.int8:
            norm = (img_resized.astype(np.float32) / 255.0)
            in_tensor = np.round(norm / in_scale + in_zero).astype(np.int8)
        else:
            in_tensor = (img_resized.astype(np.float32) / 255.0).astype(np.float32)
            
        in_tensor = np.expand_dims(in_tensor, axis=0)
        interpreter.set_tensor(input_details["index"], in_tensor)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details["index"])[0]  # shape: (grid_size, grid_size, 11)
        
        # Dequantize output if int8
        if output_details["dtype"] == np.int8:
            probs = (output_data.astype(np.float32) - out_zero) * out_scale
        else:
            probs = output_data.astype(np.float32)
            
        # Decode FOMO grid cells
        raw_boxes = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                cell_probs = probs[gy, gx, 1:11]  # 10 object classes
                best_cls = int(np.argmax(cell_probs))
                score = float(cell_probs[best_cls])
                
                if score >= conf_threshold:
                    cx = (gx + 0.5) / float(grid_size)
                    cy = (gy + 0.5) / float(grid_size)
                    
                    # Box dimensions from priors
                    cls_prior = priors.get(str(best_cls), {"w": 0.1, "h": 0.1})
                    pw = cls_prior["w"]
                    ph = cls_prior["h"]
                    
                    xmin = max(0.0, round((cx - pw / 2.0) * ORIG_WIDTH, 1))
                    ymin = max(0.0, round((cy - ph / 2.0) * ORIG_HEIGHT, 1))
                    xmax = min(ORIG_WIDTH, round((cx + pw / 2.0) * ORIG_WIDTH, 1))
                    ymax = min(ORIG_HEIGHT, round((cy + ph / 2.0) * ORIG_HEIGHT, 1))
                    
                    raw_boxes.append({
                        "class_id": best_cls,
                        "score": round(score, 4),
                        "bbox": [xmin, ymin, xmax, ymax]
                    })
                    
        # Apply Per-Class NMS
        if use_nms:
            final_boxes = []
            for c in range(10):
                c_boxes = [b for b in raw_boxes if b["class_id"] == c]
                if not c_boxes:
                    continue
                coords = [b["bbox"] for b in c_boxes]
                scores = [b["score"] for b in c_boxes]
                keep = nms(coords, scores, iou_thresh=iou_thresh)
                for k in keep:
                    final_boxes.append(c_boxes[k])
            final_boxes.sort(key=lambda b: b["score"], reverse=True)
            box_list = final_boxes[:100]
        else:
            raw_boxes.sort(key=lambda b: b["score"], reverse=True)
            box_list = raw_boxes[:100]
        
        results.append(["DETECTION", sample_id, json.dumps(box_list, separators=(',', ':'))])
        
    print(f"[✔] Micro-FOMO TFLite inference complete for {len(results)} images.")
    return results

def save_detection_csv(results, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerows(results)
    print(f"[✔] Predictions saved to: {out_csv}")

if __name__ == "__main__":
    tflite = "/home/bim/Projects/EAI-2026/warmup/weights/micro_fomo/micro_fomo_int8.tflite"
    test_dir = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
    out_csv = "/home/bim/Projects/EAI-2026/warmup/submissions/fomo_nms_preds.csv"
    
    if os.path.exists(tflite):
        rows = infer_micro_fomo_tflite(tflite, test_dir, conf_threshold=0.18, iou_thresh=0.50, use_nms=True)
        save_detection_csv(rows, out_csv)
    else:
        print("Model file does not exist yet:", tflite)
