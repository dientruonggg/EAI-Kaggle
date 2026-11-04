import os
import sys
import glob
import json
import csv
import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ORIG_W = 320.0
ORIG_H = 240.0
NUM_CLASSES = 7

def nms(boxes, scores, iou_thresh=0.45):
    """Standard non-maximum suppression in numpy."""
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

def infer_test_images(
    tflite_model_path,
    test_images_dir,
    output_csv=None,
    conf_thresh=0.20,
    iou_thresh=0.45,
    max_boxes_per_image=100
):
    """
    Runs single-model inference on test images (strictly no ensemble / no TTA).
    """
    print(f"[*] Initializing TFLite Interpreter: {tflite_model_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    # Identify classification & box regression tensors
    cls_det = [d for d in output_details if d["shape"][-1] == NUM_CLASSES + 1][0]
    box_det = [d for d in output_details if d["shape"][-1] == 4][0]

    in_shape = input_details["shape"]  # e.g., [1, 160, 160, 3]
    img_h, img_w = in_shape[1], in_shape[2]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    cls_scale, cls_zero = cls_det.get("quantization", (0.0, 0))
    box_scale, box_zero = box_det.get("quantization", (0.0, 0))
    grid_size = cls_det["shape"][1]

    test_imgs = sorted(glob.glob(os.path.join(test_images_dir, "*.png")) +
                       glob.glob(os.path.join(test_images_dir, "*.jpg")))
    print(f"[*] Found {len(test_imgs)} test images in {test_images_dir}")

    results = []
    total_detections = 0

    for img_p in test_imgs:
        sample_id = os.path.splitext(os.path.basename(img_p))[0]
        img = cv2.imread(img_p)
        if img is None:
            results.append({"task_type": "DETECTION", "sample_id": sample_id, "prediction": "[]"})
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)

        # Quantize input if model is INT8
        if input_details["dtype"] == np.int8:
            norm = (resized.astype(np.float32) / 255.0)
            in_tensor = np.round(norm / in_scale + in_zero).astype(np.int8)[None, ...]
        else:
            in_tensor = (resized.astype(np.float32) / 255.0)[None, ...]

        interpreter.set_tensor(input_details["index"], in_tensor)
        interpreter.invoke()

        # Extract predictions
        raw_cls = interpreter.get_tensor(cls_det["index"])[0]
        raw_box = interpreter.get_tensor(box_det["index"])[0]

        # Dequantize if INT8
        if cls_det["dtype"] == np.int8:
            pred_cls = (raw_cls.astype(np.float32) - cls_zero) * cls_scale
        else:
            pred_cls = raw_cls.astype(np.float32)

        if box_det["dtype"] == np.int8:
            pred_box = (raw_box.astype(np.float32) - box_zero) * box_scale
        else:
            pred_box = raw_box.astype(np.float32)

        boxes = []
        scores = []
        classes = []

        for gy in range(grid_size):
            for gx in range(grid_size):
                probs = pred_cls[gy, gx]
                max_c = int(np.argmax(probs[1:]))  # 0..NUM_CLASSES-1
                score = float(probs[max_c + 1])

                if score >= conf_thresh:
                    off_x, off_y, bw, bh = pred_box[gy, gx]
                    cx_norm = (gx + off_x) / float(grid_size)
                    cy_norm = (gy + off_y) / float(grid_size)

                    # Denormalize to original image size (320 x 240)
                    xmin = max(0.0, (cx_norm - bw / 2.0) * ORIG_W)
                    ymin = max(0.0, (cy_norm - bh / 2.0) * ORIG_H)
                    xmax = min(ORIG_W, (cx_norm + bw / 2.0) * ORIG_W)
                    ymax = min(ORIG_H, (cy_norm + bh / 2.0) * ORIG_H)

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        scores.append(score)
                        classes.append(max_c)

        # NMS
        if boxes:
            keep_idx = nms(boxes, scores, iou_thresh=iou_thresh)
            # Cap at max_boxes_per_image
            keep_idx = keep_idx[:max_boxes_per_image]

            formatted_boxes = []
            for ki in keep_idx:
                b = boxes[ki]
                formatted_boxes.append({
                    "class_id": int(classes[ki]),
                    "score": round(float(scores[ki]), 4),
                    "bbox": [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)]
                })
            results.append({
                "task_type": "DETECTION",
                "sample_id": sample_id,
                "prediction": json.dumps(formatted_boxes)
            })
            total_detections += len(formatted_boxes)
        else:
            results.append({"task_type": "DETECTION", "sample_id": sample_id, "prediction": "[]"})

    print(f"[✔] Inference completed. Total detections: {total_detections} across {len(test_imgs)} images.")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["task_type", "sample_id", "prediction"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[✔] Saved detection predictions to: {output_csv}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run single-model detection inference")
    parser.add_argument("--model", default=os.path.join(PROJECT_ROOT, "weights/mobilenet_bbox_160_int8.tflite"))
    parser.add_argument("--images_dir", default=os.path.join(PROJECT_ROOT, "data/test/images"))
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "submissions/detection_preds.csv"))
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    if os.path.exists(args.model):
        infer_test_images(args.model, args.images_dir, output_csv=args.output,
                          conf_thresh=args.conf, iou_thresh=args.iou)
    else:
        print(f"[!] Model not found: {args.model}")
