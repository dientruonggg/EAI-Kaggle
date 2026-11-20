#!/usr/bin/env python3
"""
ADVERSARIAL STRESS TEST HARNESS: GAP 3 (NMS & Score Calibration) & R5 (Hardware & Quantization)
Empirical verification suite executed by teamwork_challenger_2.
"""
import os
import sys
import json
import csv
import glob
import tempfile
import numpy as np
import tensorflow as tf

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.infer import per_class_nms, nms, parse_conf_thresh, infer_test_images
from src.detection.model import build_hierarchical_12layer_detector, MAX_ALLOWED_PARAMS, MAX_ALLOWED_MODEL_SIZE_KB
from submit import validate_submission

PASS = "[✔] PASS"
FAIL = "[✘] FAIL"

def run_test_1_multiclass_stacked_overlap():
    print("\n" + "=" * 75)
    print("TEST 1: Synthetic Scenes with Heavy Multi-Class Overlap (50 boxes, 7 classes)")
    print("=" * 75)
    
    # 50 boxes stacked at identical coordinates [50.0, 50.0, 150.0, 150.0]
    base_box = [50.0, 50.0, 150.0, 150.0]
    boxes = []
    scores = []
    classes = []
    
    # Generate 50 boxes distributed among 7 classes
    # e.g., class 0 gets 8 boxes, classes 1..6 get 7 boxes each = 50 boxes
    expected_top_scores = {}
    box_count = 0
    for c in range(7):
        n_c = 8 if c == 0 else 7
        c_scores = []
        for j in range(n_c):
            # vary scores from 0.30 to 0.95
            sc = round(0.30 + (j + 1) * 0.08 + (c * 0.01), 4)
            boxes.append(list(base_box))
            scores.append(sc)
            classes.append(c)
            c_scores.append(sc)
            box_count += 1
        expected_top_scores[c] = max(c_scores)

    assert box_count == 50, f"Expected 50 boxes, got {box_count}"
    print(f"[*] Generated {box_count} stacked candidate boxes across 7 classes at identical IoU=1.0")

    # 1. Test legacy class-agnostic NMS
    legacy_keep = nms(boxes, scores, iou_thresh=0.45)
    print(f"[*] Legacy class-agnostic NMS kept: {len(legacy_keep)} box(es) (Suppressed {box_count - len(legacy_keep)} boxes)")
    assert len(legacy_keep) == 1, f"Legacy NMS should suppress to 1 box, got {len(legacy_keep)}"

    # 2. Test per-class NMS
    per_class_results = per_class_nms(boxes, scores, classes, num_classes=7, iou_thresh=0.45)
    print(f"[*] Per-class NMS kept: {len(per_class_results)} box(es)")
    
    # Assertions
    assert len(per_class_results) == 7, f"Expected exactly 7 boxes (1 per class), got {len(per_class_results)}"
    
    kept_classes = [r["class_id"] for r in per_class_results]
    assert sorted(kept_classes) == list(range(7)), f"Expected all classes 0..6 represented, got {kept_classes}"
    
    for r in per_class_results:
        cid = r["class_id"]
        sc = r["score"]
        expected_sc = expected_top_scores[cid]
        assert abs(sc - expected_sc) < 1e-4, f"Class {cid} expected top score {expected_sc}, got {sc}"
        print(f"    - Class {cid}: top score {sc:.4f} retained (matches expected {expected_sc:.4f})")

    # 3. Adversarial jittered boxes: small perturbations (IoU ~ 0.85)
    jittered_boxes = []
    for i, b in enumerate(boxes):
        jit = (i % 5) * 1.0
        jittered_boxes.append([b[0] + jit, b[1] + jit, b[2] + jit, b[3] + jit])
    
    jitter_results = per_class_nms(jittered_boxes, scores, classes, num_classes=7, iou_thresh=0.45)
    assert len(jitter_results) == 7, f"Jittered heavy overlap expected 7 boxes, got {len(jitter_results)}"
    print(f"[*] Jittered heavy overlap (IoU ~ 0.85): kept exactly 7 distinct classes.")

    print(f"{PASS} Test 1: Per-Class NMS preserves distinct classes under heavy stacking without killing co-located objects.")
    return True

def run_test_2_100_boxes_hard_ceiling():
    print("\n" + "=" * 75)
    print("TEST 2: 100 Boxes Per Image Hard Ceiling Adversarial Stress Test")
    print("=" * 75)

    # Generate 500 non-overlapping boxes in a 20x25 grid over 320x240 image space
    # Grid cell size: ~15x9 pixels, each box 8x5 pixels -> zero overlap (IoU = 0)
    boxes_500 = []
    scores_500 = []
    classes_500 = []
    count = 0
    for r in range(20):
        for c in range(25):
            x1 = c * 12.0 + 1.0
            y1 = r * 11.0 + 1.0
            x2 = x1 + 8.0
            y2 = y1 + 6.0
            boxes_500.append([x1, y1, x2, y2])
            scores_500.append(round(0.50 + (count % 50) * 0.009, 4))
            classes_500.append(count % 7)
            count += 1
            if count >= 500:
                break
        if count >= 500:
            break

    assert len(boxes_500) == 500, f"Expected 500 boxes, got {len(boxes_500)}"
    print(f"[*] Generated 500 non-overlapping candidates passing threshold across grid.")

    # Scenario A: Default ceiling (max_boxes_per_image=100)
    res_default = per_class_nms(boxes_500, scores_500, classes_500, max_boxes_per_image=100)
    print(f"[*] Scenario A (max_boxes_per_image=100): returned {len(res_default)} boxes")
    assert len(res_default) == 100, f"Expected 100 boxes, got {len(res_default)}"

    # Scenario B: Adversarial caller passes max_boxes_per_image=500
    res_adv = per_class_nms(boxes_500, scores_500, classes_500, max_boxes_per_image=500)
    print(f"[*] Scenario B (Adversarial max_boxes_per_image=500): returned {len(res_adv)} boxes")
    assert len(res_adv) == 100, f"Adversarial ceiling test failed: got {len(res_adv)} boxes > 100!"

    # Scenario C: Adversarial caller passes max_boxes_per_image=None
    res_none = per_class_nms(boxes_500, scores_500, classes_500, max_boxes_per_image=None)
    print(f"[*] Scenario C (max_boxes_per_image=None): returned {len(res_none)} boxes")
    assert len(res_none) == 100, f"Expected 100 boxes on None, got {len(res_none)}"

    # Scenario D: Verify sorting by score descending
    scores_retained = [r["score"] for r in res_default]
    assert scores_retained == sorted(scores_retained, reverse=True), "Retained boxes not strictly sorted descending!"
    print(f"[*] Top score: {scores_retained[0]:.4f}, 100th score: {scores_retained[-1]:.4f} (Strictly descending)")

    print(f"{PASS} Test 2: 100 boxes hard ceiling is strictly invariant against caller configuration.")
    return True

def run_test_3_sweetspot_inference_60_images():
    print("\n" + "=" * 75)
    print("TEST 3: Sweet Spot Density Gate Inference on 60 Test Images (2,600 <= N <= 2,800)")
    print("=" * 75)

    test_images_dir = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
    model_path = "/home/bim/Projects/EAI-2026/qualifiers/weights/mobilenet_bbox_160_int8.tflite"
    out_csv = "/home/bim/Projects/EAI-2026/qualifiers/submissions/stress_sweetspot_60.csv"

    test_files = sorted(glob.glob(os.path.join(test_images_dir, "*.*")))
    print(f"[*] Detected {len(test_files)} test images in {test_images_dir}")
    assert len(test_files) == 60, f"Expected exactly 60 test images, got {len(test_files)}"
    assert os.path.exists(model_path), f"INT8 model not found at {model_path}"

    # Run inference with target_sweetspot=True
    predictions = infer_test_images(
        model_path=model_path,
        test_images_dir=test_images_dir,
        output_csv=out_csv,
        conf_thresh=0.12,
        target_sweetspot=True
    )

    total_detections = sum(len(json.loads(p["prediction"])) for p in predictions)
    avg_boxes = total_detections / len(predictions)
    print(f"[*] Total detections across 60 images: {total_detections} (Avg: {avg_boxes:.2f} boxes/img)")

    assert 2600 <= total_detections <= 2800, (
        f"DENSITY GATE VIOLATION: Total detections {total_detections} outside [2600, 2800]!"
    )
    print(f"[★] DENSITY GATE SATISFIED: 2,600 <= {total_detections} <= 2,800 (Target sweet spot: 43.3 - 46.7 boxes/img)")

    # Verify per-image bounds
    for p in predictions:
        boxes = json.loads(p["prediction"])
        assert 1 <= len(boxes) <= 100, f"Image {p['sample_id']} has {len(boxes)} boxes outside [1, 100]"

    print(f"{PASS} Test 3: 60-image inference strictly produces {total_detections} boxes in sweet spot.")
    return out_csv, total_detections

def run_test_4_submission_validator_compliance(submission_csv):
    print("\n" + "=" * 75)
    print("TEST 4: Submission Serialization & Zero-Warning Validator Compliance")
    print("=" * 75)

    assert os.path.exists(submission_csv), f"Submission CSV missing: {submission_csv}"
    
    # 1. Run official validator
    print(f"[*] Validating generated submission via submit.validate_submission...")
    is_valid = validate_submission(submission_csv, expected_det_count=60)
    assert is_valid is True, "validate_submission returned False or failed"
    print(f"[*] submit.py validate_submission returned PASSED cleanly.")

    # 2. Adversarial Validator Stress Tests
    print(f"[*] Running adversarial negative checks against validator...")
    
    # Case A: Degenerate box (xmax <= xmin)
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        bad_csv = f.name
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerow(["DETECTION", "bad_1", json.dumps([{"class_id": 0, "score": 0.8, "bbox": [100.0, 50.0, 50.0, 150.0]}])])
    try:
        validate_submission(bad_csv)
        assert False, "Validator failed to catch degenerate box xmax <= xmin!"
    except ValueError as e:
        print(f"    - [✔] Correctly rejected degenerate box: {e}")
    finally:
        os.remove(bad_csv)

    # Case B: Exceeding 100 boxes
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        bad_csv = f.name
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        boxes_101 = [{"class_id": 0, "score": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]} for _ in range(101)]
        writer.writerow(["DETECTION", "bad_2", json.dumps(boxes_101)])
    try:
        validate_submission(bad_csv)
        assert False, "Validator failed to catch > 100 boxes!"
    except ValueError as e:
        print(f"    - [✔] Correctly rejected >100 boxes per image: {e}")
    finally:
        os.remove(bad_csv)

    # Case C: Invalid class_id (class_id=7)
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        bad_csv = f.name
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerow(["DETECTION", "bad_3", json.dumps([{"class_id": 7, "score": 0.8, "bbox": [10.0, 10.0, 50.0, 50.0]}])])
    try:
        validate_submission(bad_csv)
        assert False, "Validator failed to catch invalid class_id=7!"
    except ValueError as e:
        print(f"    - [✔] Correctly rejected out-of-range class_id: {e}")
    finally:
        os.remove(bad_csv)

    print(f"{PASS} Test 4: Submission strictly conforms to official validator with 0 warnings/errors.")
    return True

def run_test_5_hardware_parameter_limit():
    print("\n" + "=" * 75)
    print("TEST 5: Microcontroller Target Parameter Limit Audit (< 200,000 parameters)")
    print("=" * 75)

    model = build_hierarchical_12layer_detector(input_shape=(160, 160, 3), num_classes=7)
    total_params = model.count_params()
    trainable_params = sum(np.prod(v.shape) for v in model.trainable_weights)
    non_trainable_params = sum(np.prod(v.shape) for v in model.non_trainable_weights)

    print(f"[*] MicroEdgeCNN Total Parameters: {total_params:,}")
    print(f"    - Trainable Parameters:     {trainable_params:,}")
    print(f"    - Non-Trainable Parameters: {non_trainable_params:,}")
    print(f"[*] Competition Limit:          {MAX_ALLOWED_PARAMS:,}")
    print(f"[*] Margin under budget:        {MAX_ALLOWED_PARAMS - total_params:,} ({(1.0 - total_params/MAX_ALLOWED_PARAMS)*100:.2f}% headroom)")

    assert total_params < MAX_ALLOWED_PARAMS, f"Parameter limit violated: {total_params} >= {MAX_ALLOWED_PARAMS}"
    assert total_params == 50228, f"Unexpected parameter count: {total_params} (expected 50,228)"

    # Verify all model weights are finite
    for w in model.get_weights():
        assert np.all(np.isfinite(w)), "Found non-finite (NaN or Inf) weights in model!"
    print(f"[*] Verified: 100% of network weights are finite and initialized.")

    print(f"{PASS} Test 5: Model parameter limit strictly satisfied with 74.89% headroom.")
    return True

def run_test_6_quantization_and_tflm_operators():
    print("\n" + "=" * 75)
    print("TEST 6: Full Integer INT8 PTQ & TensorFlow Lite Micro Compatibility Audit")
    print("=" * 75)

    tflite_path = "/home/bim/Projects/EAI-2026/qualifiers/weights/mobilenet_bbox_160_int8.tflite"
    assert os.path.exists(tflite_path), f"Quantized model missing at {tflite_path}"

    # 1. File size check <= 1.5 MB
    file_size_bytes = os.path.getsize(tflite_path)
    file_size_kb = file_size_bytes / 1024.0
    print(f"[*] Quantized INT8 model file size: {file_size_kb:.2f} KB ({file_size_bytes:,} bytes)")
    print(f"[*] Hardware Flash Limit:           {MAX_ALLOWED_MODEL_SIZE_KB} KB (1.5 MB)")
    print(f"[*] Flash utilization:              {(file_size_kb / MAX_ALLOWED_MODEL_SIZE_KB) * 100:.2f}%")

    assert file_size_kb <= MAX_ALLOWED_MODEL_SIZE_KB, (
        f"Flash budget exceeded: {file_size_kb:.2f} KB > {MAX_ALLOWED_MODEL_SIZE_KB} KB!"
    )

    # 2. Tensor types & shape check
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[*] Input Tensor Details:")
    for inp in input_details:
        print(f"    - Name: '{inp['name']}', Shape: {inp['shape'].tolist()}, Dtype: {inp['dtype']}")
        assert inp["dtype"] == np.int8, f"Input tensor {inp['name']} is NOT np.int8 (got {inp['dtype']})!"

    print(f"[*] Output Tensor Details ({len(output_details)} heads):")
    output_names = []
    for out in output_details:
        print(f"    - Name: '{out['name']}', Shape: {out['shape'].tolist()}, Dtype: {out['dtype']}")
        assert out["dtype"] == np.int8, f"Output tensor {out['name']} is NOT np.int8 (got {out['dtype']})!"
        output_names.append(out["name"])

    assert len(output_details) == 3, f"Expected 3 output heads (conf, cls, box), got {len(output_details)}"

    # 3. TFLM Operator Compatibility Audit
    TFLM_STANDARD_WHITELIST = {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "ADD",
        "LOGISTIC",
        "SOFTMAX",
        "CONCATENATION",
        "RESHAPE",
        "STRIDED_SLICE",
        "EXPAND_DIMS",
        "SHAPE",
        "TILE",
        "RESIZE_NEAREST_NEIGHBOR",
        "PAD"
    }

    # Extract true flatbuffer operator codes
    with open(tflite_path, "rb") as f:
        buf = f.read()
    
    from tensorflow.lite.python import schema_py_generated as schema_fb
    fb_model = schema_fb.Model.GetRootAsModel(buf, 0)
    flatbuffer_ops = []
    for i in range(fb_model.OperatorCodesLength()):
        op_code = fb_model.OperatorCodes(i)
        code = op_code.DeprecatedBuiltinCode()
        if code == schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
            code = op_code.BuiltinCode()
        name = [k for k, v in schema_fb.BuiltinOperator.__dict__.items() if v == code]
        flatbuffer_ops.append(name[0] if name else str(code))

    print(f"[*] True FlatBuffer compiled operators ({len(flatbuffer_ops)}): {flatbuffer_ops}")
    unsupported_ops = set(flatbuffer_ops) - TFLM_STANDARD_WHITELIST
    assert len(unsupported_ops) == 0, (
        f"UNSUPPORTED TFLM OPERATORS DETECTED: {unsupported_ops}! Disqualification risk!"
    )
    print(f"[✔] 100% of operators are within standard TensorFlow Lite Micro whitelist.")

    # 4. Live INT8 Inference Execution
    dummy_input = np.zeros((1, 160, 160, 3), dtype=np.int8)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    
    for out in output_details:
        out_data = interpreter.get_tensor(out["index"])
        assert out_data.dtype == np.int8, f"Live output dtype is not np.int8!"
        assert np.all(np.isfinite(out_data)), f"Live output contains non-finite values!"
    print(f"[*] Live INT8 inference execution on synthetic input completed successfully.")

    print(f"{PASS} Test 6: INT8 PTQ model, size <= 1.5MB, int8 in/out, and 100% TFLM ops verified.")
    return True

def main():
    print("\n" + "#" * 75)
    print("STARTING EMPIRICAL ADVERSARIAL STRESS TEST SUITE: GAP 3 & R5")
    print("#" * 75)

    test_1 = run_test_1_multiclass_stacked_overlap()
    test_2 = run_test_2_100_boxes_hard_ceiling()
    sub_csv, total_det = run_test_3_sweetspot_inference_60_images()
    test_4 = run_test_4_submission_validator_compliance(sub_csv)
    test_5 = run_test_5_hardware_parameter_limit()
    test_6 = run_test_6_quantization_and_tflm_operators()

    all_passed = test_1 and test_2 and (total_det >= 2600 and total_det <= 2800) and test_4 and test_5 and test_6
    
    print("\n" + "#" * 75)
    print("ALL 6 EMPIRICAL STRESS TESTS COMPLETED:")
    print(f"  1. Multi-class stacked overlap (50 boxes, 7 classes):   {PASS}")
    print(f"  2. 100-boxes per image hard ceiling:                   {PASS}")
    print(f"  3. Density sweetspot 60-image inference (N={total_det}):    {PASS}")
    print(f"  4. Submission serialization & validator conformance:   {PASS}")
    print(f"  5. Hardware parameter limit (< 200,000 params):        {PASS}")
    print(f"  6. INT8 PTQ, size <= 1.5MB, int8 in/out & TFLM ops:    {PASS}")
    print(f"OVERALL EMPIRICAL VERDICT: {'APPROVE' if all_passed else 'REQUEST_CHANGES'}")
    print("#" * 75 + "\n")
    
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
