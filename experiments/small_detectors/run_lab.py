"""Reproduce the offline FOMO and YOLO-style smoke experiment."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
FILE_NAME = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_128_no_top.h5"
CHECKPOINT_SHA256 = "f555b66d2d4b2d89360b92aee6f36a17278a13596cdf9917620240507c20deef"
HEIGHT, WIDTH = 96, 128
CLASSES = ("bottle", "cup", "phone", "book", "mouse", "box", "toy_car")
PREVIEW = REPO / "DuyToan/best/val_preview.png"
CROPS = {"00011": (29, 63), "00044": (1502, 63), "00066": (29, 417)}
CROP_WIDTH, CROP_HEIGHT = 449, 337
SEED = 20261120
MAX_BOXES = 100
FOMO_SCORE = 0.18
YOLO_SCORE = 0.30
YOLO_IOU = 0.50
ANCHORS = (((6, 8), (10, 12), (15, 18)), ((20, 24), (30, 36), (46, 56)))
TIMED_RUNS = 30
COMMON_OPERATORS = {"ADD", "CONV_2D", "DEPTHWISE_CONV_2D"}
EXPECTED_OPERATORS = {
    "fomo": COMMON_OPERATORS,
    "yolo": COMMON_OPERATORS | {"CONCATENATION", "RESIZE_NEAREST_NEIGHBOR"},
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_count(variables):
    return sum(int(np.prod(variable.shape)) for variable in variables)


def weight_fingerprint(variables):
    checksum = hashlib.sha256()
    for variable in variables:
        array = np.asarray(variable.numpy(), dtype="<f4")
        checksum.update(array.tobytes())
    return checksum.hexdigest()


def build_models(weights):
    layers = tf.keras.layers
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(HEIGHT, WIDTH, 3), alpha=0.35, include_top=False, weights=None
    )
    stride8 = backbone.get_layer("block_6_expand_relu").output
    stride16 = backbone.get_layer("block_12_add").output
    prefixes = {
        "fomo": tf.keras.Model(backbone.input, stride8),
        "yolo": tf.keras.Model(backbone.input, stride16),
    }
    before = {name: weight_fingerprint(model.weights) for name, model in prefixes.items()}
    backbone.load_weights(weights)
    transfer = {}
    for name, prefix in prefixes.items():
        after = weight_fingerprint(prefix.weights)
        assert after != before[name]
        transfer[name] = {
            "backbone_total_including_bn": prefix.count_params(),
            "random_backbone_sha256": before[name],
            "loaded_backbone_sha256": after,
        }

    grid = layers.Conv2D(32, 1, activation="relu", name="fomo_features")(stride8)
    centroids = layers.Conv2D(len(CLASSES) + 1, 1, name="fomo_logits")(grid)
    fomo = tf.keras.Model(backbone.input, centroids, name="fomo")

    coarse = layers.Conv2D(32, 1, activation="relu", name="neck_coarse")(stride16)
    upsampled = layers.Lambda(
        lambda image: tf.image.resize(image, (HEIGHT // 8, WIDTH // 8), method="nearest"),
        name="neck_nearest_resize",
    )(coarse)
    lateral = layers.Conv2D(32, 1, activation="relu", name="neck_lateral")(stride8)
    fine = layers.Concatenate()([lateral, upsampled])
    fine = layers.DepthwiseConv2D(3, padding="same", activation="relu")(fine)
    # Each anchor predicts tx, ty, tw, th, objectness and seven class logits.
    channels = len(ANCHORS[0]) * (5 + len(CLASSES))
    fine_logits = layers.Conv2D(channels, 1, name="yolo_stride8")(fine)
    coarse_logits = layers.Conv2D(channels, 1, name="yolo_stride16")(coarse)
    yolo = tf.keras.Model(backbone.input, [fine_logits, coarse_logits], name="yolo")

    models = {"fomo": fomo, "yolo": yolo}
    for name, model in models.items():
        prefix = prefixes[name]
        assert {id(weight) for weight in prefix.weights} <= {id(weight) for weight in model.weights}
        assert weight_fingerprint(prefix.weights) == transfer[name]["loaded_backbone_sha256"]
        transfer[name]["head_or_neck_random_parameters"] = model.count_params() - prefix.count_params()
    return models, transfer


def preprocess(rgb):
    resized = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 127.5 - 1.0)[None, ...]


def smoke_images(path):
    preview = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if preview is None or preview.shape[:2] != (770, 1980):
        raise ValueError("Expected the 1980 × 770 DuyToan validation contact sheet")
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    return {
        name: preprocess(rgb[y:y + CROP_HEIGHT, x:x + CROP_WIDTH])
        for name, (x, y) in CROPS.items()
    }


def representative_samples():
    """Synthetic-only calibration keeps preview/validation inputs out of calibration."""
    rng = np.random.default_rng(SEED)
    grid_y, grid_x = np.indices((HEIGHT, WIDTH))
    for value in (0, 64, 128, 192, 255):
        yield [np.full((1, HEIGHT, WIDTH, 3), value / 127.5 - 1.0, dtype=np.float32)]
    for index in range(16):
        random_rgb = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        ramp = ((grid_x * (index + 1) + grid_y * 2) % 256).astype(np.uint8)
        random_rgb[:, :, index % 3] = ramp
        yield [preprocess(random_rgb)]


def convert_int8(model_function, destination, representative_dataset=representative_samples):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([model_function.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    assert len(model_bytes) <= 1_500_000
    destination.write_bytes(model_bytes)
    return model_bytes


def audit_flatbuffer(model_bytes, expected_operators):
    model = schema.Model.GetRootAsModel(model_bytes, 0)
    graph = model.Subgraphs(0)
    names = {value: name for name, value in vars(schema.BuiltinOperator).items() if isinstance(value, int)}
    types = {value: name for name, value in vars(schema.TensorType).items() if isinstance(value, int)}
    operations = Counter()
    for index in range(graph.OperatorsLength()):
        opcode = model.OperatorCodes(graph.Operators(index).OpcodeIndex())
        operations[f"{names[opcode.BuiltinCode()]} v{opcode.Version()}"] += 1
    tensor_types = Counter(types[graph.Tensors(i).Type()] for i in range(graph.TensorsLength()))
    assert model.SubgraphsLength() == 1
    assert set(tensor_types) <= {"INT8", "INT32"}
    actual_operators = {name.split()[0] for name in operations}
    assert actual_operators == expected_operators, (actual_operators, expected_operators)
    return {"schema": model.Version(), "operators": dict(sorted(operations.items())), "tensor_types": dict(tensor_types)}


def describe_tensor(tensor):
    return dict(index=int(tensor["index"]), shape=tensor["shape"].tolist(),
                dtype=np.dtype(tensor["dtype"]).name, quantization=list(tensor["quantization"]))


def lite_runner(model_bytes):
    interpreter = tf.lite.Interpreter(
        model_content=model_bytes,
        num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    input_tensor, = interpreter.get_input_details()
    outputs = sorted(interpreter.get_output_details(), key=lambda item: item["shape"][1], reverse=True)
    assert tuple(input_tensor["shape"]) == (1, HEIGHT, WIDTH, 3)
    assert all(t["dtype"] == np.int8 for t in [input_tensor, *outputs])

    def run(image):
        scale, zero_point = input_tensor["quantization"]
        quantized = np.clip(np.rint(image / scale + zero_point), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_tensor["index"], quantized)
        interpreter.invoke()
        result = []
        for tensor in outputs:
            scale, zero_point = tensor["quantization"]
            data = interpreter.get_tensor(tensor["index"])
            result.append((data.astype(np.float32) - zero_point) * scale)
        return result

    return run, describe_tensor(input_tensor), [describe_tensor(tensor) for tensor in outputs]


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fomo_centroids(outputs, score_threshold=FOMO_SCORE):
    logits = outputs[0][0]
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    winning_class = probabilities.argmax(axis=-1)
    points = []
    for channel, name in enumerate(CLASSES, 1):
        scores = probabilities[:, :, channel]
        mask = ((winning_class == channel) & (scores >= score_threshold)).astype(np.uint8)
        count, _, statistics, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, count):
            x, y = (centroids[i] + 0.5) * 8
            location = ("trai", "giua", "phai")[min(int(x * 3 / WIDTH), 2)]
            points.append({"class": name, "xy": [round(x, 2), round(y, 2)], "location": location,
                           "cells": int(statistics[i, cv2.CC_STAT_AREA])})
    return points


def decode_fomo(outputs):
    points = fomo_centroids(outputs)
    return {"count": dict(Counter(p["class"] for p in points)),
            "locations": dict(Counter(p["location"] for p in points)), "sample_centroids": points[:5]}


def yolo_detections(outputs, score_threshold=YOLO_SCORE):
    boxes, scores, class_ids = [], [], []
    for logits, anchors in zip(outputs, ANCHORS, strict=True):
        grid_height, grid_width = logits.shape[1:3]
        rows = logits[0].reshape(grid_height, grid_width, len(anchors), 5 + len(CLASSES))
        class_scores = sigmoid(rows[..., 5:])
        best_class = class_scores.argmax(axis=-1)
        confidence = sigmoid(rows[..., 4]) * class_scores.max(axis=-1)
        for row, col, anchor in np.argwhere(confidence >= score_threshold):
            logits = rows[row, col, anchor]
            center = (np.array([col, row]) + sigmoid(logits[:2])) * (WIDTH / grid_width)
            size = np.asarray(anchors[anchor]) * np.exp(np.clip(logits[2:4], -4, 4))
            left, top = np.clip(center - size / 2, [0, 0], [WIDTH, HEIGHT])
            right, bottom = np.clip(center + size / 2, [0, 0], [WIDTH, HEIGHT])
            boxes.append([float(left), float(top), float(right - left), float(bottom - top)])
            scores.append(float(confidence[row, col, anchor]))
            class_ids.append(int(best_class[row, col, anchor]))
    keep = []
    for class_id in range(len(CLASSES)):
        indices = [i for i, value in enumerate(class_ids) if value == class_id]
        selected = cv2.dnn.NMSBoxes([boxes[i] for i in indices], [scores[i] for i in indices],
                                    score_threshold, YOLO_IOU)
        keep.extend(indices[int(index)] for index in selected)
    keep = sorted(keep, key=lambda i: -scores[i])[:MAX_BOXES]
    return [
        {"class_id": class_ids[i], "score": scores[i],
         "bbox": [boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]]}
        for i in keep
    ]


def decode_yolo(outputs):
    detections = yolo_detections(outputs)
    return {
        "count": len(detections),
        "sample_boxes": [
            {"class": CLASSES[item["class_id"]], "score": round(item["score"], 5),
             "xyxy": [round(value, 2) for value in item["bbox"]]}
            for item in detections[:5]
        ],
    }


def benchmark(run, image):
    for _ in range(5):
        run(image)
    times = []
    for _ in range(TIMED_RUNS):
        start = perf_counter()
        run(image)
        times.append((perf_counter() - start) * 1000)
    return {"runs": TIMED_RUNS, "p50_ms": round(float(np.median(times)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3)}


def run_experiment(weights, destination):
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(SEED)
    cv2.setNumThreads(1)
    if sha256(weights) != CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected MobileNetV2 checkpoint SHA-256: {weights}")

    destination.mkdir(parents=True, exist_ok=True)
    images = smoke_images(PREVIEW)
    models, transfer = build_models(weights)
    report = {
        "tensorflow": tf.__version__, "keras": tf.keras.version(), "numpy": np.__version__,
        "opencv": cv2.__version__, "thread_count": 1, "seed": SEED,
        "source_sha256": CHECKPOINT_SHA256, "preview_sha256": sha256(PREVIEW),
        "crop_xy": CROPS, "crop_wh": [CROP_WIDTH, CROP_HEIGHT], "input_rgb_hw": [HEIGHT, WIDTH],
        "calibration": "21 synthetic RGB tensors, no EAI preview or hidden test",
        "models": {},
    }
    for name, model in models.items():
        total = model.count_params()
        trainable = parameter_count(model.trainable_weights)
        bn_state = parameter_count(model.non_trainable_weights)
        assert total == trainable + bn_state <= 200_000
        assert all("moving_" in weight.name for weight in model.non_trainable_weights)

        model_function = tf.function(lambda image: model(image, training=False),
                                     input_signature=[tf.TensorSpec((1, HEIGHT, WIDTH, 3), tf.float32)])
        def run_float(image):
            return [np.asarray(out) for out in tf.nest.flatten(model_function(image))]

        path = destination / f"{name}_lab_int8.tflite"
        model_bytes = convert_int8(model_function, path)
        run_int8, input_tensor, output_tensors = lite_runner(model_bytes)
        decoder = decode_fomo if name == "fomo" else decode_yolo
        expected_shapes = [(1, 12, 16, 8)] if name == "fomo" else [(1, 12, 16, 36), (1, 6, 8, 36)]
        predictions = {}
        for image_id, image in images.items():
            floats = run_float(image)
            integers = run_int8(image)
            for outputs in (floats, integers):
                assert [out.shape for out in outputs] == expected_shapes
                assert all(np.isfinite(out).all() and np.ptp(out) > 0 for out in outputs)
            predictions[image_id] = {
                "float": decoder(floats), "int8": decoder(integers),
                "raw_logit_mae": [round(float(np.mean(np.abs(a - b))), 6) for a, b in zip(floats, integers, strict=True)],
            }
        image = next(iter(images.values()))
        report["models"][name] = {
            "parameters": {"trainable": trainable, "bn_running_state": bn_state, "total": total},
            "transfer": transfer[name], "smoke": predictions,
            "float": {"input_dtype": "float32", "outputs": expected_shapes, "cpu": benchmark(run_float, image)},
            "int8": {"bytes": len(model_bytes), "sha256": sha256(path), "input": input_tensor,
                     "outputs": output_tensors, "cpu": benchmark(run_int8, image),
                     **audit_flatbuffer(model_bytes, EXPECTED_OPERATORS[name])},
        }
    (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path.home() / ".keras/models" / FILE_NAME)
    parser.add_argument("--out", type=Path, default=LAB / "artifacts")
    arguments = parser.parse_args()
    run_experiment(arguments.weights, arguments.out)
