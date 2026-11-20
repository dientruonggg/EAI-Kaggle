"""Train the fixed EAI CPU pilot and evaluate once on the coordinator's held-out split."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np
import tensorflow as tf

from run_lab import (
    ANCHORS, CHECKPOINT_SHA256, CLASSES, EXPECTED_OPERATORS, FILE_NAME, FOMO_SCORE,
    HEIGHT, LAB, REPO, SEED, WIDTH, audit_flatbuffer, build_models, convert_int8,
    fomo_centroids, lite_runner, preprocess, sha256, weight_fingerprint, yolo_detections,
)
from prepare_coco import COCO128_URL, COCO_TO_EAI, IMAGE_ROOT, LABELS_URL

EPOCHS = 4
COCO_EPOCHS = 4
BATCH = 8
LEARNING_RATE = 0.002
ORIG_WIDTH, ORIG_HEIGHT = 320, 240
CALIBRATION_COUNT = 24
FOMO_POSITIVE_WEIGHT = 12.0
YOLO_BOX_LOSS_WEIGHT = 4.0
MAP_SCORE_FLOOR = 0.02
COUNT_SCORE = 0.30
CENTROID_RADIUS = 8.0
SOURCE = "egde-ai-challenge-2026-lan-2-qualifiers"
COCO_S3_ROOT = "https://s3.amazonaws.com/images.cocodataset.org/train2017/"
COCO_COORDINATOR_IMAGES = 120
COCO_LARGE_PER_CLASS = 1600
COCO_LARGE_IMAGES = COCO_LARGE_PER_CLASS * len(COCO_TO_EAI)
COCO_REGIONAL_S3_ROOT = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/"


def verify_split(dataset):
    manifest = dataset / "sources.json"
    sources = json.loads(manifest.read_bytes())
    if sources.get("failures") != []:
        raise ValueError("Dataset not ready: sources.json failures must be an empty list")
    if (sources["seed"], sources["source"]) != (SEED, SOURCE):
        raise ValueError("Unexpected source or seed")
    paths, image_hashes, split_hashes = {}, {}, {}
    for split, expected in (("train", 96), ("val", 24)):
        pairs = {}
        images = set()
        checksum = hashlib.sha256()
        for source_image, source_label in sources[split]:
            image_name, label_name = Path(source_image).name, Path(source_label).name
            image_path = dataset / split / "images" / image_name
            label_path = dataset / split / "labels" / label_name
            if (source_image != f"train/images/{image_name}" or source_label != f"train/labels/{label_name}"
                    or image_path.stem != label_path.stem or image_path.stem in pairs):
                raise ValueError(f"Unexpected source pair: {source_image}, {source_label}")
            pairs[image_path.stem] = (image_path, label_path)
            image_hash = sha256(image_path)
            images.add(image_hash)
            checksum.update(f"{image_name} {image_hash} {label_name} {sha256(label_path)}\n".encode())
        if len(pairs) != expected:
            raise ValueError(f"Unexpected {split} size: {len(pairs)}")
        paths[split], image_hashes[split], split_hashes[split] = pairs, images, checksum.hexdigest()
    if paths["train"].keys() & paths["val"].keys() or image_hashes["train"] & image_hashes["val"]:
        raise ValueError("Train/val overlap in IDs or image SHA-256")
    return paths, {"manifest_sha256": sha256(manifest), "split_sha256": split_hashes,
                   "source": SOURCE, "seed": SEED, "train": 96, "val": 24}


def validate_coco_quota(manifest):
    mapping = {str(coco_id): eai_id for coco_id, eai_id in COCO_TO_EAI.items()}
    if (manifest.get("failures", []) != []
            or (manifest.get("source"), manifest["seed"], manifest["mapping"], manifest.get("labels_url")) !=
            ("COCO train2017", SEED, mapping, LABELS_URL)
    ):
        raise ValueError("Unexpected COCO subset provenance")
    items = manifest["images"]
    quota = Counter(item["quota_class"] for item in items)
    per_class = manifest["per_class"]
    archive = manifest.get("image_archive")
    large = per_class == COCO_LARGE_PER_CLASS
    if (set(quota) != set(COCO_TO_EAI.values()) or not (1 <= per_class <= 32 or large)
            or any(not 1 <= n <= per_class for n in quota.values())):
        raise ValueError(f"Unexpected per-class COCO quotas: {quota}")
    if archive is None and quota != Counter({cls: per_class for cls in COCO_TO_EAI.values()}):
        raise ValueError(f"Network subset is missing quota images: {quota}")
    if archive is not None and archive["url"] != COCO128_URL:
        raise ValueError("Unknown fallback archive source")
    if large and (archive is not None or len(items) != COCO_LARGE_IMAGES
                  or tuple(manifest.get("classes", ())) != CLASSES[:len(mapping)]):
        raise ValueError("The 8k selection must cover five COCO classes with 1,600 images each")
    seen = set()
    for item in items:
        stem = item["id"]
        if (len(stem) != 12 or not stem.isascii() or not stem.isdigit() or stem in seen
                or item["url"] != f"{IMAGE_ROOT}{stem}.jpg"):
            raise ValueError(f"Invalid COCO image ID or URL: {stem}")
        if large and item.get("download_url") != f"{COCO_REGIONAL_S3_ROOT}{stem}.jpg":
            raise ValueError(f"Unexpected COCO regional S3 URL: {stem}")
        seen.add(stem)
    return quota


def verify_coco(dataset):
    manifest_path = dataset / "sources.json"
    if not manifest_path.is_file():
        raise ValueError(f"COCO subset not ready: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_bytes())
    mapping = {str(coco_id): eai_id for coco_id, eai_id in COCO_TO_EAI.items()}
    if "label_archive_url" in manifest:
        return verify_coordinator_coco(dataset, manifest_path, manifest, mapping)
    quota = validate_coco_quota(manifest)
    large = manifest["per_class"] == COCO_LARGE_PER_CLASS
    pairs = {}
    for item in manifest["images"]:
        stem = item["id"]
        image, label = dataset / "images" / f"{stem}.jpg", dataset / "labels" / f"{stem}.txt"
        if large and (not isinstance(item.get("image_bytes"), int) or item["image_bytes"] <= 0
                      or image.stat().st_size != item["image_bytes"] or not item.get("image_sha256")):
            raise ValueError(f"Missing or inconsistent finalized JPEG size/hash: {stem}")
        labels = label.read_text().splitlines()
        if (sha256(image), sha256(label)) != (item["image_sha256"], item["label_sha256"]) or labels != item["labels"]:
            raise ValueError(f"COCO image/label changed since acquisition: {stem}")
        classes = {cls for cls, *_ in parse_yolo_labels(labels, label)}
        if item["quota_class"] not in classes or not classes <= set(COCO_TO_EAI.values()):
            raise ValueError(f"Unexpected mapped COCO classes in {label}")
        pairs[stem] = image, label
    return pairs, {
        "manifest_sha256": sha256(manifest_path), "labels_sha256": manifest["labels_sha256"],
        "images": len(pairs), "quota": dict(quota), "epochs": COCO_EPOCHS, "mapping": mapping,
        "image_archive": manifest.get("image_archive"),
    }


def verify_coordinator_coco(dataset, manifest_path, manifest, mapping):
    items = manifest["images"]
    if (manifest.get("label_archive_url") != LABELS_URL or manifest.get("seed") != SEED
            or not manifest.get("source", "").startswith("COCO train2017,")
            or manifest.get("five_only") is not True or manifest.get("mapping") != mapping
            or tuple(manifest.get("class_names", ())) != CLASSES or manifest.get("failures", []) != []
            or len(items) != COCO_COORDINATOR_IMAGES):
        raise ValueError("Coordinator COCO manifest must describe 120 complete mapped train2017 pairs")
    quota = Counter(item["selected_for"] for item in items)
    if quota != Counter({name: 24 for name in CLASSES[:len(mapping)]}):
        raise ValueError(f"Unexpected coordinator COCO quotas: {quota}")
    labels_zip = dataset.parent / "coco2017labels.zip"
    if sha256(labels_zip) != manifest["label_archive_sha256"]:
        raise ValueError("COCO labels archive checksum differs from source manifest")
    pairs, counts = {}, Counter()
    with ZipFile(labels_zip) as archive:
        for item in items:
            stem = item["id"]
            if (len(stem) != 12 or not stem.isascii() or not stem.isdigit() or stem in pairs
                    or item["url"] != f"{COCO_S3_ROOT}{stem}.jpg"):
                raise ValueError(f"Invalid COCO image ID or S3 URL: {stem}")
            image = dataset / "train" / "images" / f"{stem}.jpg"
            label = dataset / "train" / "labels" / f"{stem}.txt"
            if (sha256(image), sha256(label)) != (item["image_sha256"], item["label_sha256"]):
                raise ValueError(f"COCO image or label changed since acquisition: {stem}")
            lines = archive.read(f"coco/labels/train2017/{stem}.txt").decode("ascii").splitlines()
            remapped = []
            for line in lines:
                fields = line.split()
                if (len(fields) == 5 and fields[0] in mapping
                        and all(0 <= float(coordinate) <= 1 for coordinate in fields[1:])):
                    remapped.append(" ".join([str(mapping[fields[0]]), *fields[1:]]))
            if remapped != label.read_text().splitlines():
                raise ValueError(f"COCO mapping differs from official labels: {stem}")
            classes = [row.split()[0] for row in remapped]
            if str(CLASSES.index(item["selected_for"])) not in classes:
                raise ValueError(f"COCO quota label missing in {label}")
            counts.update(classes)
            pairs[stem] = image, label
    if counts != Counter(manifest["class_counts"]):
        raise ValueError(f"COCO mapped label counts differ from source manifest: {counts}")
    return pairs, {
        "manifest_sha256": sha256(manifest_path), "labels_sha256": manifest["label_archive_sha256"],
        "images": len(pairs), "quota": dict(quota), "epochs": COCO_EPOCHS, "mapping": mapping,
        "image_source": COCO_S3_ROOT, "class_counts": dict(counts),
    }


def parse_yolo_labels(lines, label_path):
    boxes = []
    for line in lines:
        if not line.strip():
            continue
        class_text, *values = line.split()
        cls = int(class_text)
        cx, cy, width, height = map(float, values)
        if not (0 <= cls < len(CLASSES) and 0 <= cx <= 1 and 0 <= cy <= 1
                and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid label in {label_path}: {line}")
        boxes.append((cls, cx, cy, width, height))
    return boxes


def read_split(pairs, require_eai_size=True):
    images, labels = {}, {}
    for stem, (image_path, label_path) in pairs.items():
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None or (require_eai_size and bgr.shape != (ORIG_HEIGHT, ORIG_WIDTH, 3)):
            raise ValueError(f"Unreadable image or wrong EAI source size: {image_path}")
        images[stem] = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        labels[stem] = parse_yolo_labels(label_path.read_text().splitlines(), label_path)
    return images, labels


def fomo_targets(labels):
    rows, cols = HEIGHT // 8, WIDTH // 8
    target = np.zeros((len(labels), rows, cols), dtype=np.int32)
    areas = np.full(target.shape, np.inf)
    collisions = 0
    for image_index, boxes in enumerate(labels.values()):
        for cls, cx, cy, width, height in boxes:
            index = image_index, min(int(cy * rows), rows - 1), min(int(cx * cols), cols - 1)
            collisions += int(target[index] != 0)
            area = width * height
            if area < areas[index]:
                target[index], areas[index] = cls + 1, area
    return target, collisions


def yolo_targets(labels):
    grids = [(HEIGHT // stride, WIDTH // stride) for stride in (8, 16)]
    anchors = np.asarray(ANCHORS).reshape(-1, 2)
    targets = [np.zeros((len(labels), rows, cols, len(ANCHORS[0]), 5 + len(CLASSES)),
                        dtype=np.float32) for rows, cols in grids]
    quality = [np.zeros(target.shape[:-1]) for target in targets]
    collisions = 0
    for image_index, boxes in enumerate(labels.values()):
        for cls, cx, cy, width, height in boxes:
            size = np.array([width * WIDTH, height * HEIGHT])
            intersection = np.minimum(size, anchors).prod(axis=1)
            iou = intersection / (size.prod() + anchors.prod(axis=1) - intersection)
            index = int(np.argmax(iou))
            level, anchor = divmod(index, len(ANCHORS[0]))
            rows, cols = grids[level]
            x, y = min(cx * cols, cols - 1e-6), min(cy * rows, rows - 1e-6)
            row, col = int(y), int(x)
            slot = image_index, row, col, anchor
            collisions += int(quality[level][slot] > 0)
            if iou[index] > quality[level][slot]:
                target = targets[level][slot]
                target[:] = 0
                target[:4] = [x - col, y - row, *np.log(size / anchors[index])]
                target[4] = 1
                target[5 + cls] = 1
                quality[level][slot] = iou[index]
    return tuple(targets), collisions


def fomo_loss(logits, target, coco_mask=False):
    if coco_mask:
        # Background plus the five mapped classes. The two missing channels get zero gradient.
        logits = logits[..., :1 + len(COCO_TO_EAI)]
    cell_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
    weights = tf.where(target > 0, FOMO_POSITIVE_WEIGHT, 1.0)
    return tf.reduce_sum(cell_loss * weights) / tf.reduce_sum(weights)


def yolo_loss(outputs, targets, coco_mask=False):
    total = []
    for output, target in zip(outputs, targets, strict=True):
        logits = tf.reshape(output, tf.shape(target))
        mask = target[..., 4]
        positives = tf.maximum(tf.reduce_sum(mask), 1.0)
        negatives = tf.maximum(tf.reduce_sum(1.0 - mask), 1.0)
        objectness = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=logits[..., 4])
        object_loss = tf.reduce_sum(objectness * mask) / positives
        object_loss += tf.reduce_sum(objectness * (1.0 - mask)) / negatives
        end = 5 + len(COCO_TO_EAI) if coco_mask else 5 + len(CLASSES)
        classes = tf.nn.sigmoid_cross_entropy_with_logits(labels=target[..., 5:end], logits=logits[..., 5:end])
        class_loss = tf.reduce_sum(tf.reduce_mean(classes, axis=-1) * mask) / positives
        offset = tf.concat([tf.sigmoid(logits[..., :2]), logits[..., 2:4]], axis=-1)
        error = tf.abs(offset - target[..., :4])
        smooth_l1 = tf.where(error < 1.0, 0.5 * tf.square(error), error - 0.5)
        box_loss = tf.reduce_sum(tf.reduce_mean(smooth_l1, axis=-1) * mask) / positives
        total.append(object_loss + class_loss + YOLO_BOX_LOSS_WEIGHT * box_loss)
    return tf.reduce_mean(total)


def encoded_targets(name, labels):
    if name == "fomo":
        target, collisions = fomo_targets(labels)
        return (target,), collisions
    return yolo_targets(labels)


def read_coco_batch(pairs, indices, name):
    selected = dict(pairs[int(index)] for index in indices)
    images, labels = read_split(selected, require_eai_size=False)
    targets, collisions = encoded_targets(name, labels)
    return np.concatenate(list(images.values())), targets, sum(map(len, labels.values())), collisions


def fit_model(name, model, images, labels, transfer, coco_mask=False, stream=False):
    cutoff = "block_6_expand_relu" if name == "fomo" else "block_12_add"
    prefix = tf.keras.Model(model.input, model.get_layer(cutoff).output)
    for layer in prefix.layers:
        layer.trainable = False
    model_variables = model.trainable_weights
    head_before = weight_fingerprint(model_variables)
    assert sum(np.prod(weight.shape) for weight in model_variables) == transfer["head_or_neck_random_parameters"]
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    optimizer.build(model_variables)
    if name == "fomo":
        loss = lambda output, target: fomo_loss(output, target[0], coco_mask)
    else:
        loss = lambda output, target: yolo_loss(output, target, coco_mask)
    if stream:
        if not coco_mask:
            raise ValueError("File-backed batches are for the COCO stage")
        paths = tuple(images.items())
    else:
        targets, collisions = encoded_targets(name, labels)
        object_count = sum(map(len, labels.values()))

    missing_columns = () if not coco_mask else tuple(
        (model.get_layer(layer_name), columns)
        for layer_name, columns in (
            [("fomo_logits", [6, 7])] if name == "fomo" else
            [(layer_name, [anchor * (5 + len(CLASSES)) + 5 + c
                           for anchor in range(len(ANCHORS[0])) for c in (5, 6)])
             for layer_name in ("yolo_stride8", "yolo_stride16")]
        )
    )
    before_missing = [[weight[..., indices].copy() for weight in layer.get_weights()]
                      for layer, indices in missing_columns]

    @tf.function
    def train_step(batch_images, batch_targets):
        with tf.GradientTape() as tape:
            output = model(batch_images, training=False)
            value = loss(output, batch_targets)
        gradients = tape.gradient(value, model_variables)
        optimizer.apply_gradients(zip(gradients, model_variables, strict=True))
        return value

    rng = np.random.default_rng(SEED)
    losses = []
    start = perf_counter()
    epochs = COCO_EPOCHS if coco_mask else EPOCHS
    for epoch in range(epochs):
        order = rng.permutation(len(images))
        values = []
        if stream:
            object_count = collisions = 0
        for offset in range(0, len(images), BATCH):
            indices = order[offset:offset + BATCH]
            if stream:
                batch_images, batch_targets, batch_objects, batch_collisions = read_coco_batch(paths, indices, name)
                object_count += batch_objects
                collisions += batch_collisions
            else:
                batch_images = images[indices]
                batch_targets = tuple(target[indices] for target in targets)
            values.append(float(train_step(batch_images, batch_targets)))
        average = float(np.mean(values))
        if not np.isfinite(average):
            raise ValueError(f"{name}: non-finite loss in epoch {epoch + 1}")
        losses.append(average)
        print(f"{'COCO' if coco_mask else 'EAI'} {name} epoch {epoch + 1}/{epochs}: train_loss={average:.5f}", flush=True)
    head_after = weight_fingerprint(model_variables)
    assert weight_fingerprint(prefix.weights) == transfer["loaded_backbone_sha256"]
    assert head_before != head_after
    for (layer, indices), previous in zip(missing_columns, before_missing, strict=True):
        assert all(np.array_equal(weight[..., indices], before)
                   for weight, before in zip(layer.get_weights(), previous, strict=True))
    return {
        "wall_seconds": round(perf_counter() - start, 3), "loss": losses, "target_collisions": collisions,
        "objects_assigned": object_count - collisions,
        "head_before_sha256": head_before, "head_after_sha256": head_after,
        "backbone_sha256_after_fit": weight_fingerprint(prefix.weights),
    }


def build_predictors(models, train, destination):
    predictors, exports = {}, {}
    for name, model in models.items():
        function = tf.function(lambda image, model=model: model(image, training=False),
                               input_signature=[tf.TensorSpec((1, HEIGHT, WIDTH, 3), tf.float32)])

        def representative_dataset():
            for image in train[:CALIBRATION_COUNT]:
                yield [image[None, ...]]

        path = destination / f"{name}_pilot_int8.tflite"
        model_bytes = convert_int8(function, path, representative_dataset)
        run_int8, input_tensor, output_tensors = lite_runner(model_bytes)
        exports[name] = {"bytes": len(model_bytes), "sha256": sha256(path), "input": input_tensor,
                         "outputs": output_tensors, **audit_flatbuffer(model_bytes, EXPECTED_OPERATORS[name])}
        predictors[name] = {
            "fp32": lambda image, f=function: [out.numpy() for out in tf.nest.flatten(f(image))],
            "int8": run_int8,
        }
    return predictors, exports


def region(x):
    return min(2, max(0, int(x * 3)))


def count_metrics(target, predicted):
    overall, by_class, by_class_region = [], [], []
    for stem, points in target.items():
        truth = Counter((cls, region(x)) for cls, x, _ in points)
        actual = Counter((cls, region(x)) for cls, x, _ in predicted[stem])
        overall.append(abs(truth.total() - actual.total()))
        for cls in range(len(CLASSES)):
            differences = [truth[cls, location] - actual[cls, location] for location in range(3)]
            by_class.append(abs(sum(differences)))
            by_class_region.extend(map(abs, differences))
    return {"total_mae_per_image": float(np.mean(overall)),
            "mean_abs_error_per_image_class": float(np.mean(by_class)),
            "mean_abs_error_per_image_class_region": float(np.mean(by_class_region)),
            "exact_total_images": overall.count(0),
            "objects_predicted": sum(len(points) for points in predicted.values())}


def centroid_metrics(target, predicted):
    distances, location_matches = [], 0
    for stem, truth in target.items():
        actual = predicted[stem]
        candidates = sorted(
            (float(np.hypot((x - tx) * WIDTH, (y - ty) * HEIGHT)), i, j)
            for i, (cls, x, y) in enumerate(actual)
            for j, (tc, tx, ty) in enumerate(truth) if cls == tc
        )
        used_predictions, used_targets = set(), set()
        for distance, i, j in candidates:
            if distance > CENTROID_RADIUS:
                break
            if i not in used_predictions and j not in used_targets:
                used_predictions.add(i)
                used_targets.add(j)
                distances.append(distance)
                location_matches += int(region(actual[i][1]) == region(truth[j][1]))
    matched = len(distances)
    number_truth = sum(len(points) for points in target.values())
    number_predicted = sum(len(points) for points in predicted.values())
    return {
        **count_metrics(target, predicted),
        "target_objects": number_truth, "centroid_matches": matched,
        "centroid_precision": matched / max(1, number_predicted), "centroid_recall": matched / number_truth,
        "centroid_f1": 2 * matched / (number_predicted + number_truth),
        "matched_centroid_mean_distance_input_px": float(np.mean(distances)) if matched else None,
        "matched_location_accuracy": location_matches / matched if matched else None,
        "location_recall_all_ground_truth": location_matches / number_truth,
    }


def bbox_metrics(ground_truth, detections, target_centroids):
    evaluator = REPO / "src/qualifiers/src/evaluation.py"
    if not evaluator.is_file():
        evaluator = REPO / "qualifiers/evaluation.py"
    calculate_map50 = runpy.run_path(str(evaluator))["calculate_map50"]
    map50, class_aps = calculate_map50(ground_truth, detections)
    centers = {}
    for stem, boxes in detections.items():
        centers[stem] = [
            (item["class_id"], (x1 + x2) / (2 * ORIG_WIDTH), (y1 + y2) / (2 * ORIG_HEIGHT))
            for item in boxes if item["score"] >= COUNT_SCORE
            for x1, y1, x2, y2 in [item["bbox"]]
        ]
    return {"bbox_map50": map50, "bbox_ap50_by_class": {CLASSES[k]: v for k, v in class_aps.items()},
            "detections_for_map": sum(len(boxes) for boxes in detections.values()),
            "counts_at_0_30": count_metrics(target_centroids, centers)}


def evaluate(predictors, val_images, val_labels):
    point_truth = {stem: [(c, x, y) for c, x, y, _, _ in boxes] for stem, boxes in val_labels.items()}
    box_truth = {
        stem: [{"class_id": c, "bbox": [(x - w / 2) * ORIG_WIDTH, (y - h / 2) * ORIG_HEIGHT,
                                        (x + w / 2) * ORIG_WIDTH, (y + h / 2) * ORIG_HEIGHT]}
               for c, x, y, w, h in boxes] for stem, boxes in val_labels.items()
    }
    metrics, raw_predictions = {}, {}
    scale = np.array([ORIG_WIDTH / WIDTH, ORIG_HEIGHT / HEIGHT] * 2)
    for name, variants in predictors.items():
        metrics[name], raw_predictions[name] = {}, {}
        for precision, predict in variants.items():
            predictions = {}
            for stem, image in val_images.items():
                outputs = predict(image)
                if name == "fomo":
                    predictions[stem] = [
                        (CLASSES.index(point["class"]), point["xy"][0] / WIDTH, point["xy"][1] / HEIGHT)
                        for point in fomo_centroids(outputs)
                    ]
                else:
                    predictions[stem] = [
                        {**item, "bbox": (np.asarray(item["bbox"]) * scale).tolist()}
                        for item in yolo_detections(outputs, MAP_SCORE_FLOOR)
                    ]
            metrics[name][precision] = (centroid_metrics(point_truth, predictions) if name == "fomo" else
                                        bbox_metrics(box_truth, predictions, point_truth))
            raw_predictions[name][precision] = predictions
    return metrics, raw_predictions, box_truth, point_truth


def frozen_predictions(python, pairs):
    script = """
import json, runpy, sys
release = runpy.run_path(sys.argv[1])
detector = release['Detector']()
files = json.load(sys.stdin)
print(json.dumps({stem: detector.predict(path) for stem, path in files.items()}))
"""
    output = subprocess.run([str(python), "-B", "-c", script, str(REPO / "release_frozen/infer.py")],
                            input=json.dumps({stem: str(paths[0]) for stem, paths in pairs.items()}),
                            text=True, capture_output=True, check=True, timeout=90)
    predictions = json.loads(output.stdout)
    assert predictions.keys() == pairs.keys()
    return predictions


def main(args):
    coco_pairs, coco = ({}, None) if args.coco is None else verify_coco(args.coco)
    large_coco = len(coco_pairs) == COCO_LARGE_IMAGES
    if large_coco:
        if not args.train_large_coco:
            raise ValueError("8k training requires --train-large-coco after coordinator handoff")
        if args.out is None or (args.out.exists() and any(args.out.iterdir())):
            raise ValueError("Choose a new or empty --out directory for the 8k run")
        if any(args.out.resolve().is_relative_to(path.resolve()) for path in (args.coco, args.dataset)):
            raise ValueError("Use an output directory outside the datasets")
    elif args.train_large_coco:
        raise ValueError("--train-large-coco requires the 8k dataset")
    if args.out is None:
        args.out = LAB / ("coco_eai_artifacts" if args.coco else "pilot_artifacts")

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(SEED)
    cv2.setNumThreads(1)
    start = perf_counter()
    pairs, provenance = verify_split(args.dataset)
    if sha256(args.weights) != CHECKPOINT_SHA256:
        raise ValueError("Unexpected ImageNet H5 hash")
    train_images, train_labels = read_split(pairs["train"])
    train = np.concatenate(list(train_images.values()))
    models, transfer = build_models(args.weights)
    assert all(model.count_params() <= 200_000 for model in models.values())
    args.out.mkdir(parents=True, exist_ok=True)
    if coco is not None:
        if large_coco:
            train_coco, coco_labels = coco_pairs, None
            coco["loading"] = {"mode": "on-demand batches", "batch": BATCH, "prefetch": 0}
        else:
            coco_images, coco_labels = read_split(coco_pairs, require_eai_size=False)
            train_coco = np.concatenate(list(coco_images.values()))
        coco["training"] = {
            name: fit_model(name, model, train_coco, coco_labels, transfer[name], coco_mask=True, stream=large_coco)
            for name, model in models.items()
        }
    fits = {name: fit_model(name, model, train, train_labels, transfer[name]) for name, model in models.items()}
    if coco is not None:
        assert all(fits[name]["head_before_sha256"] == coco["training"][name]["head_after_sha256"]
                   for name in fits)
    predictors, exports = build_predictors(models, train, args.out)
    # First decode/parse validation inputs only after fitting and INT8 calibration are complete.
    val_images, val_labels = read_split(pairs["val"])
    metrics, predictions, boxes, points = evaluate(predictors, val_images, val_labels)
    predictions["dtnet_frozen"] = frozen_predictions(args.baseline_python, pairs["val"])
    metrics["dtnet_frozen"] = bbox_metrics(boxes, predictions["dtnet_frozen"], points)
    data = {
        "dataset": provenance, "input_rgb_hw": [HEIGHT, WIDTH], "source_h5_sha256": CHECKPOINT_SHA256,
        "epochs": EPOCHS, "batch": BATCH, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "threads": {"tf_intra": 2, "tf_inter": 1, "tflite": 1},
        "fomo_positive_weight": FOMO_POSITIVE_WEIGHT, "yolo_box_loss_weight": YOLO_BOX_LOSS_WEIGHT,
        "calibration_train_ids": list(train_images)[:CALIBRATION_COUNT],
        "thresholds": {"fomo_score": FOMO_SCORE, "centroid_radius_input_px": CENTROID_RADIUS,
                       "yolo_score_floor_map50": MAP_SCORE_FLOOR, "bbox_count_score": COUNT_SCORE},
        "objects": {"train": dict(Counter(CLASSES[c] for boxes in train_labels.values() for c, *_ in boxes)),
                    "val": dict(Counter(CLASSES[c] for boxes in val_labels.values() for c, *_ in boxes))},
        "parameters_total_with_bn": {name: model.count_params() for name, model in models.items()},
        "transfer": transfer, "training": fits, "exports": exports, "metrics": metrics, "coco": coco,
        "dtnet_reference_may_include_val_in_training": True, "wall_seconds": round(perf_counter() - start, 3),
    }
    for name, artifact in (("pilot_results.json", data), ("pilot_predictions.json", predictions)):
        (args.out / name).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-coco-selection", type=Path, help="Read only the COCO selection JSON and check quotas")
    checks.add_argument("--verify-coco", type=Path, help="Verify all finalized COCO image/label pairs without training")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--weights", type=Path, default=Path.home() / ".keras/models" / FILE_NAME)
    parser.add_argument("--coco", type=Path, help="Complete COCO train2017 subset from prepare_coco.py")
    parser.add_argument("--train-large-coco", action="store_true", help="Opt in to batched 8k training after handoff")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    if arguments.check_coco_selection is not None:
        counts = validate_coco_quota(json.loads(arguments.check_coco_selection.read_bytes()))
        print(json.dumps({"selected": counts.total(), "quota": {CLASSES[k]: v for k, v in counts.items()},
                          "stage": "selection only"}, indent=2))
    elif arguments.verify_coco is not None:
        _, provenance = verify_coco(arguments.verify_coco)
        print(json.dumps(provenance, indent=2))
    else:
        if arguments.dataset is None or arguments.baseline_python is None:
            parser.error("--dataset and --baseline-python are required for training")
        main(arguments)
