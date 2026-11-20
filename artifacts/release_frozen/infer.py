"""Local inference for the frozen 98.50673 qualifiers baseline."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent
DETECT_MODEL = Path("weights/dtnet_nwd_tal_full6k_int8.tflite")
INTENT_MODEL = Path("weights/text_cnn_int8.tflite")
INTENT_VOCAB = Path("weights/text_cnn_vocab.json")

IMAGE_WIDTH, IMAGE_HEIGHT = 320, 240
STRIDE = 8
MAX_DISTANCE_CELLS = 20.0
SCORE_THRESHOLD = 0.020
NMS_IOU = 0.65
MAX_DETECTIONS = 100
CLASS_NAMES = ("bottle", "cup", "phone", "book", "mouse", "box", "toy_car")
INTENTS = ("COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE")
INTENT_MAX_LENGTH = 18
GUARDED_INTENTS = {"COUNT", "EXIST", "LOCATION"}
STOPWORDS = {
    "ban", "oi", "cho", "toi", "hoi", "xin", "minh",
    "vay", "nhi", "the", "a", "gium", "giup",
}
TARGET_ALIASES = (
    "chai", "binh nuoc", "chai nuoc", "lo", "chai nhua", "chai thuy tinh",
    "coc", "ly", "tach", "coc uong nuoc", "ly nuoc",
    "dien thoai", "smartphone", "dien thoai thong minh", "dt", "chiec dien thoai", "dien thoai di dong",
    "sach", "cuon sach", "quyen sach", "tap vo", "giao trinh", "tai lieu",
    "chuot", "chuot may tinh", "con chuot", "chuot khong day", "chuot quang",
    "hop", "thung", "hop giay", "cai hop", "hop carton", "hop qua", "thung giay",
    "oto do choi", "xe do choi", "xe con do choi", "xe hoi do choi", "xe mo hinh", "xe oto do choi",
    "cai o to do choi", "cai oto do choi", "o to do choi", "oto", "o to",
)


def read_frozen_asset(relative_path):
    path = ROOT / relative_path
    if path.resolve() != path:
        raise ValueError(f"Expected a local file at {relative_path}")
    manifest = (ROOT / "SHA256SUMS").read_text(encoding="ascii")
    checksums = {}
    for line in manifest.splitlines():
        checksum, filename = line.split("  ", 1)
        checksums[filename] = checksum
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != checksums.get(relative_path.as_posix()):
        raise ValueError(f"SHA-256 mismatch for {relative_path}")
    return content


def open_model(relative_path):
    interpreter = Interpreter(model_content=read_frozen_asset(relative_path))
    interpreter.allocate_tensors()
    return interpreter


def class_nms(boxes, scores):
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while len(order):
        current = order[0]
        keep.append(current)
        others = order[1:]
        if not len(others):
            break
        width = np.maximum(0.0, np.minimum(x2[current], x2[others]) - np.maximum(x1[current], x1[others]))
        height = np.maximum(0.0, np.minimum(y2[current], y2[others]) - np.maximum(y1[current], y1[others]))
        intersection = width * height
        iou = intersection / (areas[current] + areas[others] - intersection + 1e-7)
        order = others[iou <= NMS_IOU]
    return keep[:MAX_DETECTIONS]


def decode_detections(class_logits, box_logits):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(class_logits, -30, 30)))
    distances = (1.0 / (1.0 + np.exp(-np.clip(box_logits, -30, 30)))) * (MAX_DISTANCE_CELLS * STRIDE)
    _, grid_height, grid_width = probabilities.shape
    grid_x = ((np.arange(grid_width) + 0.5) * STRIDE).reshape(1, grid_width)
    grid_y = ((np.arange(grid_height) + 0.5) * STRIDE).reshape(grid_height, 1)
    coordinates = (
        np.clip(grid_x - distances[0], 0, IMAGE_WIDTH),
        np.clip(grid_y - distances[1], 0, IMAGE_HEIGHT),
        np.clip(grid_x + distances[2], 0, IMAGE_WIDTH),
        np.clip(grid_y + distances[3], 0, IMAGE_HEIGHT),
    )

    detections = []
    for class_id, class_scores in enumerate(probabilities):
        mask = class_scores >= SCORE_THRESHOLD
        if not np.any(mask):
            continue
        scores = class_scores[mask]
        boxes = np.stack([coordinate[mask] for coordinate in coordinates], axis=1)
        keep = class_nms(boxes, scores)
        detections.append(np.column_stack((np.full(len(keep), class_id), boxes[keep], scores[keep])))

    if not detections:
        return []
    rows = np.concatenate(detections)
    if len(rows) > MAX_DETECTIONS:
        # BTC preserves class order when the global cap is not reached.
        rows = rows[np.argsort(-rows[:, 5])[:MAX_DETECTIONS]]

    predictions = []
    for class_id, x1, y1, x2, y2, score in rows:
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append({
            "class_id": int(class_id),
            "score": round(float(score), 5),
            "bbox": [round(float(value), 2) for value in (x1, y1, x2, y2)],
        })
    return predictions


class Detector:
    def __init__(self):
        self.interpreter = open_model(DETECT_MODEL)
        self.input_tensor = self.interpreter.get_input_details()[0]
        outputs = self.interpreter.get_output_details()
        class_index = 0 if outputs[0]["shape"][-1] == len(CLASS_NAMES) else 1
        self.class_output = outputs[class_index]
        self.box_output = outputs[1 - class_index]

    def output_logits(self, tensor):
        scale, zero_point = tensor["quantization"]
        output = self.interpreter.get_tensor(tensor["index"])[0].astype(np.float32)
        return ((output - zero_point) * scale).transpose(2, 0, 1)

    def predict(self, image_path):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (IMAGE_WIDTH, IMAGE_HEIGHT):
            rgb = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        normalized = rgb.astype(np.float32) / 255.0
        scale, zero_point = self.input_tensor["quantization"]
        quantized = np.round(normalized / scale + zero_point).clip(-128, 127).astype(np.int8)

        self.interpreter.set_tensor(self.input_tensor["index"], quantized[None, ...])
        self.interpreter.invoke()
        return decode_detections(self.output_logits(self.class_output), self.output_logits(self.box_output))


def strip_accents(text):
    decomposed = unicodedata.normalize("NFD", text)
    return re.sub(r"[\u0300-\u036f]", "", decomposed).replace("đ", "d").replace("Đ", "d")


def has_target_class(text):
    normalized = strip_accents(text).lower()
    return any(re.search(r"\b" + re.escape(alias) + r"\b", normalized) for alias in TARGET_ALIASES)


class IntentClassifier:
    def __init__(self):
        self.vocab = json.loads(read_frozen_asset(INTENT_VOCAB))
        self.interpreter = open_model(INTENT_MODEL)
        self.input_tensor = self.interpreter.get_input_details()[0]
        self.output_tensor = self.interpreter.get_output_details()[0]

    def encode(self, text):
        normalized = unicodedata.normalize("NFC", strip_accents(text))
        words = re.sub(r"[^\w\s]", " ", normalized.lower()).split()
        unknown_id = self.vocab.get("<UNK>", 1)
        token_ids = [self.vocab.get(word, unknown_id) for word in words if word not in STOPWORDS][:INTENT_MAX_LENGTH]
        token_ids += [self.vocab.get("<PAD>", 0)] * (INTENT_MAX_LENGTH - len(token_ids))
        return np.array([token_ids], dtype=self.input_tensor["dtype"])

    def predict(self, text):
        self.interpreter.set_tensor(self.input_tensor["index"], self.encode(text))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_tensor["index"])[0]
        if self.output_tensor["dtype"] == np.int8:
            scale, zero_point = self.output_tensor["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale
        label = INTENTS[int(np.argmax(output))]
        if label in GUARDED_INTENTS and not has_target_class(text):
            return "OUT_OF_SCOPE"
        return label


def compact_json(prediction):
    return json.dumps(prediction, separators=(",", ":"))


def submission_rows(image_dir, question_csv):
    if not image_dir.is_dir():
        raise ValueError(f"Cannot read image directory {image_dir}")
    images = sorted((*image_dir.glob("*.png"), *image_dir.glob("*.jpg")))
    if not images:
        raise ValueError(f"No .png or .jpg files in {image_dir}")

    with question_csv.open(encoding="utf-8", newline="") as questions:
        reader = csv.DictReader(questions)
        detector = Detector()
        classifier = IntentClassifier()
        yield ["task_type", "sample_id", "prediction"]
        for image_path in images:
            yield ["DETECTION", image_path.stem, compact_json(detector.predict(image_path))]
        for row in reader:
            question_id = row.get("question_id") or row.get("sample_id") or row.get("id")
            text = (row.get("question") or row.get("text") or "").strip()
            yield ["INTENT", question_id, classifier.predict(text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("detect", help="Write bounding boxes as JSON").add_argument("image", type=Path)
    commands.add_parser("intent", help="Write the predicted intent label").add_argument("question")
    batch = commands.add_parser("submission", help="Write BTC-format CSV to stdout")
    batch.add_argument("--images", type=Path, required=True)
    batch.add_argument("--questions", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "detect":
            print(compact_json(Detector().predict(args.image)))
        elif args.command == "intent":
            print(IntentClassifier().predict(args.question))
        else:
            csv.writer(sys.stdout).writerows(submission_rows(args.images, args.questions))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
