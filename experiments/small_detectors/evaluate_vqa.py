"""Score fixed, label-derived EAI questions with oracle and frozen TextCNN intent."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import runpy

REPO = Path(__file__).resolve().parents[2]
SEED = 20261120
WORDS = ("chai nước", "cốc", "điện thoại", "quyển sách", "chuột máy tính", "hộp giấy", "ô tô đồ chơi")
LOCATIONS = ("ben trai", "o giua", "ben phai")
INTENTS = ("COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE")
COUNT_SCORE = 0.30


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region(x):
    return min(2, max(0, int(3 * x)))


def answer(intent, targets, objects, classes):
    counts = Counter(cls for cls, _ in objects)
    if intent == "COUNT":
        return str(sum(counts[cls] for cls in targets))
    if intent == "EXIST":
        return "Co" if any(counts[cls] for cls in targets) else "Khong"
    if intent == "LIST":
        return ", ".join(f"{counts[c]} {name}" for c, name in enumerate(classes) if counts[c]) or "Khong co"
    if intent == "LOCATION":
        votes = Counter(region(x) for cls, x in objects if cls in targets)
        if not votes:
            return "Khong co"
        chosen = max(range(len(LOCATIONS)), key=lambda i: (votes[i], -i))
        return LOCATIONS[chosen]
    if intent == "OUT_OF_SCOPE":
        return "OOS"
    raise ValueError(f"Unexpected intent {intent}")


def load_truth(dataset, expected_manifest):
    path = dataset / "sources.json"
    manifest = json.loads(path.read_bytes())
    if manifest.get("failures") != [] or manifest["seed"] != SEED or digest(path) != expected_manifest:
        raise ValueError("Dataset manifest differs from detector pilot")
    truth = {}
    checksum = hashlib.sha256()
    for image, label in manifest["val"]:
        image_path = dataset / "val/images" / Path(image).name
        label_path = dataset / "val/labels" / Path(label).name
        checksum.update(f"{image_path.name} {digest(image_path)} {label_path.name} {digest(label_path)}\n".encode())
        truth[label_path.stem] = [(int(fields[0]), float(fields[1]))
                                 for row in label_path.read_text().splitlines() if (fields := row.split())]
    if len(truth) != 24:
        raise ValueError("Expected exactly 24 pilot validation IDs")
    return truth, checksum.hexdigest()


def make_questions(truth, classes):
    rng = random.Random(SEED)
    questions = []
    for index, (stem, objects) in enumerate(truth.items()):
        present = sorted({cls for cls, _ in objects})
        absent = sorted(set(range(len(classes))) - set(present))
        count_cls = rng.choice(present)
        exist_cls = rng.choice(absent if index % 2 and absent else present)
        locations = {c: {region(x) for cls, x in objects if cls == c} for c in present}
        unambiguous = [c for c in present if len(locations[c]) == 1]
        rows = [
            ("COUNT", [count_cls], f"Có bao nhiêu {WORDS[count_cls]} trong hình?"),
            ("EXIST", [exist_cls], f"Trong ảnh có {WORDS[exist_cls]} không?"),
            ("LIST", [], "Trong ảnh có những đồ vật gì?"),
        ]
        if unambiguous:
            loc_cls = rng.choice(unambiguous)
            rows.append(("LOCATION", [loc_cls], f"{WORDS[loc_cls].capitalize()} ở vị trí nào trong ảnh?"))
        oos = ("Thời tiết hôm nay thế nào?", "Ngày mai là thứ mấy?")
        rows.append(("OUT_OF_SCOPE", [], oos[index % len(oos)]))
        for kind, targets, text in rows:
            questions.append(dict(question_id=f"{stem}:{kind}", image_id=stem, question=text,
                                  intent=kind, class_ids=targets, expected_answer=answer(kind, targets, objects, classes)))
    return questions


def point_outputs(predictions):
    result = {}
    for model in ("fomo", "yolo"):
        for fmt, images in predictions[model].items():
            result[f"{model}_{fmt}"] = {
                image_id: ([(cls, x) for cls, x, _ in rows] if model == "fomo" else bbox_centers(rows))
                for image_id, rows in images.items()
            }
    result["dtnet_frozen"] = {stem: bbox_centers(rows) for stem, rows in predictions["dtnet_frozen"].items()}
    return result


def bbox_centers(rows):
    return [(row["class_id"], (row["bbox"][0] + row["bbox"][2]) / 640.0)
            for row in rows if row["score"] >= COUNT_SCORE]


def summary(rows, key):
    totals = Counter(row["intent"] for row in rows)
    correct = Counter(row["intent"] for row in rows if row[key])
    by_intent = {k: {"n": totals[k], "correct": correct[k], "accuracy": correct[k] / totals[k]} for k in INTENTS}
    combined = (correct["COUNT"] + correct["EXIST"]) / (totals["COUNT"] + totals["EXIST"])
    return {"accuracy": sum(correct.values()) / len(rows), "by_intent": by_intent,
            "btc_weighted": .35 * combined + .25 * by_intent["LIST"]["accuracy"]
            + .30 * by_intent["LOCATION"]["accuracy"] + .10 * by_intent["OUT_OF_SCOPE"]["accuracy"]}


def main(args):
    pilot_file = args.predictions / "pilot_results.json"
    pilot = json.loads(pilot_file.read_bytes())
    if pilot["thresholds"]["bbox_count_score"] != COUNT_SCORE:
        raise ValueError("VQA count score must match the pilot")
    truth, checksum = load_truth(args.dataset, pilot["dataset"]["manifest_sha256"])
    if checksum != pilot["dataset"]["split_sha256"]["val"]:
        raise ValueError("Held-out image/label bytes changed")
    pred_path = args.predictions / "pilot_predictions.json"
    outputs = point_outputs(json.loads(pred_path.read_bytes()))
    outputs["ground_truth_detections"] = truth
    if any(value.keys() != truth.keys() for value in outputs.values()):
        raise ValueError("All predictions must cover the same held-out image IDs")

    release = runpy.run_path(str(REPO / "release_frozen/infer.py"))
    classes = release["CLASS_NAMES"]
    intent = release["IntentClassifier"]()
    normalize = lambda text: release["strip_accents"](text).lower()
    slots = tuple(re.compile(r"\b" + re.escape(normalize(word)) + r"\b") for word in WORDS)
    questions = make_questions(truth, classes)
    intent_rows = []
    for q in questions:
        text = normalize(q["question"])
        targets = [c for c, slot in enumerate(slots) if slot.search(text)]
        assert targets == q["class_ids"], q
        predicted_intent = intent.predict(q["question"])
        intent_rows.append({
            **q, "parsed_class_ids": targets, "predicted_intent": predicted_intent,
            "intent_correct": predicted_intent == q["intent"],
        })
    scores, answers = {}, {}
    for name, predictions in outputs.items():
        rows = []
        for q in intent_rows:
            evidence = predictions[q["image_id"]]
            oracle = answer(q["intent"], q["parsed_class_ids"], evidence, classes)
            pipeline = answer(q["predicted_intent"], q["parsed_class_ids"], evidence, classes)
            rows.append(dict(question_id=q["question_id"], intent=q["intent"],
                             oracle_answer=oracle, pipeline_answer=pipeline,
                             oracle_correct=oracle == q["expected_answer"],
                             pipeline_correct=pipeline == q["expected_answer"]))
        scores[name] = {"oracle_intent": summary(rows, "oracle_correct"),
                        "frozen_textcnn": summary(rows, "pipeline_correct")}
        answers[name] = rows
    assert scores["ground_truth_detections"]["oracle_intent"]["accuracy"] == 1.0
    args.out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "vqa_questions.json": intent_rows,
        "vqa_answers.json": answers,
        "vqa_results.json": {
            "question_seed": SEED, "n_images": len(truth), "n_questions": len(questions),
            "questions_by_intent": dict(Counter(q["intent"] for q in questions)),
            "val_split_sha256": checksum, "detector_predictions_sha256": digest(pred_path),
            "textcnn_sha256": digest(REPO / "release_frozen/weights/text_cnn_int8.tflite"),
            "intent": summary(intent_rows, "intent_correct"), "answers": scores,
        },
    }
    for filename, payload in payloads.items():
        (args.out / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(questions)} questions on {len(truth)} images; TextCNN {sum(q['intent_correct'] for q in intent_rows)}/{len(questions)}")
    for name, score in scores.items():
        print(f"{name:24s} oracle={score['oracle_intent']['accuracy']:.4f}  pipeline={score['frozen_textcnn']['accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
