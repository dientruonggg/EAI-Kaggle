"""Verify the lab artifacts and invoke them independently with LiteRT on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory

from ai_edge_litert.interpreter import Interpreter
import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
ARTIFACTS = LAB / "artifacts"
REPO = LAB.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_baseline(crops):
    frozen = runpy.run_path(str(REPO / "release_frozen/infer.py"))
    detector = frozen["Detector"]()
    summaries = {}
    with TemporaryDirectory(prefix="eai-baseline-crops-") as directory:
        for image_id, rgb in crops.items():
            image_path = Path(directory) / f"{image_id}.png"
            if not cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not write baseline fixture {image_path}")
            boxes = detector.predict(image_path)
            assert 0 < len(boxes) <= frozen["MAX_DETECTIONS"]
            assert all(0 <= box["class_id"] < len(frozen["CLASS_NAMES"]) for box in boxes)
            assert all(
                0 <= x1 <= x2 <= frozen["IMAGE_WIDTH"] and 0 <= y1 <= y2 <= frozen["IMAGE_HEIGHT"]
                for x1, y1, x2, y2 in (box["bbox"] for box in boxes)
            )
            summaries[image_id] = len(boxes)
    print(f"PASS: frozen Detect smoke bbox counts {json.dumps(summaries)}")


def main(include_baseline):
    results = json.loads((ARTIFACTS / "results.json").read_bytes())
    preview_path = REPO / "DuyToan/best/val_preview.png"
    assert digest(preview_path) == results["preview_sha256"]
    preview = cv2.cvtColor(cv2.imread(str(preview_path)), cv2.COLOR_BGR2RGB)
    height, width = results["input_rgb_hw"]
    crop_width, crop_height = results["crop_wh"]
    crops = {
        image_id: preview[y:y + crop_height, x:x + crop_width]
        for image_id, (x, y) in results["crop_xy"].items()
    }
    cases = 0
    for name, model in results["models"].items():
        path = ARTIFACTS / f"{name}_lab_int8.tflite"
        info = model["int8"]
        assert digest(path) == info["sha256"]
        assert path.stat().st_size == info["bytes"] <= 1_500_000
        params = model["parameters"]
        assert params["trainable"] + params["bn_running_state"] == params["total"] <= 200_000
        transfer = model["transfer"]
        assert transfer["random_backbone_sha256"] != transfer["loaded_backbone_sha256"]
        assert transfer["backbone_total_including_bn"] + transfer["head_or_neck_random_parameters"] == params["total"]

        lite = Interpreter(model_path=str(path), num_threads=1)
        lite.allocate_tensors()
        incoming, = lite.get_input_details()
        assert tuple(incoming["shape"]) == (1, height, width, 3) and incoming["dtype"] == np.int8
        scale, zero_point = incoming["quantization"]
        outputs = sorted(lite.get_output_details(), key=lambda t: t["shape"][1], reverse=True)
        assert [t["shape"].tolist() for t in outputs] == [t["shape"] for t in info["outputs"]]
        assert all(t["dtype"] == np.int8 for t in outputs)

        for image_id, crop in crops.items():
            rgb = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
            image = rgb.astype(np.float32) / 127.5 - 1.0
            image = np.rint(image / scale + zero_point).clip(-128, 127).astype(np.int8)
            lite.set_tensor(incoming["index"], image[None])
            lite.invoke()
            for tensor in outputs:
                values = lite.get_tensor(tensor["index"])
                assert values.size and values.min() < values.max()
            assert image_id in model["smoke"]
            cases += 1
    assert cases == 6
    print(f"PASS: SHA-256, INT8 I/O and LiteRT Invoke for {cases} model × image cases")
    if include_baseline:
        verify_frozen_baseline(crops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Also invoke frozen Detect on the same crops")
    main(parser.parse_args().baseline)
