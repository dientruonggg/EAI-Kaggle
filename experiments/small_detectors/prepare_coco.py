"""Select COCO train2017 labels and acquire only the selected JPEGs."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import re
from time import sleep
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

SEED = 20261120
COCO_TO_EAI = {39: 0, 41: 1, 67: 2, 73: 3, 64: 4}
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
IMAGE_ROOT = "https://images.cocodataset.org/train2017/"
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
LABEL_MEMBER = re.compile(r"coco/labels/train2017/(\d{12})\.txt")
COCO128_MEMBER = re.compile(r"coco128/images/train2017/(\d{12})\.jpg")
MAX_JPEG_BYTES = 16 * 1024 * 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_samples(labels, per_class, available):
    pools = defaultdict(list)
    with ZipFile(labels) as archive:
        for member in archive.namelist():
            match = LABEL_MEMBER.fullmatch(member)
            if match is None:
                continue
            image_id = match[1]
            if available is not None and image_id not in available:
                continue
            mapped = []
            for line in archive.read(member).decode().splitlines():
                fields = line.split()
                if fields and int(fields[0]) in COCO_TO_EAI:
                    if len(fields) != 5:
                        raise ValueError(f"Expected bbox labels in {member}")
                    mapped.append((COCO_TO_EAI[int(fields[0])], fields[1:]))
            if mapped:
                for class_id in {label[0] for label in mapped}:
                    pools[class_id].append((image_id, mapped))

    rng = random.Random(SEED)
    selected, used = [], set()
    order = sorted(COCO_TO_EAI.values(), key=lambda cls: (len(pools[cls]), cls) if available is not None else cls)
    for class_id in order:
        eligible = sorted((image_id, labels) for image_id, labels in pools[class_id] if image_id not in used)
        count = per_class if available is None else min(per_class, len(eligible))
        if not count or len(eligible) < count:
            raise ValueError(f"Only {len(eligible)} examples available for EAI class {class_id}")
        for image_id, mapped in rng.sample(eligible, count):
            used.add(image_id)
            selected.append({"id": image_id, "quota_class": class_id, "url": f"{IMAGE_ROOT}{image_id}.jpg",
                             "labels": [" ".join([str(cls), *box]) for cls, box in mapped]})
    return selected


def valid_jpeg(image):
    return 4 <= len(image) <= MAX_JPEG_BYTES and image.startswith(b"\xff\xd8") and image.endswith(b"\xff\xd9")


def fetch_image(url):
    for attempt in range(2):
        try:
            with urlopen(url, timeout=8) as response:
                image = response.read(MAX_JPEG_BYTES + 1)
            if not valid_jpeg(image):
                raise ValueError(f"Invalid JPEG response from {url}")
            return image
        except (URLError, TimeoutError) as error:
            if attempt == 1:
                raise SystemExit(f"COCO image download failed: {url}: {error}") from None
            sleep(0.4)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    image_source = ZipFile(args.image_archive) if args.image_archive else nullcontext()
    with image_source as archive:
        prepare(args, archive)


def prepare(args, archive):
    available = None
    if archive is not None:
        available = {match[1]: name for name in archive.namelist() if (match := COCO128_MEMBER.fullmatch(name))}
        if len(available) != 128:
            raise ValueError("Expected 128 train2017 JPEGs in the official Ultralytics COCO128 archive")
    manifest = {
        "source": "COCO train2017", "labels_url": LABELS_URL, "labels_sha256": digest(args.labels),
        "seed": SEED, "mapping": COCO_TO_EAI, "per_class": args.per_class,
        "images": select_samples(args.labels, args.per_class, available),
    }
    if archive is not None:
        manifest["image_archive"] = {"url": COCO128_URL, "sha256": digest(args.image_archive)}
    (args.out / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Selected {len(manifest['images'])} unique train2017 images", flush=True)
    if args.selection_only:
        return

    image_dir, label_dir = args.out / "images", args.out / "labels"
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    for item in manifest["images"]:
        image_path = image_dir / f"{item['id']}.jpg"
        label_path = label_dir / f"{item['id']}.txt"
        if not image_path.exists():
            image = fetch_image(item["url"]) if archive is None else archive.read(available[item["id"]])
            if not valid_jpeg(image):
                raise ValueError(f"Invalid JPEG: {item['id']}")
            image_path.write_bytes(image)
        if archive is not None and image_path.read_bytes() != archive.read(available[item["id"]]):
            raise ValueError(f"Cached JPEG differs from COCO128 archive: {image_path}")
        if not valid_jpeg(image_path.read_bytes()):
            raise ValueError(f"Invalid cached JPEG: {image_path}")
        label_path.write_text("\n".join(item["labels"]) + "\n")
        item["image_sha256"] = digest(image_path)
        item["label_sha256"] = digest(label_path)
    temporary = args.out / "sources.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.out / "sources.json")
    print(f"Ready: {args.out / 'sources.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, choices=range(1, 33), default=32)
    parser.add_argument("--image-archive", type=Path, help="Use an existing official COCO128 ZIP without network")
    parser.add_argument("--selection-only", action="store_true")
    main(parser.parse_args())
