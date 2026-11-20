#!/usr/bin/env python3
"""Restore the exact EAI COCO train2017 subset from its sealed manifest."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE.parents[2] / "datasets" / "coco2017_eai_8k"
MANIFEST = HERE / "sources.json.gz"
MANIFEST_SHA256 = "dc83fdfeeb9d62e772288e7a1ad94a69fe827cd2643e0e5b1d9a70f99e780227"
CLASSES = ("bottle", "cup", "phone", "book", "mouse")
MAPPING = {"39": 0, "41": 1, "67": 2, "73": 3, "64": 4}
PER_CLASS = 1600
EXPECTED_IMAGES = len(CLASSES) * PER_CLASS
REGIONAL = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/"
GLOBAL = "https://s3.amazonaws.com/images.cocodataset.org/train2017/"
COCO = "https://images.cocodataset.org/train2017/"
BUFFER_SIZE = 128 * 1024
HTTP_TIMEOUT = 20
MAX_ATTEMPTS = 3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def label_bytes(item):
    return ("\n".join(item["labels"]) + "\n").encode("ascii")


def load_manifest():
    raw = gzip.decompress(MANIFEST.read_bytes())
    if digest(raw) != MANIFEST_SHA256:
        raise ValueError("Locked manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    rows = manifest["images"]
    if (manifest["source"], manifest["seed"], manifest["per_class"], tuple(manifest["classes"]),
            manifest["mapping"], len(rows)) != (
            "COCO train2017", 20261120, PER_CLASS, CLASSES, MAPPING, EXPECTED_IMAGES):
        raise ValueError("Unexpected COCO subset provenance")

    seen = set()
    quota = Counter()
    for item in rows:
        stem = item["id"]
        if (not re.fullmatch(r"[0-9]{12}", stem) or stem in seen
                or item["url"] != f"{COCO}{stem}.jpg" or item["download_url"] != f"{REGIONAL}{stem}.jpg"
                or not isinstance(item["image_bytes"], int) or item["image_bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", item["image_sha256"])):
            raise ValueError(f"Invalid locked image entry: {stem}")
        if digest(label_bytes(item)) != item["label_sha256"]:
            raise ValueError(f"Locked label SHA-256 mismatch: {stem}")
        classes = {int(line.split()[0]) for line in item["labels"]}
        if item["quota_class"] not in classes or not classes <= set(MAPPING.values()):
            raise ValueError(f"Invalid mapped labels: {stem}")
        seen.add(stem)
        quota[item["quota_class"]] += 1
    if quota != Counter({i: PER_CLASS for i in range(len(CLASSES))}):
        raise ValueError(f"Unexpected primary-class quota: {quota}")
    return raw, rows


def paths(dest, item):
    stem = item["id"]
    return dest / "images" / f"{stem}.jpg", dest / "labels" / f"{stem}.txt"


def image_matches(path, item):
    if not path.is_file() or path.stat().st_size != item["image_bytes"]:
        return False
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(BUFFER_SIZE), b""):
            sha.update(block)
    return sha.hexdigest() == item["image_sha256"]


def bytes_match(path, data):
    return path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data


def image_issue(dest, item):
    image_path, label_path = paths(dest, item)
    if not image_matches(image_path, item):
        return "JPEG missing or size/SHA-256 mismatch"
    if not bytes_match(label_path, label_bytes(item)):
        return "label missing or byte mismatch"
    return None


def atomic_write(path, writer):
    handle = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            writer(handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_image(path, item):
    urls = (item["download_url"], f"{GLOBAL}{item['id']}.jpg")
    errors = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        url = urls[(attempt - 1) % len(urls)]
        def write_verified(handle):
            sha = hashlib.sha256()
            size = 0
            with urlopen(url, timeout=HTTP_TIMEOUT) as response:
                for block in iter(lambda: response.read(BUFFER_SIZE), b""):
                    size += len(block)
                    if size > item["image_bytes"]:
                        raise ValueError("JPEG exceeds locked byte count")
                    sha.update(block)
                    handle.write(block)
            if size != item["image_bytes"] or sha.hexdigest() != item["image_sha256"]:
                raise ValueError("JPEG size or SHA-256 mismatch")

        try:
            atomic_write(path, write_verified)
            return
        except (OSError, HTTPException, ValueError) as error:
            errors.append(f"{url}: {error}")
            if attempt < MAX_ATTEMPTS:
                sleep(attempt)
    raise RuntimeError(f"{item['id']}: failed after {MAX_ATTEMPTS} attempts\n" + "\n".join(errors))


def restore_pair(dest, item):
    image_path, label_path = paths(dest, item)
    downloaded = not image_matches(image_path, item)
    if downloaded:
        download_image(image_path, item)
    expected_label = label_bytes(item)
    if not bytes_match(label_path, expected_label):
        atomic_write(label_path, lambda file: file.write(expected_label))
    return downloaded


def check_ready_marker(dest, raw):
    marker = dest / "sources.json"
    if marker.exists() and not bytes_match(marker, raw):
        raise ValueError(f"Existing marker differs from locked manifest: {marker}")
    return marker


def verify(dest, raw, rows):
    marker = check_ready_marker(dest, raw)
    expected_classes = ("\n".join(CLASSES) + "\n").encode("ascii")
    if not bytes_match(dest / "classes.txt", expected_classes):
        raise ValueError("classes.txt missing or byte mismatch")
    for item in rows:
        issue = image_issue(dest, item)
        if issue:
            raise ValueError(f"{item['id']}: {issue}")
    status = "ready marker exists" if marker.is_file() else "no ready marker"
    print(f"Verified {len(rows)}/{EXPECTED_IMAGES} JPEG/label pairs and classes.txt ({status})")


def restore(dest, raw, rows, workers, complete):
    if complete and len(rows) != EXPECTED_IMAGES:
        raise ValueError("Cannot publish a ready marker for an incomplete selection")
    marker = check_ready_marker(dest, raw)
    expected_classes = ("\n".join(CLASSES) + "\n").encode("ascii")
    pending = [item for item in rows if image_issue(dest, item)]
    classes_ok = bytes_match(dest / "classes.txt", expected_classes)
    if pending or not classes_ok:
        # A failed or interrupted repair must leave no ready marker for inconsistent bytes.
        marker.unlink(missing_ok=True)
        (dest / "images").mkdir(parents=True, exist_ok=True)
        (dest / "labels").mkdir(parents=True, exist_ok=True)
        if not classes_ok:
            atomic_write(dest / "classes.txt", lambda file: file.write(expected_classes))

    downloaded = 0
    print(f"Selected: {len(rows)}; verified cache: {len(rows) - len(pending)}; pending: {len(pending)}", flush=True)
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Only one group is enqueued at once so errors stop a large restoration promptly.
            for offset in range(0, len(pending), workers):
                group = pending[offset:offset + workers]
                downloaded += sum(pool.map(lambda item: restore_pair(dest, item), group))
                done = offset + len(group)
                if done // 256 != offset // 256 or done == len(pending):
                    print(f"Restored {done}/{len(pending)} pending pairs", flush=True)

    verify(dest, raw, rows)
    if complete and not marker.is_file():
        atomic_write(marker, lambda file: file.write(raw))
        print(f"Published {marker}")
    elif not complete:
        print("Partial run: no ready marker published.")
    print(f"Downloaded JPEGs: {downloaded}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="dataset destination")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="show locked selection without writes or network")
    mode.add_argument("--verify", action="store_true", help="read-only: verify local files and an existing marker")
    parser.add_argument("--limit", type=int, help="process the first N locked entries, without publishing sources.json")
    parser.add_argument("--workers", type=int, default=8, help="concurrent downloads, 1–32")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit < EXPECTED_IMAGES:
        parser.error(f"--limit must be between 1 and {EXPECTED_IMAGES - 1}; omit it for the full set")
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")

    try:
        raw, rows = load_manifest()
        selected = rows if args.limit is None else rows[:args.limit]
        if args.dry_run:
            size = sum(item["image_bytes"] for item in selected)
            print(f"{args.dest}: {len(selected)}/{EXPECTED_IMAGES} locked JPEGs, {size:,} JPEG bytes")
            for item in selected[:3]:
                print(item["download_url"])
        elif args.verify:
            verify(args.dest, raw, selected)
        else:
            restore(args.dest, raw, selected, args.workers, complete=args.limit is None)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
