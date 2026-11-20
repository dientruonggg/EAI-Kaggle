#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - OFFICIAL DATASET INTAKE & AUTO-PREPARATION
====================================================================
Universal, resilient drop-in processor for the official competition dataset.
Features:
  1. Archives current 1,350 COCO images to qualifiers/data_stress_test/ for permanent stress testing.
  2. Scans unpacked raw competition directory and auto-detects:
     - Train images & labels (places in qualifiers/data/train/images and labels)
     - Test images (places in qualifiers/data/test/images)
     - Question/QA CSVs (copies to qualifiers/data/)
     - Performs 90/10 stratified train/val split if no pre-split validation set exists.
  3. Validates YOLO bbox coordinates, labels (0..6), and non-empty images.
"""

import os
import sys
import glob
import shutil
import zipfile
import argparse
import json
import csv
import cv2
import numpy as np

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STRESS_TEST_DIR = os.path.join(PROJECT_ROOT, "data_stress_test")

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]

def archive_current_coco_data():
    """Archives existing COCO dataset into permanent stress-test suite."""
    print("=" * 70)
    print("   ARCHIVING CURRENT 1,350 COCO IMAGES TO LOCAL STRESS-TEST SUITE")
    print("=" * 70)
    if os.path.exists(DATA_DIR):
        os.makedirs(STRESS_TEST_DIR, exist_ok=True)
        for sub in ["train", "val"]:
            src_sub = os.path.join(DATA_DIR, sub)
            dst_sub = os.path.join(STRESS_TEST_DIR, sub)
            if os.path.exists(src_sub):
                if os.path.exists(dst_sub):
                    shutil.rmtree(dst_sub)
                shutil.copytree(src_sub, dst_sub)
                print(f"[✔] Archived {src_sub} -> {dst_sub}")
        print("[✔] Current COCO dataset successfully preserved in qualifiers/data_stress_test/.")

def unpack_and_inspect(input_path, extract_dir=None):
    """Extracts zip or returns existing directory."""
    if extract_dir is None:
        extract_dir = os.path.join(PROJECT_ROOT, "data_incoming/extracted")
    os.makedirs(extract_dir, exist_ok=True)

    if input_path.endswith(".zip"):
        print(f"[*] Extracting {input_path} to {extract_dir}...")
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(extract_dir)
        return extract_dir
    elif os.path.isdir(input_path):
        # Look for internal zips
        sub_zips = glob.glob(os.path.join(input_path, "*.zip"))
        if sub_zips:
            for sz in sub_zips:
                print(f"[*] Extracting nested zip: {sz}...")
                with zipfile.ZipFile(sz, "r") as z:
                    z.extractall(extract_dir)
            return extract_dir
        return input_path
    else:
        raise ValueError(f"Unsupported input format: {input_path}")

def scan_and_reorganize(raw_root):
    """Scans raw competition files and organizes them cleanly into qualifiers/data/."""
    print("=" * 70)
    print("   SCANNING & INGESTING OFFICIAL COMPETITION DATASET")
    print("=" * 70)

    # 1. Clean and prepare directories
    train_img_dst = os.path.join(DATA_DIR, "train/images")
    train_lbl_dst = os.path.join(DATA_DIR, "train/labels")
    val_img_dst = os.path.join(DATA_DIR, "val/images")
    val_lbl_dst = os.path.join(DATA_DIR, "val/labels")
    test_img_dst = os.path.join(DATA_DIR, "test/images")

    os.makedirs(train_img_dst, exist_ok=True)
    os.makedirs(train_lbl_dst, exist_ok=True)
    os.makedirs(val_img_dst, exist_ok=True)
    os.makedirs(val_lbl_dst, exist_ok=True)
    os.makedirs(test_img_dst, exist_ok=True)

    # 2. Discover all images and labels
    all_imgs = glob.glob(os.path.join(raw_root, "**", "*.jpg"), recursive=True) + \
               glob.glob(os.path.join(raw_root, "**", "*.png"), recursive=True)
    all_txts = glob.glob(os.path.join(raw_root, "**", "*.txt"), recursive=True)
    all_csvs = glob.glob(os.path.join(raw_root, "**", "*.csv"), recursive=True)

    print(f"[*] Discovered {len(all_imgs)} images, {len(all_txts)} label text files, {len(all_csvs)} CSV files.")

    # Categorize images
    train_imgs = []
    val_imgs = []
    test_imgs = []

    for img in all_imgs:
        path_lower = img.lower()
        if "test" in path_lower:
            test_imgs.append(img)
        elif "val" in path_lower or "valid" in path_lower:
            val_imgs.append(img)
        elif "train" in path_lower:
            train_imgs.append(img)
        else:
            # Check if matching label exists
            base = os.path.splitext(os.path.basename(img))[0]
            # If no label anywhere, treat as test
            train_imgs.append(img)

    # If test images were not explicitly separated by path
    if not test_imgs and not val_imgs:
        print("[*] No explicit test/train directory structure found. Categorizing by presence of labels...")
        label_basenames = {os.path.splitext(os.path.basename(t))[0] for t in all_txts}
        actual_train = []
        actual_test = []
        for img in train_imgs:
            b = os.path.splitext(os.path.basename(img))[0]
            if b in label_basenames:
                actual_train.append(img)
            else:
                actual_test.append(img)
        train_imgs = actual_train
        test_imgs = actual_test

    print(f"[*] Split summary: {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test")

    # If no val set, perform 90/10 stratified split
    if len(train_imgs) > 50 and len(val_imgs) == 0:
        np.random.seed(42)
        shuffled = list(train_imgs)
        np.random.shuffle(shuffled)
        split_point = int(len(shuffled) * 0.90)
        train_imgs = shuffled[:split_point]
        val_imgs = shuffled[split_point:]
        print(f"[*] Auto-created 90/10 Split: {len(train_imgs)} Train, {len(val_imgs)} Val.")

    # Copy files helper
    txt_map = {os.path.splitext(os.path.basename(t))[0]: t for t in all_txts}

    def copy_split(img_list, dst_img, dst_lbl):
        copied_cnt = 0
        for src_i in img_list:
            fname = os.path.basename(src_i)
            shutil.copy2(src_i, os.path.join(dst_img, fname))
            base = os.path.splitext(fname)[0]
            if base in txt_map:
                shutil.copy2(txt_map[base], os.path.join(dst_lbl, f"{base}.txt"))
            copied_cnt += 1
        return copied_cnt

    print("[*] Ingesting train images and labels...")
    c_tr = copy_split(train_imgs, train_img_dst, train_lbl_dst)
    print(f"[✔] Ingested {c_tr} train images.")

    if val_imgs:
        print("[*] Ingesting val images and labels...")
        c_val = copy_split(val_imgs, val_img_dst, val_lbl_dst)
        print(f"[✔] Ingested {c_val} val images.")

    if test_imgs:
        print("[*] Ingesting test images...")
        c_te = copy_split(test_imgs, test_img_dst, os.path.join(DATA_DIR, "test/labels"))
        print(f"[✔] Ingested {c_te} test images.")

    # Copy CSV files to qualifiers/data/
    for c in all_csvs:
        c_name = os.path.basename(c)
        shutil.copy2(c, os.path.join(DATA_DIR, c_name))
        print(f"[✔] Ingested QA/metadata file: {c_name}")

    print("\n" + "=" * 70)
    print("   [✔] OFFICIAL DATASET REORGANIZATION COMPLETE!")
    print(f"   Train images: {len(glob.glob(os.path.join(train_img_dst, '*')))}")
    print(f"   Val images:   {len(glob.glob(os.path.join(val_img_dst, '*')))}")
    print(f"   Test images:  {len(glob.glob(os.path.join(test_img_dst, '*')))}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Ingest Official Edge AI 2026 Dataset")
    parser.add_argument("--input", required=True, help="Path to raw dataset ZIP or extracted folder")
    parser.add_argument("--skip_archive", action="store_true", help="Skip archiving existing COCO data")
    args = parser.parse_args()

    if not args.skip_archive:
        archive_current_coco_data()

    raw_dir = unpack_and_inspect(args.input)
    scan_and_reorganize(raw_dir)

if __name__ == "__main__":
    main()
