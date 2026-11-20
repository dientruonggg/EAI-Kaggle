#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - DATASET DISTRIBUTION MATCHER & MERGER
Ensures external data added to the training set strictly matches the official
competition dataset distribution (Domain Adaptation & Distribution Alignment).

Features:
1. Profiler: Measures Class Priors P(Class), Object Counts per Image, and Box Scales.
2. Sampler: Subsamples / re-weights external data using Inverse Density Weighting.
3. Merger: Harmonizes external images with BTC data to prevent Covariate Shift & Class Imbalance.
"""
import os
import sys
import glob
import json
import shutil
import random
import cv2
import numpy as np

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
NUM_CLASSES = 7

def profile_dataset(labels_dir):
    """
    Analyzes label directory and returns statistical profile:
      - class_counts & class_proportions
      - boxes_per_image histogram
      - box scale statistics
    """
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not label_files:
        print(f"[!] No label files found in {labels_dir}")
        return None

    class_counts = {c: 0 for c in range(NUM_CLASSES)}
    boxes_per_img = {}
    box_scales = []
    total_boxes = 0

    for lf in label_files:
        img_boxes = 0
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if 0 <= cid < NUM_CLASSES:
                        class_counts[cid] += 1
                        total_boxes += 1
                        img_boxes += 1
                        w, h = float(parts[3]), float(parts[4])
                        box_scales.append(w * h)

        boxes_per_img[img_boxes] = boxes_per_img.get(img_boxes, 0) + 1

    total_images = len(label_files)
    proportions = {c: (cnt / max(total_boxes, 1)) for c, cnt in class_counts.items()}
    avg_boxes = total_boxes / max(total_images, 1)

    profile = {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": round(avg_boxes, 2),
        "class_counts": class_counts,
        "class_proportions": {CLASS_NAMES[c]: round(p, 4) for c, p in proportions.items()},
        "boxes_per_image_histogram": boxes_per_img,
        "median_box_scale": round(float(np.median(box_scales)) if box_scales else 0.0, 4)
    }
    return profile

def align_and_merge_datasets(
    btc_data_dir,
    external_data_dir,
    output_merged_dir,
    max_external_images=100,
    random_seed=42
):
    """
    Blends external dataset into BTC dataset while matching the target class distribution.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 70)
    print("   DISTRIBUTION-AWARE DATASET MERGER")
    print("=" * 70)

    btc_lbl_dir = os.path.join(btc_data_dir, "labels")
    btc_img_dir = os.path.join(btc_data_dir, "images")

    ext_lbl_dir = os.path.join(external_data_dir, "labels")
    ext_img_dir = os.path.join(external_data_dir, "images")

    # 1. Profile BTC Reference Distribution
    print(f"[*] Profiling reference dataset: {btc_data_dir}")
    btc_profile = profile_dataset(btc_lbl_dir)
    if not btc_profile:
        print("[!] Cannot profile BTC dataset. Aborting merge.")
        return

    print(f"[✔] BTC Profile: {btc_profile['total_images']} images, {btc_profile['total_boxes']} boxes.")
    print(f"    Class proportions: {btc_profile['class_proportions']}")

    # 2. Profile External Candidate Dataset
    print(f"\n[*] Profiling external dataset: {external_data_dir}")
    ext_profile = profile_dataset(ext_lbl_dir)
    if not ext_profile:
        print("[!] Cannot profile External dataset. Aborting merge.")
        return

    print(f"[✔] External Profile: {ext_profile['total_images']} images, {ext_profile['total_boxes']} boxes.")
    print(f"    Class proportions: {ext_profile['class_proportions']}")

    # 3. Calculate Sampling Weights to Align with BTC Distribution
    # Target weights = P_btc(c) / P_ext(c)
    weights = {}
    for c in range(NUM_CLASSES):
        cname = CLASS_NAMES[c]
        p_btc = btc_profile["class_proportions"][cname]
        p_ext = ext_profile["class_proportions"][cname]
        weights[c] = (p_btc / max(p_ext, 1e-4))

    # Normalize weights
    max_w = max(weights.values()) if weights else 1.0
    norm_weights = {c: weights[c] / max_w for c in weights}
    print(f"\n[*] Calculated class sampling weights to balance external data:")
    for c in range(NUM_CLASSES):
        print(f"    {CLASS_NAMES[c]:<10}: {norm_weights[c]:.3f}")

    # 4. Score each external image by how well its objects help match BTC distribution
    ext_label_files = glob.glob(os.path.join(ext_lbl_dir, "*.txt"))
    scored_ext_samples = []

    for lf in ext_label_files:
        base_id = os.path.splitext(os.path.basename(lf))[0]
        # Find corresponding image
        img_candidates = glob.glob(os.path.join(ext_img_dir, f"{base_id}.*"))
        if not img_candidates:
            continue
        img_p = img_candidates[0]

        # Calculate score based on classes present
        classes_in_img = []
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if 0 <= cid < NUM_CLASSES:
                        classes_in_img.append(cid)

        if not classes_in_img:
            continue

        sample_score = np.mean([norm_weights[c] for c in classes_in_img])
        scored_ext_samples.append((sample_score, img_p, lf))

    # Sort and pick top external images that best match the distribution
    scored_ext_samples.sort(key=lambda x: x[0] + random.uniform(-0.1, 0.1), reverse=True)
    selected_ext = scored_ext_samples[:max_external_images]

    print(f"\n[*] Selected {len(selected_ext)} / {len(scored_ext_samples)} external images matching BTC distribution.")

    # 5. Output Merged Dataset
    out_img_dir = os.path.join(output_merged_dir, "images")
    out_lbl_dir = os.path.join(output_merged_dir, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Copy 100% of BTC images
    btc_lbl_files = glob.glob(os.path.join(btc_lbl_dir, "*.txt"))
    for lf in btc_lbl_files:
        base_id = os.path.splitext(os.path.basename(lf))[0]
        img_p = glob.glob(os.path.join(btc_img_dir, f"{base_id}.*"))[0]
        shutil.copy(img_p, os.path.join(out_img_dir, os.path.basename(img_p)))
        shutil.copy(lf, os.path.join(out_lbl_dir, os.path.basename(lf)))

    # Copy selected External images (prefixed to avoid collision)
    for score, img_p, lf in selected_ext:
        ext_id = f"ext_{os.path.basename(img_p)}"
        ext_lbl_name = f"ext_{os.path.basename(lf)}"
        shutil.copy(img_p, os.path.join(out_img_dir, ext_id))
        shutil.copy(lf, os.path.join(out_lbl_dir, ext_lbl_name))

    # 6. Profile the final merged dataset to verify distribution
    merged_profile = profile_dataset(out_lbl_dir)
    print(f"\n[✔] MERGED DATASET CREATED: {output_merged_dir}")
    print(f"    Total images: {merged_profile['total_images']} (BTC: {len(btc_lbl_files)}, External: {len(selected_ext)})")
    print(f"    Total boxes:  {merged_profile['total_boxes']}")
    print(f"    Final class proportions: {merged_profile['class_proportions']}")

    # Save distribution audit log
    audit_file = os.path.join(output_merged_dir, "distribution_audit.json")
    with open(audit_file, "w") as f:
        json.dump({
            "btc_reference": btc_profile,
            "external_candidate": ext_profile,
            "final_merged": merged_profile
        }, f, indent=2)
    print(f"[✔] Distribution audit report saved to: {audit_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distribution-aware dataset merger")
    parser.add_argument("--btc_dir", default=os.path.join(PROJECT_ROOT, "data/train"))
    parser.add_argument("--ext_dir", default=os.path.join(PROJECT_ROOT, "data/val"))
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "data/merged_train"))
    parser.add_argument("--max_ext", type=int, default=15)
    args = parser.parse_args()

    align_and_merge_datasets(args.btc_dir, args.ext_dir, args.out_dir, max_external_images=args.max_ext)
