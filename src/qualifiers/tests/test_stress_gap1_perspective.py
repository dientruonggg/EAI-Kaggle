#!/usr/bin/env python3
"""
Adversarial Stress Test Suite: Gap 1 — Tabletop Perspective Augmentation
Empirical verification of qualifiers/src/detection/dataset.py:apply_tabletop_perspective

Tests:
1. Extreme pitch angles (limits, negative, extreme high, inverted range).
2. Degenerate, zero-area, and out-of-boundary input boxes.
3. High volume fuzzing (500 randomized trials).
"""

import sys
import os
import numpy as np

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.dataset import apply_tabletop_perspective

def assert_box_invariants(boxes, img_w, img_h, context=""):
    for idx, (cid, cx, cy, bw, bh) in enumerate(boxes):
        assert np.isfinite(cx), f"[{context}] box {idx} cx is not finite: {cx}"
        assert np.isfinite(cy), f"[{context}] box {idx} cy is not finite: {cy}"
        assert np.isfinite(bw), f"[{context}] box {idx} bw is not finite: {bw}"
        assert np.isfinite(bh), f"[{context}] box {idx} bh is not finite: {bh}"

        assert 0.0 <= cx <= 1.0, f"[{context}] box {idx} cx out of bounds: {cx}"
        assert 0.0 <= cy <= 1.0, f"[{context}] box {idx} cy out of bounds: {cy}"
        assert 0.0 < bw <= 1.0, f"[{context}] box {idx} bw out of bounds: {bw}"
        assert 0.0 < bh <= 1.0, f"[{context}] box {idx} bh out of bounds: {bh}"

        x1 = cx - bw / 2.0
        x2 = cx + bw / 2.0
        y1 = cy - bh / 2.0
        y2 = cy + bh / 2.0

        # Non-inversion invariant
        assert x2 >= x1, f"[{context}] box {idx} x-inverted: x1={x1}, x2={x2}"
        assert y2 >= y1, f"[{context}] box {idx} y-inverted: y1={y1}, y2={y2}"

        # Coordinate bounds (with small epsilon for float precision)
        assert x1 >= -1e-5, f"[{context}] box {idx} x1 < 0: {x1}"
        assert y1 >= -1e-5, f"[{context}] box {idx} y1 < 0: {y1}"
        assert x2 <= 1.0 + 1e-5, f"[{context}] box {idx} x2 > 1: {x2}"
        assert y2 <= 1.0 + 1e-5, f"[{context}] box {idx} y2 > 1: {y2}"

        # Degenerate filtering invariant: width and height must be >= 4.0 pixels
        pixel_w = bw * float(img_w)
        pixel_h = bh * float(img_h)
        assert pixel_w >= 3.99, f"[{context}] box {idx} width < 4 px: {pixel_w}"
        assert pixel_h >= 3.99, f"[{context}] box {idx} height < 4 px: {pixel_h}"


def test_pitch_angles():
    print(">>> [Test 1] Running Pitch Angle Stress Tests...")
    img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    boxes = [
        (0, 0.5, 0.5, 0.2, 0.2),
        (1, 0.2, 0.2, 0.1, 0.1),
        (2, 0.8, 0.8, 0.15, 0.15),
    ]

    pitch_test_configs = [
        ("standard", (30.0, 60.0)),
        ("min_boundary", (30.0, 30.0)),
        ("max_boundary", (60.0, 60.0)),
        ("single_angle_45", (45.0, 45.0)),
        ("negative_pitch", (-45.0, -10.0)),
        ("zero_pitch", (0.0, 0.0)),
        ("low_pitch", (0.0, 15.0)),
        ("high_pitch_75", (70.0, 80.0)),
        ("extreme_high_89", (85.0, 89.0)),
        ("inverted_range", (60.0, 30.0)),
        ("extreme_inverted", (90.0, 10.0)),
    ]

    for name, p_range in pitch_test_configs:
        for trial in range(10):
            w_img, w_boxes = apply_tabletop_perspective(img, boxes, pitch_range=p_range, prob=1.0)
            assert w_img.shape == img.shape, f"Shape mismatch in {name}: {w_img.shape}"
            assert w_img.dtype == img.dtype, f"Dtype mismatch in {name}: {w_img.dtype}"
            assert_box_invariants(w_boxes, 320, 240, context=f"Pitch_{name}_trial_{trial}")

    print("  [✔] Test 1: Pitch Angle Stress Tests Passed (110 trials).")


def test_degenerate_and_boundary_boxes():
    print(">>> [Test 2] Running Degenerate & Boundary Box Tests...")
    img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    adversarial_box_sets = [
        ("tiny_1px", [(0, 0.5, 0.5, 1.0 / 320.0, 1.0 / 240.0)]),
        ("subpixel", [(0, 0.5, 0.5, 0.1 / 320.0, 0.1 / 240.0)]),
        ("zero_size", [(0, 0.5, 0.5, 0.0, 0.0)]),
        ("zero_width", [(0, 0.5, 0.5, 0.0, 0.2)]),
        ("zero_height", [(0, 0.5, 0.5, 0.2, 0.0)]),
        ("top_left_corner", [(0, 0.0, 0.0, 0.1, 0.1)]),
        ("top_right_corner", [(1, 1.0, 0.0, 0.1, 0.1)]),
        ("bottom_left_corner", [(2, 0.0, 1.0, 0.1, 0.1)]),
        ("bottom_right_corner", [(3, 1.0, 1.0, 0.1, 0.1)]),
        ("out_of_bounds_neg", [(0, -0.5, -0.5, 0.2, 0.2)]),
        ("out_of_bounds_pos", [(0, 1.5, 1.5, 0.2, 0.2)]),
        ("oversized_box", [(0, 0.5, 0.5, 2.0, 2.0)]),
        ("inverted_width", [(0, 0.5, 0.5, -0.2, 0.2)]),
        ("inverted_height", [(0, 0.5, 0.5, 0.2, -0.2)]),
        ("empty_list", []),
    ]

    for name, box_list in adversarial_box_sets:
        for trial in range(10):
            w_img, w_boxes = apply_tabletop_perspective(img, box_list, pitch_range=(30.0, 60.0), prob=1.0)
            assert w_img.shape == img.shape
            assert_box_invariants(w_boxes, 320, 240, context=f"Degenerate_{name}_trial_{trial}")

    # Edge cases on image input
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    e_img, e_boxes = apply_tabletop_perspective(empty_img, [(0, 0.5, 0.5, 0.2, 0.2)], prob=1.0)
    assert e_img.size == 0

    none_img, n_boxes = apply_tabletop_perspective(None, [(0, 0.5, 0.5, 0.2, 0.2)], prob=1.0)
    assert none_img is None

    # Single pixel image
    one_px_img = np.zeros((1, 1, 3), dtype=np.uint8)
    o_img, o_boxes = apply_tabletop_perspective(one_px_img, [(0, 0.5, 0.5, 0.5, 0.5)], prob=1.0)
    assert o_img.shape == (1, 1, 3)

    # Float32 normalized image
    flt_img = np.random.uniform(0.0, 1.0, (160, 160, 3)).astype(np.float32)
    f_img, f_boxes = apply_tabletop_perspective(flt_img, [(0, 0.5, 0.5, 0.3, 0.3)], prob=1.0)
    assert f_img.shape == (160, 160, 3)
    assert f_img.dtype == np.float32
    assert_box_invariants(f_boxes, 160, 160, context="Float32_image")

    print("  [✔] Test 2: Degenerate & Boundary Box Tests Passed (150+ trials).")


def test_high_volume_fuzzing(num_trials=500):
    print(f">>> [Test 3] Running High Volume Fuzzing ({num_trials} randomized trials)...")
    np.random.seed(42)

    sizes = [(160, 160), (240, 320), (128, 128), (200, 200)]

    for trial in range(num_trials):
        h, w = sizes[trial % len(sizes)]
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        # Generate 0 to 15 random boxes with arbitrary coords
        num_boxes = np.random.randint(0, 15)
        boxes = []
        for _ in range(num_boxes):
            cid = np.random.randint(0, 7)
            # Fuzz coordinates: normal, negative, exceeding 1, tiny, huge
            mode = np.random.choice(["normal", "corner", "out_of_bounds", "tiny", "huge"])
            if mode == "normal":
                cx = np.random.uniform(0.1, 0.9)
                cy = np.random.uniform(0.1, 0.9)
                bw = np.random.uniform(0.05, 0.5)
                bh = np.random.uniform(0.05, 0.5)
            elif mode == "corner":
                cx = np.random.choice([0.0, 1.0])
                cy = np.random.choice([0.0, 1.0])
                bw = np.random.uniform(0.02, 0.3)
                bh = np.random.uniform(0.02, 0.3)
            elif mode == "out_of_bounds":
                cx = np.random.uniform(-0.5, 1.5)
                cy = np.random.uniform(-0.5, 1.5)
                bw = np.random.uniform(0.05, 0.4)
                bh = np.random.uniform(0.05, 0.4)
            elif mode == "tiny":
                cx = np.random.uniform(0.0, 1.0)
                cy = np.random.uniform(0.0, 1.0)
                bw = np.random.uniform(0.0, 4.0 / w)
                bh = np.random.uniform(0.0, 4.0 / h)
            else: # huge
                cx = np.random.uniform(0.2, 0.8)
                cy = np.random.uniform(0.2, 0.8)
                bw = np.random.uniform(0.8, 2.0)
                bh = np.random.uniform(0.8, 2.0)

            boxes.append((cid, cx, cy, bw, bh))

        p_min = np.random.uniform(20.0, 40.0)
        p_max = np.random.uniform(50.0, 70.0)

        w_img, w_boxes = apply_tabletop_perspective(img, boxes, pitch_range=(p_min, p_max), prob=1.0)

        assert w_img.shape == (h, w, 3), f"Trial {trial}: Shape changed from {(h, w, 3)} to {w_img.shape}"
        assert np.all(np.isfinite(w_img)), f"Trial {trial}: Non-finite pixel found in warped image"
        assert_box_invariants(w_boxes, w, h, context=f"Fuzz_trial_{trial}")

    print(f"  [✔] Test 3: High Volume Fuzzing Passed ({num_trials} trials, 0 failures).")


if __name__ == "__main__":
    print("=" * 60)
    print("ADVERSARIAL STRESS TEST: GAP 1 (PERSPECTIVE AUGMENTATION)")
    print("=" * 60)
    test_pitch_angles()
    test_degenerate_and_boundary_boxes()
    test_high_volume_fuzzing(500)
    print("=" * 60)
    print("ALL GAP 1 STRESS TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)
