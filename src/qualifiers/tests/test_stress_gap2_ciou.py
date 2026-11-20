#!/usr/bin/env python3
"""
Adversarial Stress Test Suite: Gap 2 — CIoU Loss for Bounding Box Regression
Empirical verification of qualifiers/src/detection/losses.py:decoupled_ciou_loss

Tests:
1. Exact identical bounding boxes (IoU = 1.0, distance = 0.0 -> loss ~0.0).
2. Completely disjoint bounding boxes at opposite corners of the image.
3. Extreme aspect ratio discrepancies (w/h = 100 vs w/h = 0.01, w/h = 10,000 vs 1e-4).
4. Numerical stability: Zero NaNs, Zero Infs, strictly finite backprop gradients.
5. Background-only images (conf_mask = 0 everywhere, loss = 0.0, zero gradients).
6. Multi-scale grid test (16x16, 20x20, 24x24).
7. End-to-end model training integration with MicroEdgeCNN.
"""

import sys
import os
import numpy as np
import tensorflow as tf

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.losses import decoupled_ciou_loss, decoupled_box_loss, masked_smooth_l1_loss
from src.detection.model import build_hierarchical_12layer_detector


def test_identical_boxes():
    print(">>> [Test 1] Identical Bounding Boxes (IoU = 1.0, Dist = 0.0)...")
    batch_size = 4
    grid_size = 20

    # Create matching predictions and targets
    y_true_np = np.zeros((batch_size, grid_size, grid_size, 5), dtype=np.float32)
    y_pred_np = np.zeros((batch_size, grid_size, grid_size, 4), dtype=np.float32)

    # Populate several cells with objects
    cells = [(0, 5, 5), (1, 10, 10), (2, 0, 0), (3, 19, 19)]
    for b, gy, gx in cells:
        dx, dy, w, h = 0.5, 0.5, 0.3, 0.4
        y_true_np[b, gy, gx] = [dx, dy, w, h, 1.0] # conf_mask = 1.0
        y_pred_np[b, gy, gx] = [dx, dy, w, h]

    y_true = tf.constant(y_true_np)
    y_pred = tf.Variable(y_pred_np)

    with tf.GradientTape() as tape:
        loss = decoupled_ciou_loss(y_true, y_pred)
    grad = tape.gradient(loss, y_pred)

    loss_val = float(loss.numpy())
    print(f"  Identical boxes loss: {loss_val:.8e}")
    assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
    assert abs(loss_val) < 1e-6, f"Identical boxes should have loss ~0.0, got {loss_val}"
    assert np.all(np.isfinite(grad.numpy())), "Gradients for identical boxes contain non-finite values"

    # Also check aliases
    alias_loss1 = float(decoupled_box_loss(y_true, y_pred).numpy())
    alias_loss2 = float(masked_smooth_l1_loss(y_true, y_pred).numpy())
    assert abs(alias_loss1) < 1e-6
    assert abs(alias_loss2) < 1e-6
    print("  [✔] Test 1: Identical Bounding Boxes Passed.")


def test_disjoint_opposite_corners():
    print(">>> [Test 2] Disjoint Opposite Corner Bounding Boxes...")
    batch_size = 2
    grid_size = 20

    y_true_np = np.zeros((batch_size, grid_size, grid_size, 5), dtype=np.float32)
    y_pred_np = np.zeros((batch_size, grid_size, grid_size, 4), dtype=np.float32)

    # True box at top-left corner (0, 0)
    y_true_np[0, 0, 0] = [0.0, 0.0, 0.05, 0.05, 1.0]
    # Pred box predicts object at bottom-right corner (19, 19)
    y_pred_np[0, 0, 0] = [19.0, 19.0, 0.05, 0.05] # huge offset to opposite corner

    # True box at bottom-left corner (19, 0)
    y_true_np[1, 19, 0] = [0.0, 0.0, 0.02, 0.02, 1.0]
    # Pred box predicts at top-right corner (0, 19)
    y_pred_np[1, 19, 0] = [19.0, -19.0, 0.02, 0.02]

    y_true = tf.constant(y_true_np)
    y_pred = tf.Variable(y_pred_np)

    with tf.GradientTape() as tape:
        loss = decoupled_ciou_loss(y_true, y_pred)
    grad = tape.gradient(loss, y_pred)

    loss_val = float(loss.numpy())
    print(f"  Disjoint opposite boxes loss: {loss_val:.6f}")
    assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
    assert loss_val > 1.0, f"Disjoint boxes at opposite corners should have CIoU loss > 1.0, got {loss_val}"
    assert np.all(np.isfinite(grad.numpy())), "Gradients contain non-finite values"

    print("  [✔] Test 2: Disjoint Opposite Corner Boxes Passed.")


def test_extreme_aspect_ratios():
    print(">>> [Test 3] Extreme Aspect Ratio Discrepancies (w/h = 100 vs 0.01)...")
    batch_size = 4
    grid_size = 20

    aspect_ratio_cases = [
        # (true_w, true_h), (pred_w, pred_h), description
        ((1.0, 0.01), (0.01, 1.0), "w/h=100 vs w/h=0.01"),
        ((0.01, 1.0), (1.0, 0.01), "w/h=0.01 vs w/h=100"),
        ((1.0, 1e-4), (1e-4, 1.0), "w/h=10000 vs w/h=1e-4"),
        ((1e-4, 1.0), (1.0, 1e-4), "w/h=1e-4 vs w/h=10000"),
        ((1e-4, 1e-4), (1.0, 1.0), "tiny square vs full square"),
    ]

    for (tw, th), (pw, ph), desc in aspect_ratio_cases:
        y_true_np = np.zeros((batch_size, grid_size, grid_size, 5), dtype=np.float32)
        y_pred_np = np.zeros((batch_size, grid_size, grid_size, 4), dtype=np.float32)

        y_true_np[0, 5, 5] = [0.5, 0.5, tw, th, 1.0]
        y_pred_np[0, 5, 5] = [0.5, 0.5, pw, ph]

        y_true = tf.constant(y_true_np)
        y_pred = tf.Variable(y_pred_np)

        with tf.GradientTape() as tape:
            loss = decoupled_ciou_loss(y_true, y_pred)
        grad = tape.gradient(loss, y_pred)

        loss_val = float(loss.numpy())
        grad_np = grad.numpy()

        assert np.isfinite(loss_val), f"Loss not finite for {desc}: {loss_val}"
        assert loss_val >= 0.0, f"Loss is negative for {desc}: {loss_val}"
        assert not np.any(np.isnan(grad_np)), f"NaN gradient encountered for {desc}"
        assert not np.any(np.isinf(grad_np)), f"Inf gradient encountered for {desc}"
        assert np.all(np.isfinite(grad_np)), f"Non-finite gradient for {desc}"

    print("  [✔] Test 3: Extreme Aspect Ratio Discrepancies Passed (All finite gradients, 0 NaNs).")


def test_numerical_stability_fuzzing(num_trials=200):
    print(f">>> [Test 4] Numerical Stability & Random Fuzzing ({num_trials} trials)...")
    np.random.seed(1337)

    for trial in range(num_trials):
        bs = np.random.choice([1, 2, 4])
        gs = np.random.choice([16, 18, 20])

        # Random predictions with arbitrary values (positive, negative, large)
        y_pred_np = np.random.uniform(-5.0, 5.0, (bs, gs, gs, 4)).astype(np.float32)

        # Random targets with arbitrary conf_masks
        y_true_np = np.zeros((bs, gs, gs, 5), dtype=np.float32)
        # Randomly activate 0 to 10 objects
        num_objs = np.random.randint(0, 11)
        for _ in range(num_objs):
            b = np.random.randint(0, bs)
            gy = np.random.randint(0, gs)
            gx = np.random.randint(0, gs)
            # Random target bbox
            dx = np.random.uniform(-1.0, 2.0)
            dy = np.random.uniform(-1.0, 2.0)
            w = np.random.uniform(-0.1, 1.5)
            h = np.random.uniform(-0.1, 1.5)
            y_true_np[b, gy, gx] = [dx, dy, w, h, 1.0]

        y_true = tf.constant(y_true_np)
        y_pred = tf.Variable(y_pred_np)

        with tf.GradientTape() as tape:
            loss = decoupled_ciou_loss(y_true, y_pred)
        grad = tape.gradient(loss, y_pred)

        loss_val = float(loss.numpy())
        grad_np = grad.numpy()

        assert np.isfinite(loss_val), f"Trial {trial}: Loss is not finite: {loss_val}"
        assert loss_val >= 0.0, f"Trial {trial}: Loss is negative: {loss_val}"
        assert not np.any(np.isnan(grad_np)), f"Trial {trial}: NaN in gradients"
        assert not np.any(np.isinf(grad_np)), f"Trial {trial}: Inf in gradients"

    print(f"  [✔] Test 4: Numerical Stability Fuzzing Passed ({num_trials} trials, 0 NaNs, 0 Infs).")


def test_background_only_images():
    print(">>> [Test 5] Background-Only Images (conf_mask = 0 everywhere)...")
    batch_size = 4
    grid_size = 20

    y_true_np = np.zeros((batch_size, grid_size, grid_size, 5), dtype=np.float32) # conf_mask is 0 everywhere
    y_pred_np = np.random.uniform(-2.0, 2.0, (batch_size, grid_size, grid_size, 4)).astype(np.float32)

    y_true = tf.constant(y_true_np)
    y_pred = tf.Variable(y_pred_np)

    with tf.GradientTape() as tape:
        loss = decoupled_ciou_loss(y_true, y_pred)
    grad = tape.gradient(loss, y_pred)

    loss_val = float(loss.numpy())
    grad_np = grad.numpy()

    assert loss_val == 0.0, f"Loss should be exactly 0.0 for pure background, got {loss_val}"
    assert np.all(grad_np == 0.0), "Gradients should be strictly 0.0 for pure background"
    assert not np.any(np.isnan(grad_np)), "Gradients contain NaN on background-only image"

    print("  [✔] Test 5: Background-Only Images Passed (Loss = 0.0, Gradients = 0.0).")


def test_multiscale_grids():
    print(">>> [Test 6] Multi-Scale Grids (16x16, 18x18, 20x20, 24x24)...")
    grids = [16, 18, 20, 24]
    for g in grids:
        y_true_np = np.zeros((2, g, g, 5), dtype=np.float32)
        y_pred_np = np.zeros((2, g, g, 4), dtype=np.float32)

        y_true_np[0, g//2, g//2] = [0.5, 0.5, 0.2, 0.2, 1.0]
        y_pred_np[0, g//2, g//2] = [0.6, 0.4, 0.25, 0.18]

        y_true = tf.constant(y_true_np)
        y_pred = tf.Variable(y_pred_np)

        with tf.GradientTape() as tape:
            loss = decoupled_ciou_loss(y_true, y_pred)
        grad = tape.gradient(loss, y_pred)

        loss_val = float(loss.numpy())
        assert np.isfinite(loss_val) and loss_val > 0.0
        assert np.all(np.isfinite(grad.numpy()))

    print("  [✔] Test 6: Multi-Scale Grids Passed.")


def test_model_end_to_end_training():
    print(">>> [Test 7] End-to-End MicroEdgeCNN Training Integration...")
    model = build_hierarchical_12layer_detector(input_shape=(160, 160, 3), num_classes=7)

    # Verify param count constraint (< 200,000 parameters)
    total_params = model.count_params()
    print(f"  Model Parameter Count: {total_params} (Limit: < 200,000)")
    assert total_params < 200000, f"Model exceeds 200k param limit: {total_params}"

    from src.detection.losses import decoupled_conf_loss, decoupled_cls_loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'conf_output': decoupled_conf_loss,
            'cls_output': decoupled_cls_loss,
            'box_output': decoupled_ciou_loss,
        },
        loss_weights={
            'conf_output': 1.5,
            'cls_output': 1.5,
            'box_output': 2.5,
        }
    )

    # Run 5 training steps on synthetic batches
    for step in range(5):
        X = np.random.uniform(0, 1, (4, 160, 160, 3)).astype(np.float32)
        Y = {
            'conf_output': np.zeros((4, 20, 20, 1), dtype=np.float32),
            'cls_output': np.zeros((4, 20, 20, 8), dtype=np.float32),
            'box_output': np.zeros((4, 20, 20, 5), dtype=np.float32),
        }
        # Add 2 objects per batch item
        for b in range(4):
            Y['conf_output'][b, b*3+2, b*3+2, 0] = 1.0
            Y['cls_output'][b, b*3+2, b*3+2, b % 7] = 1.0
            Y['cls_output'][b, b*3+2, b*3+2, 7] = 1.0 # conf_mask
            Y['box_output'][b, b*3+2, b*3+2] = [0.5, 0.5, 0.2, 0.3, 1.0]

        history = model.train_on_batch(X, Y)
        # Check all losses are finite
        if isinstance(history, list):
            for l in history:
                assert np.isfinite(l) and l >= 0.0, f"Step {step}: Non-finite loss {l}"
        else:
            assert np.isfinite(history) and history >= 0.0

    print("  [✔] Test 7: End-to-End MicroEdgeCNN Training Integration Passed (5 steps).")


if __name__ == "__main__":
    print("=" * 60)
    print("ADVERSARIAL STRESS TEST: GAP 2 (CIOU LOSS)")
    print("=" * 60)
    test_identical_boxes()
    test_disjoint_opposite_corners()
    test_extreme_aspect_ratios()
    test_numerical_stability_fuzzing(200)
    test_background_only_images()
    test_multiscale_grids()
    test_model_end_to_end_training()
    print("=" * 60)
    print("ALL GAP 2 STRESS TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)
