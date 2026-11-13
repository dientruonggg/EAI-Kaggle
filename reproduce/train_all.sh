#!/usr/bin/env bash
# ==============================================================================
# EDGE AI CHALLENGE 2026 (QUALIFIERS) - COMPLETE TRAINING & REPRODUCTION SCRIPT
# ==============================================================================
# 1-Click execution script to train both models from scratch and export submission.csv.
#
# Hardware Targets: ESP32-S3 Microcontroller (<200k params, <1.5MB INT8 Flash)
# Results: DTNet-P2 (Detection 98%+ mAP50) + TextCNN (Intent 100% Accuracy)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================================================"
echo "   EDGE AI CHALLENGE 2026 - FULL REPRODUCTION PIPELINE (TRAIN -> EXPORT -> SUBMIT)"
echo "================================================================================"

EXTRA_FLAGS=()
if [ "$1" == "--scratch" ]; then
    EXTRA_FLAGS+=("--scratch")
    echo "[*] Mode: Pure SCRATCH Training (zero pretrained weights)"
fi

echo ""
echo "[1/3] Training Vietnamese Question Intent TextCNN (ESP32-S3 INT8)..."
python3 "$SCRIPT_DIR/train_intent.py" --output_dir "$SCRIPT_DIR/weights"

echo ""
echo "[2/3] Training Object Detection DTNet-NWD-TAL (MobileNetV3 + P2 skip, 197k params <= 200,000)..."
python3 "$SCRIPT_DIR/train_detection.py" --output_dir "$SCRIPT_DIR/weights" "${EXTRA_FLAGS[@]}"

echo ""
echo "[3/3] Generating final submission.csv with Sweetspot Calibrated Thresholds..."
python3 "$SCRIPT_DIR/run_reproduce.py" --output "$REPO_ROOT/submission.csv"

echo ""
echo "================================================================================"
echo "   [★ SUCCESS] Complete reproduction finished successfully!"
echo "   Generated submission: $REPO_ROOT/submission.csv"
echo "================================================================================"
