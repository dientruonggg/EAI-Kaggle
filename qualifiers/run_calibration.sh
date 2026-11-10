#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

VENV_PYTHON="/home/dutai/EAI-2026/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="$PROJECT_ROOT/../.venv/bin/python"
fi

MODEL_PATH="$PROJECT_ROOT/weights/continuous_loop/trial_150/model_int8.tflite"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="$PROJECT_ROOT/weights/champion_ultrav2_best.keras"
fi

OUT_CSV="$PROJECT_ROOT/candidate_submissions/calibrated_trial_150_sweetspot.csv"
OUT_JSON="$PROJECT_ROOT/candidate_submissions/calibrated_trial_150_thresholds.json"
INTENT_CSV="$PROJECT_ROOT/submissions/intent_preds.csv"
if [ ! -f "$INTENT_CSV" ]; then
    INTENT_CSV="$PROJECT_ROOT/intent_preds.csv"
fi

echo "[*] Launching Per-Class Threshold Calibration on Champion Model..."
echo "[*] Model:       $MODEL_PATH"
echo "[*] Output CSV:  $OUT_CSV"
echo "[*] Output JSON: $OUT_JSON"
echo ""

exec "$VENV_PYTHON" "$PROJECT_ROOT/tune_sweetspot_calibration.py" \
    --model "$MODEL_PATH" \
    --val_images "$PROJECT_ROOT/data/val/images" \
    --val_labels "$PROJECT_ROOT/data/val/labels" \
    --test_images "$PROJECT_ROOT/data/test/images" \
    --intent_preds "$INTENT_CSV" \
    --output_csv "$OUT_CSV" \
    --output_json "$OUT_JSON" \
    "$@"
