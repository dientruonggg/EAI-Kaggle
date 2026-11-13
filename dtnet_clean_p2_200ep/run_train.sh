#!/bin/bash
# ==============================================================================
# run_train.sh - Clean-Phase 200-Epoch Fine-Tuning Execution Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Auto-detect Python Environment
if [ -f "/home/dutai/EAI-2026/.venv_torch/bin/python" ]; then
    PYTHON_BIN="/home/dutai/EAI-2026/.venv_torch/bin/python"
elif [ -f "$PROJECT_ROOT/.venv_torch/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_torch/bin/python"
elif [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

echo "=== DTNet Clean-Phase 200-Epoch Fine-Tuning Runner ==="
echo "Python binary: $PYTHON_BIN"
echo "Project root:  $PROJECT_ROOT"

# Hardware verification check
$PYTHON_BIN -c "
from dtnet_clean_p2_200ep.dtnet_max import DTNetMax, make_spec_max, count_params_strict, MAX_ALLOWED_PARAMS
p = count_params_strict(DTNetMax(make_spec_max()))
print(f'[Pre-flight Check] DTNetMax params: {p:,} / {MAX_ALLOWED_PARAMS:,} (Strict Margin: {MAX_ALLOWED_PARAMS - p:,})')
assert p <= MAX_ALLOWED_PARAMS, 'Param limit exceeded!'
"

OUTPUT_DIR="qualifiers/weights/dtnet_clean_p2_200ep"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"

INIT_WEIGHTS="qualifiers/weights/dtnet_p2_ultra_finetune/best_p2_ultra_finetune.pt"
if [ ! -f "$INIT_WEIGHTS" ]; then
    # Fallback to local / remote alternative if not present
    INIT_WEIGHTS="qualifiers/weights/dtnet_from_scratch_ultra/best_dtnet_ultra.pt"
fi

echo "Initial weights: $INIT_WEIGHTS"
echo "Log file:        $LOG_FILE"
echo "=========================================================="
echo "Launching nohup background training (200 epochs)..."

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" dtnet_clean_p2_200ep/train.py \
    --epochs 200 \
    --batch_size 64 \
    --lr 1.5e-4 \
    --min_lr 2e-6 \
    --w_box 5.5 \
    --w_neg 1.5 \
    --w_mutual 3.0 \
    --iou_gate 0.48 \
    --init_weights "$INIT_WEIGHTS" \
    --data_dir "qualifiers/data" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 4 > "$LOG_FILE" 2>&1 &

PID=$!
echo "[✔] Training started with PID: $PID"
echo "[✔] To monitor live logs, run:"
echo "    tail -f $LOG_FILE"
