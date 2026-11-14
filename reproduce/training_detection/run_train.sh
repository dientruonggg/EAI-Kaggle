#!/bin/bash
# ==============================================================================
# run_train.sh - Full-6K TAL + NWD Training Execution Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -f "/home/dutai/EAI-2026/.venv_torch/bin/python" ]; then
    PYTHON_BIN="/home/dutai/EAI-2026/.venv_torch/bin/python"
elif [ -f "$PROJECT_ROOT/.venv_torch/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_torch/bin/python"
elif [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

echo "=== DTNet Full-6K TAL + NWD Training Runner ==="
echo "Python binary: $PYTHON_BIN"
echo "Project root:  $PROJECT_ROOT"

# Strict parameter invariant check
$PYTHON_BIN -c "
from dtnet_nwd_tal_full6k.dtnet_max import DTNetMax, make_spec_max, count_params_strict, MAX_ALLOWED_PARAMS
p = count_params_strict(DTNetMax(make_spec_max()))
print(f'[Pre-flight Gate] DTNetMax Strict Params: {p:,} / {MAX_ALLOWED_PARAMS:,} (Margin: {MAX_ALLOWED_PARAMS - p:,})')
assert p <= MAX_ALLOWED_PARAMS, 'Param limit exceeded!'
"

OUTPUT_DIR="qualifiers/weights/dtnet_nwd_tal_full6k"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"

if [ -f "$OUTPUT_DIR/latest_dtnet_nwd_tal_full6k.pt" ]; then
    INIT_WEIGHTS="$OUTPUT_DIR/latest_dtnet_nwd_tal_full6k.pt"
elif [ -f "qualifiers/weights/dtnet_clean_p2_200ep/best_dtnet_clean_200ep.pt" ]; then
    INIT_WEIGHTS="qualifiers/weights/dtnet_clean_p2_200ep/best_dtnet_clean_200ep.pt"
else
    INIT_WEIGHTS="qualifiers/weights/dtnet_p2_ultra_finetune/best_p2_ultra_finetune.pt"
fi

echo "Initial weights: $INIT_WEIGHTS"
echo "Log file:        $LOG_FILE"
echo "=========================================================="
echo "Launching nohup background training (200 epochs Full-6K)..."

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" dtnet_nwd_tal_full6k/train.py \
    --epochs 200 \
    --batch_size 64 \
    --lr 1.2e-4 \
    --min_lr 2e-6 \
    --w_box 5.5 \
    --w_nwd 2.0 \
    --w_neg 1.5 \
    --w_mutual 3.0 \
    --close_mosaic 50 \
    --init_weights "$INIT_WEIGHTS" \
    --data_dir "qualifiers/data" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 4 >> "$LOG_FILE" 2>&1 &

PID=$!
echo "[✔] Training started with PID: $PID"
echo "[✔] To monitor live logs, run:"
echo "    tail -f $LOG_FILE"
