#!/bin/bash
# ==============================================================================
# run_resume_300ep.sh - Resume Full-6K TAL + NWD Training from Epoch 201 to 300
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

echo "=== DTNet Full-6K TAL + NWD (Resume to 300 Epochs) Runner ==="
echo "Python binary: $PYTHON_BIN"
echo "Project root:  $PROJECT_ROOT"

OUTPUT_DIR="qualifiers/weights/dtnet_nwd_tal_full6k"
INIT_WEIGHTS="$OUTPUT_DIR/latest_dtnet_nwd_tal_full6k.pt"
LOG_FILE="$OUTPUT_DIR/train_resume_300ep.log"

if [ ! -f "$INIT_WEIGHTS" ]; then
    INIT_WEIGHTS="$OUTPUT_DIR/best_dtnet_nwd_tal_full6k.pt"
fi

echo "Initial weights: $INIT_WEIGHTS"
echo "Log file:        $LOG_FILE"
echo "=========================================================="
echo "Launching resume training: Epochs 201 -> 300 on Clean-Phase..."

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" dtnet_nwd_tal_full6k/train.py \
    --epochs 300 \
    --batch_size 64 \
    --lr 1.2e-4 \
    --min_lr 2e-6 \
    --w_box 5.5 \
    --w_nwd 2.0 \
    --w_neg 1.5 \
    --w_mutual 3.0 \
    --close_mosaic 150 \
    --init_weights "$INIT_WEIGHTS" \
    --data_dir "qualifiers/data" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 4 >> "$LOG_FILE" 2>&1 &

PID=$!
echo "[✔] Resume training started with PID: $PID"
echo "[✔] Logging to: $LOG_FILE"
