#!/usr/bin/env bash
# ==============================================================================
# run_train.sh - Full-Topping DTNetMax From-Scratch Training Runner
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Auto-detect Python binary (prioritize torch environment)
if [ -f "$REPO_ROOT/.venv_torch/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv_torch/bin/python"
elif [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="$(which python3)"
fi

OUT_DIR="$REPO_ROOT/qualifiers/weights/dtnet_from_scratch_ultra"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/train.log"

if pgrep -f "dtnet_from_scratch_ultra/train.py" > /dev/null; then
    echo "[!] Warning: dtnet_from_scratch_ultra/train.py is already running!"
    pgrep -fl "dtnet_from_scratch_ultra/train.py"
    exit 1
fi

echo "================================================================================" | tee -a "$LOG_FILE"
echo "  Starting DTNet-From-Scratch-Ultra (Full-Topping) Training at $(date)" | tee -a "$LOG_FILE"
echo "  Python: $PYTHON" | tee -a "$LOG_FILE"
echo "  Output Directory: $OUT_DIR" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

nohup "$PYTHON" -u "$SCRIPT_DIR/train.py" \
  --epochs 400 \
  --batch_size 64 \
  --lr 2e-3 \
  --w_box 5.5 \
  --w_neg 1.5 \
  --iou_gate 0.48 \
  --close_mosaic 35 \
  --num_workers 4 \
  --data_dir "$REPO_ROOT/qualifiers/data" \
  --output_dir "$OUT_DIR" \
  "$@" \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "[✔] Process launched with PID: $PID"
echo "[✔] Live log: tail -f $LOG_FILE"
