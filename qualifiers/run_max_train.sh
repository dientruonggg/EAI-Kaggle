#!/usr/bin/env bash
set -e
cd /home/dutai/EAI-2026

OUT_DIR="/home/dutai/EAI-2026/qualifiers/weights/dtnet_p2_max"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/train.log"

if pgrep -f "train_dtnet_p2_max.py" > /dev/null; then
    echo "[!] Warning: train_dtnet_p2_max.py is already running!"
    pgrep -fl "train_dtnet_p2_max.py"
    exit 1
fi

echo "================================================================================" >> "$LOG_FILE"
echo "  Starting/Resuming DTNet-P2-Max Training at $(date)" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"

nohup /home/dutai/EAI-2026/.venv_torch/bin/python -u \
  /home/dutai/EAI-2026/qualifiers/train_dtnet_p2_max.py \
  --epochs 400 \
  --patience 35 \
  --batch_size 64 \
  --w_box 5.5 \
  --w_neg 1.5 \
  --iou_gate 0.48 \
  --num_workers 2 \
  --data_dir /home/dutai/EAI-2026/qualifiers/data \
  --output_dir "$OUT_DIR" \
  "$@" \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "Process launched with PID: $PID"
echo "Process launched with PID: $PID" >> "$LOG_FILE"
echo "Monitoring command: tail -f $LOG_FILE"
