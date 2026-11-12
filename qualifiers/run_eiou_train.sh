#!/usr/bin/env bash
set -e
cd /home/dutai/EAI-2026

OUT_DIR="/home/dutai/EAI-2026/qualifiers/weights/dtnet_p2_eiou"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/train.log"

# Check if training is already running
if pgrep -f "train_dtnet_p2_eiou.py" > /dev/null; then
    echo "[!] Warning: train_dtnet_p2_eiou.py is already running!"
    pgrep -fl "train_dtnet_p2_eiou.py"
    exit 1
fi

echo "================================================================================" > "$LOG_FILE"
echo "  Starting DTNet-P2 EIoU & Gated Hard-Negative Training at $(date)" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"

# Launch 400-epoch training from scratch with patience 35 in background with unbuffered python
nohup /home/dutai/EAI-2026/.venv_torch/bin/python -u \
  /home/dutai/EAI-2026/qualifiers/train_dtnet_p2_eiou.py \
  --epochs 400 \
  --patience 35 \
  --batch_size 64 \
  --w_box 5.5 \
  --w_neg 1.5 \
  --iou_gate 0.48 \
  --data_dir /home/dutai/EAI-2026/qualifiers/data \
  --output_dir "$OUT_DIR" \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "Process launched with PID: $PID"
echo "Process launched with PID: $PID" >> "$LOG_FILE"
echo "Monitoring command: tail -f $LOG_FILE"
