#!/usr/bin/env bash
set -e
cd /home/dutai/EAI-2026

OUT_DIR="/home/dutai/EAI-2026/qualifiers/weights/dtnet_p2_ultra_finetune"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/train.log"

if pgrep -f "train_dtnet_p2_ultra_finetune.py" > /dev/null; then
    echo "[!] Warning: train_dtnet_p2_ultra_finetune.py is already running!"
    pgrep -fl "train_dtnet_p2_ultra_finetune.py"
    exit 1
fi

echo "================================================================================" >> "$LOG_FILE"
echo "  Starting DTNet-P2-Ultra 100-Epoch Adaptive Fine-Tuning at $(date)" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"

nohup /home/dutai/EAI-2026/.venv_torch/bin/python -u \
  /home/dutai/EAI-2026/qualifiers/train_dtnet_p2_ultra_finetune.py \
  --epochs 100 \
  --batch_size 64 \
  --lr 3.5e-4 \
  --w_box 6.0 \
  --w_neg 1.5 \
  --iou_gate 0.48 \
  --num_workers 2 \
  --resume /home/dutai/EAI-2026/qualifiers/weights/dtnet_p2_ultra/best_p2_ultra.pt \
  --data_dir /home/dutai/EAI-2026/qualifiers/data \
  --output_dir "$OUT_DIR" \
  "$@" \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "Process launched with PID: $PID"
echo "Process launched with PID: $PID" >> "$LOG_FILE"
echo "Monitoring command: ssh -t dutai-bkmaker \"tail -f $LOG_FILE\""
