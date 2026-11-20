#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

REPO_ROOT="$(dirname "$PROJECT_ROOT")"
if [ "$(basename "$REPO_ROOT")" = src ]; then
    REPO_ROOT="$(dirname "$REPO_ROOT")"
fi

VENV_PATH="/home/dutai/EAI-2026/.venv"
if [ ! -d "$VENV_PATH" ]; then
    VENV_PATH="$REPO_ROOT/.venv"
fi
PYTHON="$VENV_PATH/bin/python"

LOG_DIR="$PROJECT_ROOT/candidate_submissions"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deep_train_convergence.log"

echo "[*] Launching Deep Convergence Training (400 Epoch Horizon | Patience = 10)"
echo "[*] Python Environment: $PYTHON"
echo "[*] Live Log Destination: $LOG_FILE"
echo ""

exec "$PYTHON" "$PROJECT_ROOT/src/pipeline/deep_train_convergence.py" "$@" 2>&1 | tee -a "$LOG_FILE"
