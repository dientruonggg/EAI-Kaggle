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

LOG_FILE="$PROJECT_ROOT/candidate_submissions/parallel_optimizer.log"
echo "[*] Launching Multi-Worker Parallel Optimizer with $PYTHON"
exec "$PYTHON" "$PROJECT_ROOT/src/pipeline/parallel_optimizer_pool.py" "$@" 2>&1 | tee -a "$LOG_FILE"
