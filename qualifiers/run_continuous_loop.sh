#!/usr/bin/env bash
# ==============================================================================
# EDGE AI CHALLENGE 2026 - CONTINUOUS OPTIMIZATION LOOP LAUNCHER
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUALIFIERS_DIR="$SCRIPT_DIR"
PROJECT_DIR="$(dirname "$QUALIFIERS_DIR")"

if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    VENV_PY="$PROJECT_DIR/.venv/bin/python"
    VENV_DIR="$PROJECT_DIR/.venv"
elif [ -f "$QUALIFIERS_DIR/.venv/bin/python" ]; then
    VENV_PY="$QUALIFIERS_DIR/.venv/bin/python"
    VENV_DIR="$QUALIFIERS_DIR/.venv"
else
    VENV_PY="$(which python3)"
    VENV_DIR=""
fi

# Automatically inject CUDA libraries for TensorFlow if using venv nvidia wheels
if [ -n "$VENV_DIR" ]; then
    NVIDIA_DIRS=$("$VENV_PY" -c "import glob; print(':'.join(glob.glob('$VENV_DIR/lib/python*/site-packages/nvidia/*/lib')))" 2>/dev/null || true)
    if [ -n "$NVIDIA_DIRS" ]; then
        export LD_LIBRARY_PATH="$NVIDIA_DIRS:${LD_LIBRARY_PATH:-}"
    fi
fi

# NVIDIA GPU execution stability & TFLM quantization compatibility
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

LOOP_SCRIPT="$QUALIFIERS_DIR/src/pipeline/continuous_optimizer_loop.py"
LOG_FILE="$QUALIFIERS_DIR/candidate_submissions/continuous_optimizer.log"

mkdir -p "$QUALIFIERS_DIR/candidate_submissions"

echo "=============================================================================="
echo "   LAUNCHING EDGE AI CONTINUOUS OPTIMIZATION LOOP"
echo "   Target: ESP32-S3 (< 200k params, Full INT8, <= 1.5MB)"
echo "   Environment Python: $VENV_PY"
echo "   Training Directory: $QUALIFIERS_DIR/data/train/ (Strictly isolated)"
echo "   Stress Test Suite:  $QUALIFIERS_DIR/data_stress_test/ (1,350 COCO images)"
echo "   Candidate Output:   $QUALIFIERS_DIR/candidate_submissions/"
echo "   Submission Policy:  HELD locally. NO automatic Kaggle push."
echo "=============================================================================="

export PROJECT_ROOT="$QUALIFIERS_DIR"
exec "$VENV_PY" "$LOOP_SCRIPT" "$@" 2>&1 | tee -a "$LOG_FILE"
