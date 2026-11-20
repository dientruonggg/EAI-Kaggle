#!/bin/bash
# ==============================================================================
# EDGE AI CHALLENGE 2026 - KICKOFF MEETING LISTENER & NOTE-TAKING RUNNER
# ==============================================================================

PROJECT_DIR="/home/bim/Projects/EAI-2026"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
SCRIPT="$PROJECT_DIR/qualifiers/src/audio_listener/meeting_agent.py"
NOTES_MD="$PROJECT_DIR/qualifiers/notes/KICKOFF_NOTES.md"

SOURCE="${1:-monitor}" # default to speaker monitor (online meeting audio)
MODEL="${2:-base}"    # tiny, base, or small

echo "======================================================================"
echo " 🎙️ AGENT 1: KICKOFF AUDIO LISTENER & LIVE NOTE-TAKER"
echo "======================================================================"
echo " - Nguồn âm thanh: $SOURCE (monitor = loa laptop / meet / zoom, mic = micro)"
echo " - Kích thước model Whisper: $MODEL"
echo " - File ghi chú đầu ra: $NOTES_MD"
echo "======================================================================"
echo "👉 Đang khởi chạy tiến trình lắng nghe..."

exec "$VENV_PY" "$SCRIPT" --source "$SOURCE" --model "$MODEL"
