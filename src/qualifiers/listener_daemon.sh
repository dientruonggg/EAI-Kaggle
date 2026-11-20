#!/bin/bash
# ==============================================================================
# EDGE AI CHALLENGE 2026 - KICKOFF LISTENER DAEMON CONTROLLER
# Manages background audio recording and intelligent meeting note-taking.
# Usage:
#   ./listener_daemon.sh start [monitor|mic] [tiny|base|small]
#   ./listener_daemon.sh stop
#   ./listener_daemon.sh status
#   ./listener_daemon.sh logs
# ==============================================================================

PROJECT_DIR="/home/bim/Projects/EAI-2026"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
SCRIPT="$PROJECT_DIR/qualifiers/src/audio_listener/meeting_agent.py"
NOTES_DIR="$PROJECT_DIR/qualifiers/notes"
PID_FILE="$NOTES_DIR/daemon.pid"
LOG_FILE="$NOTES_DIR/daemon.log"
NOTES_MD="$NOTES_DIR/KICKOFF_NOTES.md"

SOURCE="${2:-monitor}" # default: speaker monitor
MODEL="${3:-base}"    # default: base

case "$1" in
  start)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "⚠️ Listener Daemon đang chạy rồi (PID: $(cat "$PID_FILE"))."
      exit 0
    fi
    mkdir -p "$NOTES_DIR"
    echo "🚀 Đang khởi động Kickoff Listener chạy nền (Background Daemon)..."
    echo " - Nguồn âm thanh: $SOURCE (monitor = loa laptop / meet / zoom, mic = micro)"
    echo " - Whisper Model: $MODEL"
    echo " - Bộ lọc: RMS Silence Gating + Silero VAD + Hallucination Filter"
    echo " - Ghi chú xuất tại: $NOTES_MD"

    nohup "$VENV_PY" "$SCRIPT" --source "$SOURCE" --model "$MODEL" > "$LOG_FILE" 2>&1 &
    DAEMON_PID=$!
    echo $DAEMON_PID > "$PID_FILE"
    sleep 2

    if kill -0 "$DAEMON_PID" 2>/dev/null; then
      echo "✅ Daemon đã khởi động thành công với PID: $DAEMON_PID"
      echo "👉 Xem log thời gian thực: ./qualifiers/listener_daemon.sh logs"
      echo "👉 Xem ghi chú Markdown: cat qualifiers/notes/KICKOFF_NOTES.md"
      echo "👉 Dừng chạy nền: ./qualifiers/listener_daemon.sh stop"
    else
      echo "❌ Khởi động thất bại. Kiểm tra log tại: $LOG_FILE"
      cat "$LOG_FILE"
    fi
    ;;

  stop)
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      if kill -0 "$PID" 2>/dev/null; then
        echo "🛑 Đang gửi tín hiệu dừng tới PID: $PID (đợi lưu nốt các ghi chú cuối)..."
        kill -INT "$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
        sleep 2
        rm -f "$PID_FILE"
        echo "✅ Daemon đã dừng an toàn."
      else
        echo "⚠️ PID $PID không còn hoạt động. Đang dọn dẹp file PID."
        rm -f "$PID_FILE"
      fi
    else
      # Fallback kill by process name if pid file was lost
      PKILL_MATCH=$(pgrep -f "meeting_agent.py")
      if [ -n "$PKILL_MATCH" ]; then
        echo "🛑 Đang dừng meeting_agent.py (PID: $PKILL_MATCH)..."
        pkill -INT -f "meeting_agent.py"
        echo "✅ Đã dừng tiến trình."
      else
        echo "ℹ️ Không có Listener Daemon nào đang chạy."
      fi
    fi
    ;;

  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "🟢 Trạng thái: ĐANG CHẠY NỀN (PID: $(cat "$PID_FILE"))"
      echo "📝 File ghi chú: $NOTES_MD"
      echo "--- 5 dòng ghi chú mới nhất ---"
      tail -n 10 "$NOTES_MD" | grep -v "^#"
    else
      echo "⚪ Trạng thái: ĐANG DỪNG"
    fi
    ;;

  logs)
    if [ -f "$LOG_FILE" ]; then
      tail -f -n 25 "$LOG_FILE"
    else
      echo "Chưa có file log tại $LOG_FILE"
    fi
    ;;

  *)
    echo "Sử dụng: $0 {start|stop|status|logs} [monitor|mic] [tiny|base|small]"
    exit 1
    ;;
esac
