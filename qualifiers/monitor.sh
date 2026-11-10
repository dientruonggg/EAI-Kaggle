#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# Option 1: Direct Tmux Attach (0% CPU, 0 disk I/O, instant live stream)
if [ "$1" == "--tmux" ] || [ "$1" == "--deep" ]; then
    if [ ! -d "/home/dutai" ]; then
        echo "[*] Attaching directly to active remote tmux session on dutai-bkmaker..."
        echo "[*] (Press Ctrl+B then D to detach safely without stopping training)"
        exec ssh -t dutai-bkmaker "tmux attach -t eai_ultrav3 2>/dev/null || tmux attach -t eai_deep_p30 2>/dev/null || tmux attach -t eai_weak_boost 2>/dev/null || tmux attach -t eai_deep_train 2>/dev/null || tmux attach -t eai_parallel_pool"
    else
        TMUX_TARGET="eai_ultrav3"
        if ! tmux has-session -t "$TMUX_TARGET" 2>/dev/null; then
            TMUX_TARGET="eai_deep_p30"
            if ! tmux has-session -t "$TMUX_TARGET" 2>/dev/null; then
                TMUX_TARGET="eai_weak_boost"
                if ! tmux has-session -t "$TMUX_TARGET" 2>/dev/null; then
                    TMUX_TARGET="eai_deep_train"
                    if ! tmux has-session -t "$TMUX_TARGET" 2>/dev/null; then
                        TMUX_TARGET="eai_parallel_pool"
                    fi
                fi
            fi
        fi
        echo "[*] Attaching to local tmux session '$TMUX_TARGET'..."
        exec tmux attach -t "$TMUX_TARGET"
    fi
fi

# Option 2: Live Log Stream (tail -f)
if [ "$1" == "--tail" ]; then
    LOG_TARGET="candidate_submissions/deep_train_convergence.log"
    if [ ! -f "$PROJECT_ROOT/$LOG_TARGET" ] || [ -f "$PROJECT_ROOT/candidate_submissions/parallel_optimizer.log" -a "$1" == "--pool" ]; then
        LOG_TARGET="candidate_submissions/parallel_optimizer.log"
    fi
    if [ ! -d "/home/dutai" ]; then
        echo "[*] Streaming live log ($LOG_TARGET) from dutai-bkmaker..."
        exec ssh -t dutai-bkmaker "tail -f -n 35 ~/EAI-2026/qualifiers/$LOG_TARGET"
    else
        exec tail -f -n 35 "$PROJECT_ROOT/$LOG_TARGET"
    fi
fi

# If on client machine without --local, connect to remote server
if [ ! -d "/home/dutai" ] && [ "$1" != "--local" ]; then
    echo "[*] Connecting to live Multi-Worker TUI on Server 2 (dutai-bkmaker)..."
    exec ssh -t dutai-bkmaker "cd ~/EAI-2026/qualifiers && ./monitor.sh $*"
fi

# Run Optimized TUI Monitor
VENV_PYTHON="/home/dutai/EAI-2026/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="$PROJECT_ROOT/../.venv/bin/python"
fi
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

exec "$VENV_PYTHON" "$PROJECT_ROOT/src/pipeline/tui_monitor.py" "$@"
