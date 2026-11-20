#!/usr/bin/env python3
"""
Lightweight continuous chunk recorder.
Runs in the background, recording 20s audio segments from the unmuted default microphone
so no meeting audio is lost while the ASR model finishes loading.
"""
import os
import time
import subprocess
from datetime import datetime

CHUNKS_DIR = "/home/bim/Projects/EAI-2026/qualifiers/notes/audio_chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

CHUNK_DURATION = 20
chunk_idx = 0

print(f"[Recorder] Bắt đầu thu âm nền liên tục từ microphone (mỗi đoạn {CHUNK_DURATION}s)...")
while True:
    chunk_idx += 1
    now_dt = datetime.now()
    chunk_path = os.path.join(CHUNKS_DIR, f"live_{chunk_idx:05d}_{now_dt.strftime('%H%M%S')}.wav")

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-f", "pulse", "-i", "default",
        "-t", str(CHUNK_DURATION),
        "-ar", "16000", "-ac", "1",
        chunk_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        time.sleep(1)
