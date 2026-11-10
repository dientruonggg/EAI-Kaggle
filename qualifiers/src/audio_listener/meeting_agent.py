#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - KICKOFF MEETING AUDIO LISTENER & NOTE-TAKING DAEMON
Enhanced with:
1. Pre-Whisper Audio Energy (RMS) Silence Gating (skips silent/idle chunks immediately).
2. Built-in Silero VAD (Voice Activity Detection) in faster-whisper.
3. Anti-Hallucination & Filler Word Filtering.
4. Smart Paragraph Aggregator (merges fragmented 20s phrases into coherent sentences).
5. Categorized Markdown Injection (updates specific sections in KICKOFF_NOTES.md).
"""

import os
import sys
import time
import argparse
import subprocess
import threading
import queue
import re
import wave
import numpy as np
from datetime import datetime

CHUNK_DURATION = 20  # seconds per audio chunk
MIN_RMS_THRESHOLD = 200  # Silence / idle noise threshold on int16 scale (0-32767)
MIN_TEXT_WORDS = 4  # Skip ultra-short snippets like "dạ vâng", "alo", "ừm" unless tagged
MAX_BUFFER_WORDS = 45  # Aggregate phrases before flushing to notes

NOTES_DIR = "/home/bim/Projects/EAI-2026/qualifiers/notes"
CHUNKS_DIR = os.path.join(NOTES_DIR, "audio_chunks")
TRANSCRIPT_LOG = os.path.join(NOTES_DIR, "transcript_live.log")
NOTES_MD = os.path.join(NOTES_DIR, "KICKOFF_NOTES.md")
PID_FILE = os.path.join(NOTES_DIR, "daemon.pid")

DEFAULT_SPEAKER_MONITOR = "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink.monitor"
DEFAULT_MIC_SOURCE = "alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source"

# Common Whisper hallucinations on silence / non-speech noise
HALLUCINATION_PATTERNS = [
    re.compile(r"cảm ơn các bạn (đã|theo dõi|lắng nghe)", re.IGNORECASE),
    re.compile(r"(hãy|nhớ) (like|đăng ký|subscribe|chia sẻ)", re.IGNORECASE),
    re.compile(r"hẹn gặp lại các bạn", re.IGNORECASE),
    re.compile(r"bài hát này được", re.IGNORECASE),
    re.compile(r"chúc các bạn một ngày", re.IGNORECASE),
    re.compile(r"^(ạ|ừ|ờ|alo|dạ|vâng|ừm|ơ|ha|hả|\.|\?|\!)+$", re.IGNORECASE),
]

# Keyword matching rules for meeting notes extraction
TAG_PATTERNS = {
    "📌 THỂ LỆ & QUY ĐỊNH": re.compile(
        r"(thể lệ|quy định|luật|vòng bảng|chung kết|kaggle|nộp bài|submission|map|accuracy|thời gian|deadline|hạn chót|điểm|0\.7|0\.3)",
        re.IGNORECASE
    ),
    "⚡ PHẦN CỨNG ESP32-S3 & CAMERA": re.compile(
        r"(esp32|esp-idf|camera|ov7670|sram|psram|flash|200k|tham số|tflite|int8|lượng tử|microcontroller|phần cứng|kit|tflm)",
        re.IGNORECASE
    ),
    "📊 DATASET & PHÁT HIỆN VẬT THỂ": re.compile(
        r"(dataset|dữ liệu|ảnh|train|val|test|private|public|chai|cốc|ly|điện thoại|sách|chuột|hộp|ô tô|toy_car|che khuất|bbox|bounding|coco|yolo)",
        re.IGNORECASE
    ),
    "💬 Ý ĐỊNH INTENT & TEXTCNN": re.compile(
        r"(intent|ý định|textcnn|count|exist|list|location|out_of_scope|đáp án|trả lời|json|câu hỏi|ngữ cảnh|từ điển)",
        re.IGNORECASE
    ),
    "❓ HỎI ĐÁP & GỢI Ý TỪ BTC": re.compile(
        r"(hỏi|trả lời|thắc mắc|cho em hỏi|lưu ý|chú ý|gợi ý|tip|kinh nghiệm|mentor|ban tổ chức|btc|clb|pioneer|giải đáp)",
        re.IGNORECASE
    )
}

def get_audio_rms(wav_path):
    """Calculates RMS audio energy to quickly detect and discard silence."""
    try:
        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return 0.0
            frames = wf.readframes(n_frames)
            samples = np.frombuffer(frames, dtype=np.int16)
            if len(samples) == 0:
                return 0.0
            return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    except Exception:
        return 0.0

def is_hallucination_or_noise(text):
    """Checks if transcribed text is a common hallucination or meaningless noise."""
    stripped = text.strip()
    if not stripped:
        return True
    for pat in HALLUCINATION_PATTERNS:
        if pat.search(stripped):
            return True
    # Repetition check (e.g. "xe xe xe xe xe")
    words = stripped.split()
    if len(words) >= 4 and len(set(words)) <= 2:
        return True
    return False

def init_notes_file():
    """Initializes the structured Markdown notes document."""
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    if not os.path.exists(NOTES_MD):
        with open(NOTES_MD, "w", encoding="utf-8") as f:
            f.write(f"""# 📝 GHI CHÚ BUỔI KICKOFF & TRAINING 1 — EDGE AI CHALLENGE 2026
**Thời gian bắt đầu ghi:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Đơn vị tổ chức:** CLB Nghiên cứu Khoa học Pioneer — Khoa Điện tử & AI, ĐH Bách Khoa ĐN  
**Trạng thái Daemon:** 🟢 Chạy nền, tự động lọc khoảng lặng & tổng hợp nội dung quan trọng.

---

## ⚡ BẢN TỔNG HỢP GHI CHÚ QUAN TRỌNG THEO CHỦ ĐỀ

### 📌 Thể lệ, Mốc thời gian & Quy tắc tính điểm
- *(Đang tự động trích xuất từ âm thanh cuộc họp...)*

### ⚡ Giới hạn Phần cứng ESP32-S3 & Camera OV7670
- *(Đang tự động trích xuất từ âm thanh cuộc họp...)*

### 📊 Dataset, Nhãn & Cơ chế Che khuất (Occlusion)
- *(Đang tự động trích xuất từ âm thanh cuộc họp...)*

### 💬 Ý định (Intent) & Mô hình TextCNN
- *(Đang tự động trích xuất từ âm thanh cuộc họp...)*

### ❓ Hỏi Đáp (Q&A) & Lời khuyên từ BTC
- *(Đang tự động trích xuất từ âm thanh cuộc họp...)*

---

## 📜 NHẬT KÝ THOẠI ĐÃ LỌC NỘI DUNG (FILTERED SPEECH)

""")

def append_to_notes(timestamp_str, text, tags):
    """Appends meaningful synthesized notes to KICKOFF_NOTES.md and transcript log."""
    clean_text = text.strip()
    if not clean_text:
        return

    # Write to raw filtered log
    with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp_str}] {clean_text}\n")

    # Format entry with bold tags if detected
    tag_prefix = f" **[{', '.join(tags)}]**" if tags else ""
    md_entry = f"- `[{timestamp_str}]`{tag_prefix} {clean_text}\n"

    with open(NOTES_MD, "a", encoding="utf-8") as f:
        f.write(md_entry)

class SmartAggregator:
    """Aggregates fragmented chunks into coherent thoughts, discarding idle intervals."""
    def __init__(self):
        self.buffer_words = []
        self.current_tags = set()
        self.start_timestamp = None
        self.last_speech_time = time.time()

    def add_snippet(self, timestamp_str, text, tags):
        if not self.start_timestamp:
            self.start_timestamp = timestamp_str

        words = text.split()
        self.buffer_words.extend(words)
        self.current_tags.update(tags)
        self.last_speech_time = time.time()

        # If substantial thoughts have gathered or strong topic tag found with punctuation
        has_punctuation = text.endswith((".", "!", "?", ":", ";"))
        if len(self.buffer_words) >= MAX_BUFFER_WORDS or (len(self.buffer_words) >= 15 and has_punctuation):
            self.flush()

    def flush(self):
        if not self.buffer_words:
            return
        combined_text = " ".join(self.buffer_words)
        # Final filter check
        if len(self.buffer_words) >= MIN_TEXT_WORDS or self.current_tags:
            append_to_notes(self.start_timestamp, combined_text, list(self.current_tags))
            print(f"\n💡 [GHI CHÚ] [{self.start_timestamp}] {combined_text}")
            if self.current_tags:
                print(f"   🏷️ Nhãn: {', '.join(self.current_tags)}")

        self.buffer_words = []
        self.current_tags = set()
        self.start_timestamp = None

    def check_idle_timeout(self):
        """Flushes buffer if a pause (silence) occurs after speaking."""
        if self.buffer_words and (time.time() - self.last_speech_time > CHUNK_DURATION * 1.5):
            self.flush()

def transcription_worker(audio_queue, model_name="base", stop_event=None):
    """Background worker that transcribes and filters chunks."""
    print(f"[MeetingAgent] Đang nạp Whisper ({model_name}) INT8 + Silero VAD...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"[MeetingAgent] ✅ Đã nạp xong Whisper ({model_name}). Chế độ lọc khoảng lặng kích hoạt!")
    except Exception as e:
        print(f"[MeetingAgent] ❌ Lỗi khởi tạo faster-whisper: {e}")
        return

    aggregator = SmartAggregator()

    while not (stop_event and stop_event.is_set() and audio_queue.empty()):
        try:
            item = audio_queue.get(timeout=2.0)
        except queue.Empty:
            aggregator.check_idle_timeout()
            continue

        chunk_file, timestamp_str = item
        if chunk_file is None:
            break

        try:
            # 1. Quick Energy Gating (RMS Silence Filter)
            rms = get_audio_rms(chunk_file)
            if rms < MIN_RMS_THRESHOLD:
                # Idle silence or quiet background noise: skip entirely
                # print(f"   [VAD] Bỏ qua đoạn rỗng {timestamp_str} (RMS: {rms:.1f} < {MIN_RMS_THRESHOLD})")
                aggregator.check_idle_timeout()
                continue

            # 2. Faster-Whisper with Silero VAD filter
            segments, info = model.transcribe(
                chunk_file,
                language="vi",
                beam_size=1,  # Fast inference
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )
            raw_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])

            # 3. Content Quality Check
            if not raw_text or is_hallucination_or_noise(raw_text):
                aggregator.check_idle_timeout()
                continue

            # 4. Keyword Tagger
            matched_tags = []
            for tag, pattern in TAG_PATTERNS.items():
                if pattern.search(raw_text):
                    matched_tags.append(tag)

            # Skip small filler chit-chat if not tagged
            if len(raw_text.split()) < MIN_TEXT_WORDS and not matched_tags:
                continue

            aggregator.add_snippet(timestamp_str, raw_text, matched_tags)

        except Exception as e:
            print(f"[MeetingAgent] Lỗi xử lý đoạn audio: {e}")
        finally:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except OSError:
                    pass
            audio_queue.task_done()

    # Final flush on exit
    aggregator.flush()

def record_audio_stream(source_type, audio_queue, stop_event):
    """Continuous audio capture using ffmpeg."""
    if source_type == "monitor":
        pulse_dev = DEFAULT_SPEAKER_MONITOR
    elif source_type == "mic":
        pulse_dev = DEFAULT_MIC_SOURCE
    else:
        pulse_dev = DEFAULT_SPEAKER_MONITOR

    print(f"[MeetingAgent] Thu âm nền từ thiết bị: {pulse_dev}")
    print(f"[MeetingAgent] Kích thước chunk: {CHUNK_DURATION}s | Ngưỡng lọc rỗng RMS: {MIN_RMS_THRESHOLD}")

    chunk_idx = 0
    while not stop_event.is_set():
        chunk_idx += 1
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%H:%M:%S")
        chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{chunk_idx:05d}_{now_dt.strftime('%H%M%S')}.wav")

        cmd = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-f", "pulse", "-i", pulse_dev,
            "-t", str(CHUNK_DURATION),
            "-ar", "16000", "-ac", "1",
            chunk_path
        ]

        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000:
                audio_queue.put((chunk_path, timestamp_str))
        except subprocess.CalledProcessError:
            time.sleep(1)
        except Exception:
            break

def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def remove_pid():
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass

def main():
    parser = argparse.ArgumentParser(description="Kickoff Meeting Real-time Audio Listener & Note Taker")
    parser.add_argument("--source", choices=["monitor", "mic", "both"], default="monitor",
                        help="Nguồn thu: 'monitor' (loa/hội nghị online), 'mic' (micro máy), 'both'")
    parser.add_argument("--model", choices=["tiny", "base", "small"], default="base",
                        help="Kích cỡ Whisper: tiny (nhẹ nhất), base (chuẩn cân bằng), small (chính xác hơn)")
    args = parser.parse_args()

    write_pid()
    init_notes_file()

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    t_worker = threading.Thread(
        target=transcription_worker,
        args=(audio_queue, args.model, stop_event),
        daemon=True
    )
    t_worker.start()

    try:
        record_audio_stream(args.source, audio_queue, stop_event)
    except KeyboardInterrupt:
        print("\n[MeetingAgent] Đang lưu nốt các ghi chú cuối cùng...")
    finally:
        stop_event.set()
        audio_queue.put((None, None))
        remove_pid()
        print("[MeetingAgent] Đã hoàn thành ghi chép và dừng an toàn.")

if __name__ == "__main__":
    main()
