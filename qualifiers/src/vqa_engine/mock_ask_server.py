#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - MOCK ESP32-S3 /ask ENDPOINT
Simulates the Finals HTTP POST /ask interface specified in Section VI.1 & VI.2:
- Receives HTTP POST /ask with JSON: {"q": "Co bao nhieu chai trong hinh"}
- Runs TextCNN intent classification
- Runs Object Detection on current camera frame (or test image)
- Synthesizes standardized answer
- Responds with JSON: {"answer": "2"} in < 10 seconds.
"""
import os
import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vqa_engine.answer_generator import generate_answer
from src.vqa_engine.question_parser import normalize_text

class AskServerHandler(BaseHTTPRequestHandler):
    def _send_json_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        start_time = time.time()
        if self.path != "/ask":
            self._send_json_response(404, {"error": "Not Found. Use POST /ask"})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")

        try:
            req_data = json.loads(body)
            question = req_data.get("q", "")
        except Exception as e:
            self._send_json_response(400, {"error": f"Invalid JSON body: {str(e)}"})
            return

        print(f"\n[*] Received Question: '{question}'")

        # Mock detections (in production ESP32, this is obtained from OV7670 camera frame)
        mock_detections = [
            {"class_id": 0, "score": 0.92, "bbox": [50.0, 40.0, 95.0, 180.0]},    # bottle (left)
            {"class_id": 0, "score": 0.88, "bbox": [140.0, 50.0, 180.0, 190.0]},  # bottle (center)
            {"class_id": 1, "score": 0.85, "bbox": [240.0, 80.0, 290.0, 160.0]}   # cup (right)
        ]

        # Determine Intent (rule or TextCNN)
        norm = normalize_text(question)
        if any(w in norm for w in ["thoi tiet", "gio roi", "ten la gi", "ngay may"]):
            intent = "OUT_OF_SCOPE"
        elif any(w in norm for w in ["bao nhieu", "dem so", "may cai", "may quyen", "may chiec"]):
            intent = "COUNT"
        elif any(w in norm for w in ["co", "khong", "hay khong", "nhin thay", "thay"]):
            intent = "EXIST"
        elif any(w in norm for w in ["o dau", "ben nao", "phia nao", "vi tri"]):
            intent = "LOCATION"
        elif any(w in norm for w in ["trong anh co gi", "nhung do vat gi", "liet ke", "ke ten"]):
            intent = "LIST"
        else:
            intent = "OUT_OF_SCOPE"

        answer = generate_answer(intent, question, mock_detections)
        elapsed = time.time() - start_time

        print(f"[*] Intent: {intent} | Answer: '{answer}' | Time: {elapsed*1000:.1f}ms (Limit: 10s)")
        self._send_json_response(200, {"answer": answer})

def run_mock_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, AskServerHandler)
    print(f"[*] Mock ESP32-S3 /ask server running on http://localhost:{port}/ask")
    httpd.serve_forever()

if __name__ == "__main__":
    run_mock_server()
