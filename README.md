# 🚀 Edge AI Challenge 2026 — TinyVQA on ESP32-S3

[![Spec Kit](https://img.shields.io/badge/Spec--Kit-SDD-blue?logo=github)](https://github.com/github/spec-kit)
[![Target](https://img.shields.io/badge/Hardware-ESP32--S3_N16R8-red)](https://www.espressif.com/)
[![Camera](https://img.shields.io/badge/Sensor-OV7670-orange)]()
[![Model Limit](https://img.shields.io/badge/Params-%3C200k-green)]()
[![Quantization](https://img.shields.io/badge/Quantization-Full_INT8-purple)]()

Hệ thống Hỏi - Đáp hình ảnh trực quan (**Visual Question Answering - TinyVQA**) hoạt động **100% on-device trên vi điều khiển ESP32-S3 kết hợp Camera OV7670**, tuân thủ nghiêm ngặt quy chế không sử dụng cloud, server hay kết nối ngoài trong quá trình suy luận.

Dự án áp dụng quy chuẩn **Spec-Driven Development (SDD)** của [GitHub Spec Kit](https://github.com/github/spec-kit.git).

---

## 🎯 Chiến lược Kép (Dual Strategy)

1. **Vòng loại (Kaggle Leaderboard)**:
   - Tối ưu mAP trên tập dữ liệu đề thi bằng các mô hình chuyên biệt (MobileNetV2-SSDLite, FOMO, WBF Ensemble).
   - Huấn luyện từ xa trên Dual NVIDIA T4 qua `warmup/kaggle_runner/run_remote_train.py`.
2. **Vòng chung kết (ESP32-S3 + OV7670)**:
   - Mô hình TinyML siêu nhẹ (**< 200,000 parameters**, **$\le 1.5$ MB Flash**, **Full INT8 Quantization**).
   - Kiến trúc Neuro-Symbolic 3 chặng: NLP Intent Parser $\rightarrow$ Vision Perception (TFLite Micro) $\rightarrow$ Symbolic Reasoning (C/C++).
   - Chế độ dự phòng Classical CV thuần C (HSV Thresholding + Contours) khi có ràng buộc cấm Deep Learning.

---

## 🏗️ Kiến trúc Hệ thống (Neuro-Symbolic Pipeline)

```
[ Natural Language Question (UART/Serial) ]
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  Stage 1: Tiny NLP & Intent Parser  │ ──► Intent: {COUNT, EXISTENCE, COLOR, SPATIAL}
  │  (Regex / Tiny INT8 1D-CNN < 30KB)   │ ──► Slot: {target_class, query_attribute}
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  Stage 2: Vision Perception (INT8)  │ ◄── [ OV7670 Camera Frame (96x96 to 160x160) ]
  │  (FOMO Centroid / MobileNet-SSDLite)│ ──► Detections: [(class, x, y, w, h, score)]
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  Stage 3: Symbolic Reasoning (C/C++) │ ──► Deterministic Rule Evaluation & ROI Color Analysis
  └──────────────────┬──────────────────┘
                     │
                     ▼
          [ Final Answer String ]
```

---

## 🧭 Spec Kit SDD Workflow

Dự án tích hợp đầy đủ công cụ **Spec Kit** của GitHub với agent **Antigravity (`agy`)**.

### 🧩 Danh mục Đầy đủ 9 Kỹ năng Cốt lõi (Core SDD Skills)
Tất cả 9 kỹ năng cốt lõi được cài đặt trực tiếp tại [`.agents/skills/`](file:///home/bim/Projects/EAI-2026/.agents/skills):

| STT | Kỹ năng | Mục đích hoạt động | Sản phẩm đầu ra |
| :---: | :--- | :--- | :--- |
| **1** | [`/speckit-constitution`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-constitution/SKILL.md) | Thiết lập và cập nhật nguyên tắc, ràng buộc phần cứng & rào cản chất lượng | [`.specify/memory/constitution.md`](file:///home/bim/Projects/EAI-2026/.specify/memory/constitution.md) |
| **2** | [`/speckit-specify`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-specify/SKILL.md) | Khởi tạo đặc tả kỹ thuật tính năng với User Stories (P1-P3) và kịch bản Given/When/Then | [`specs/<id>/spec.md`](file:///home/bim/Projects/EAI-2026/specs/001-tinyvqa-esp32s3/spec.md) |
| **3** | [`/speckit-clarify`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-clarify/SKILL.md) | Quét và phát hiện vùng yêu cầu mơ hồ, đặt câu hỏi làm rõ giảm rủi ro | Cập nhật `spec.md` |
| **4** | [`/speckit-checklist`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-checklist/SKILL.md) | Tạo bảng kiểm thử xác nhận chất lượng yêu cầu trước khi triển khai | `specs/<id>/checklists.md` |
| **5** | [`/speckit-plan`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-plan/SKILL.md) | Lập kế hoạch kiến trúc, lựa chọn mô hình và cấu trúc dữ liệu kỹ thuật | `specs/<id>/plan.md` |
| **6** | [`/speckit-tasks`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-tasks/SKILL.md) | Phân rã kế hoạch thành các task độc lập, sắp xếp theo thứ tự phụ thuộc | `specs/<id>/tasks.md` |
| **7** | [`/speckit-analyze`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-analyze/SKILL.md) | Phân tích chéo tính nhất quán và độ lệch giữa `spec.md`, `plan.md`, `tasks.md` | Báo cáo kiểm định chéo |
| **8** | [`/speckit-implement`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-implement/SKILL.md) | Tiến hành viết mã nguồn và unit test tuần tự theo danh sách task | Mã nguồn hoàn thiện |
| **9** | [`/speckit-converge`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-converge/SKILL.md) | Đánh giá đối soát codebase với đặc tả, lặp lại bổ sung task cho tới khi hội tụ | Báo cáo `Converged` |

---

## ⚡ Hai Lựa chọn Pipeline Tự động hóa

Dự án cung cấp 2 pipeline định nghĩa sẵn trong [`.specify/workflows/`](file:///home/bim/Projects/EAI-2026/.specify/workflows):

### 1. 🛡️ Pipeline Toàn Diện (Full 9-Skill Guided Pipeline)
- **ID Workflow**: `full-pipeline` (hoặc lệnh chat [`/speckit-pipeline`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-pipeline/SKILL.md))
- **Chuỗi thực thi**:
  $$\text{constitution} \longrightarrow \text{specify} \longrightarrow \text{clarify} \longrightarrow \text{checklist} \longrightarrow \text{plan} \longrightarrow \text{tasks} \longrightarrow \text{analyze} \longrightarrow \text{implement} \longrightarrow \text{converge loop}$$
- **Khi nào nên dùng**: Dành cho các tính năng trọng yếu, mô hình nhúng mới, hoặc kiến trúc cần thẩm định nghiêm ngặt về ràng buộc phần cứng ESP32-S3.
- **Cách gọi từ CLI**:
  ```bash
  specify workflow run full-pipeline
  ```

### 2. 🚀 Pipeline Tinh Gọn (Fast SDD Express — Tổ hợp Ngắn)
- **ID Workflow**: `fast-sdd` (hoặc lệnh chat [`/speckit-fast`](file:///home/bim/Projects/EAI-2026/.agents/skills/speckit-fast/SKILL.md))
- **Chuỗi thực thi**:
  $$\text{specify} \longrightarrow \text{plan} \longrightarrow \text{tasks} \longrightarrow \text{implement}$$
- **Khi nào nên dùng**: Thử nghiệm nhanh các giải pháp xử lý ảnh, chỉnh sửa thuật toán, tinh chỉnh hyperparameter hoặc prototyping mà không cần dừng qua các cổng xét duyệt (gates).
- **Cách gọi từ CLI**:
  ```bash
  specify workflow run fast-sdd
  ```

---

## 📁 Cấu trúc Thư mục

```
EAI-2026/
├── .agents/                  # Antigravity Agent skills, conventions & tasks
│   └── skills/               # 9 Core Spec Kit skills + workflow runners + domain skills
├── .specify/                 # Spec Kit metadata, memory, extensions & templates
│   ├── memory/
│   │   └── constitution.md   # Project Constitution & Invariants
│   ├── templates/            # SDD Markdown templates
│   └── workflows/            # Workflow definitions (full-pipeline, fast-sdd, speckit)
├── specs/                    # Feature specifications & plans
│   └── 001-tinyvqa-esp32s3/  # Active TinyVQA on-device specification
├── docs/                     # Tài liệu cuộc thi & quy định chính thức
├── datasets/                 # Datasets dùng chung (format chuẩn YOLO)
├── qualifiers/               # Mã nguồn vòng sơ loại (Qualifiers VQA)
│   ├── src/                  # detection, intent, vqa_engine
│   ├── weights/              # Trọng số mô hình sơ loại
│   └── kaggle_runner/        # Pipeline huấn luyện Kaggle T4x2
├── warmup/                   # Thử nghiệm khởi động & đánh giá ban đầu
│   ├── src/                  # micro_fomo, micro_yolo, detection
│   └── weights/              # INT8 TFLite & pretrained weights
├── AGENTS.md                 # Hardware & Invariants cho AI agents
├── BENCHMARK_MATRIX.md       # Ma trận băng thử nghiệm mô hình ESP32-S3
└── roadmap.md                # Lộ trình học tập & thực chiến Kaggle
```

---

## ⚖️ Ràng buộc Phần cứng & Quy định (Hard Invariants)

1. **Giới hạn Tham số**: Tuyệt đối **< 200,000 tham số** cho mỗi mô hình triển khai.
2. **Flash Budget**: $\le 1.5$ MB tổng dung lượng Flash cho toàn bộ mô hình và bảng trọng số.
3. **Lượng tử hóa**: Bắt buộc **Full Integer INT8** (`int8 input/output`). Dùng Post-Training Quantization (PTQ) với $\ge 100$ ảnh đại diện cho Keras 3. Không dùng `tfmot.quantize_model` vì lỗi drop layer.
4. **Sweet Spot Gating**: Tập test 60 ảnh phải đạt số lượng box $2,600 \le N \le 2,800$ (43–47 boxes/ảnh) trước khi nộp.
5. **Intent Predictions**: File dự đoán intent (`submissions/intent_preds.csv`) bắt buộc giữ điểm tuyệt đối 30.00/30.00.
