---
name: codebase-fact-check
description: Adversarial multi-agent verification and ground-truth fact-checking for technical papers, documentation, and specs against the real codebase. Use when verifying technical claims, model names, hyperparameters, network protocols, or system architecture against source code. Enforces the invariant 'Chỉ viết những gì đang có' (Zero speculation).
---

# Codebase Ground-Truth Fact-Checking Workflow (Skill Kiểm chứng Mã nguồn)

Skill này cung cấp quy trình đa tác tử (multi-agent) đối soát và kiểm chứng tính xác thực tuyệt đối của mọi nội dung kỹ thuật trong bài báo khoa học, báo cáo dự án hoặc tài liệu thiết kế dựa trên mã nguồn thực tế đang chạy.

## Nguyên tắc Bất biến (Core Invariants)

1. **"Chỉ viết những gì đang có" (Zero Speculation & Ground-Truth Only)**:
   - Mọi thông số kỹ thuật, tên mô hình, kiến trúc mạng, giao thức mạng, cổng dịch vụ, tên hàm/class được viết vào bài báo/tài liệu PHẢI được kiểm chứng đối chiếu trực tiếp từ mã nguồn thực tế đang chạy.
   - Tuyệt đối không suy đoán hoặc viết những thành phần không tồn tại trong source code.
2. **Tách biệt Phương pháp luận vs Triển khai thực nghiệm (Separation of Concerns)**:
   - **Mục Phương pháp luận (Methodology - Mục 3)**: Chỉ trình bày nguyên lý toán học, mô hình thuật toán, kiến trúc phân tầng luồng dữ liệu. Tuyệt đối không chứa URL cụ thể, cổng port, chuỗi endpoint API chi tiết hay cấu hình môi trường triển khai cục bộ.
   - **Mục Triển khai & Kết quả (Experimental Implementation & Results - Mục 4)**: Nơi tập trung toàn bộ danh mục API chi tiết, URL máy chủ thật, thông số phần cứng kiểm nghiệm và bảng số liệu đo đạc thực tế.
3. **Văn phong Khoa học Cô đọng (Academic Conciseness)**:
   - Ngắn gọn, súc tích, đi thẳng vào trọng tâm kỹ thuật.
   - Loại bỏ hoàn toàn các đoạn văn mở đầu kể lể bối cảnh chung chung (boilerplate filler) không cần thiết cho mục phương pháp.

---

## Quy trình Kiểm chứng 4 Bước (4-Phase Multi-Agent Verification)

```
[Bản thảo / Đoạn văn bản kỹ thuật]
               │
               ▼
┌──────────────────────────────────────────────┐
│ Phase 1: Bóc tách Phát biểu & Đặt câu hỏi    │ ◄── Tác tử Phản biện (Skeptical Reviewer)
│ (Extract Claims & Generate Adversarial Qs)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Phase 2: Điều tra Mã nguồn Thực tế           │ ◄── Tác tử Điều tra (Forensic Investigator)
│ (Direct Code Inspection & Evidence Gathering)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Phase 3: Lập Ma trận Đối soát (Matrix)       │ ◄── Phân loại: VERIFIED / MISMATCH / PHANTOM / MISPLACED
│ (Discrepancy Reporting & Fact Reconciliation)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Phase 4: Hiệu chỉnh Phẫu thuật (Surgical Fix)│ ◄── Giữ đúng thực tế, tuân thủ Invariants
└──────────────────────────────────────────────┘
```

### Phase 1: Bóc tách Phát biểu & Tạo Bộ câu hỏi Phản biện (Skeptical Inquisitor)

Khi tiếp nhận bản thảo một mục kỹ thuật:
1. **Trích xuất toàn bộ các thực thể sự thật (Factual Entities)**:
   - Tên mô hình AI & backbone (ví dụ: `YOLOv8n`, `SCRFD`, `MiniFASNet v2`, `GhostFaceNet`).
   - Định dạng tệp trọng số & kích thước đĩa (ví dụ: `.rknn`, `.onnx`, `8.0 MB`).
   - Kích thước tensor đầu vào & tiền xử lý (ví dụ: `640x640`, Affine 2D `112x112`).
   - Các siêu tham số định lượng (ví dụ: Ngưỡng Cosine distance `0.55`, Top-K voting `K=5, min=3`).
   - Giao thức mạng, CSDL, cổng kết nối (FastAPI port 8000, ChromaDB HNSW Cosine, WebSocket).
2. **Khởi tạo bộ câu hỏi phản biện gắt gao (Adversarial Questions)**:
   - *Q1 (Trọng số)*: File mô hình này có thực sự nằm trong thư mục trọng số (`models/` hoặc `weights/`) không? Dung lượng thực tế là bao nhiêu byte/MB?
   - *Q2 (Khởi chạy)*: Node/Module nào trong code thực sự import và chạy suy luận mô hình này?
   - *Q3 (Siêu tham số)*: Tham số ngưỡng X được gán giá trị ở file cấu hình nào? Code logic so khớp dùng phép so sánh gì (`<=` hay `>=`)?
   - *Q4 (Luồng dữ liệu)*: Luồng xử lý có bị ngắt hoặc rẽ nhánh theo trạng thái robot không (ví dụ: Person Gating)?

### Phase 2: Điều tra Mã nguồn Thực tế (Forensic Codebase Investigation)

Tác tử điều tra (hoặc subagent `research`) quét trực tiếp trên workspace:
- **Kiểm tra tệp tin thực tế**:
  ```bash
  find src/ -name "*model*" -o -name "*.rknn" -exec ls -lh {} +
  ```
- **Kiểm tra cấu hình & hằng số**:
  Tìm chính xác nơi khai báo biến và giá trị mặc định trong `settings.py`, `.env`, hoặc config YAML.
- **Kiểm tra logic thực thi**:
  Đọc hàm thực tế (ví dụ: `face_controller.py`, `attendance_manager.py`) để thấy cách tính toán độ tương đồng, quy trình voting, và cách xử lý khi mất mạng (offline queue/outbox).

### Phase 3: Lập Ma trận Đối soát & Báo cáo Sai lệch (Cross-Checking Matrix)

Mọi phát biểu được gắn 1 trong 4 nhãn:
- **`VERIFIED` (Đã kiểm chứng)**: Khớp 100% với code (Ghi chú rõ đường dẫn file và số dòng).
- **`MISMATCH` (Sai lệch thông số/tên gọi)**: Có tính năng tương đương nhưng thông số hoặc kiến trúc khác với văn bản (ví dụ: viết ngưỡng 0.40 nhưng code là 0.55; viết YOLOv8-pose nhưng code là YOLOv8n person detector).
- **`PHANTOM` (Tính năng ảo / Ảo giác)**: Hoàn toàn không tồn tại trong codebase.
- **`MISPLACED` (Sai vị trí trình bày)**: Tính năng có thật nhưng là chi tiết triển khai/URL/API cụ thể cần đưa về Mục 4 (Thực nghiệm) thay vì Mục 3 (Phương pháp).

*Mẫu Bảng Ma trận Đối soát:*
| Nội dung trong bài viết | Thực tế trong Codebase | Trạng thái | Dẫn chứng Mã nguồn (File & Line) | Phương án xử lý |
| :--- | :--- | :--- | :--- | :--- |
| Nhận diện tư thế YOLOv8-Pose | Chỉ có YOLOv8n lọc người (`yolov8n_rk3588.rknn`) | `MISMATCH` | `pipeline.py#L42`, `models/` | Sửa thành Cổng lọc đối tượng Person Gating bằng YOLOv8n |
| Ngưỡng khoảng cách Cosine 0.40 | `RECOGNITION_THRESHOLD = 0.55` | `MISMATCH` | `settings.py#L95`, `face_controller.py#L384` | Sửa chính xác thành 0.55 (tương đồng 72.5%) |
| URL `https://manage.dutai...` trong Mục 3 | Có trong `config.py` nhưng nằm sai mục | `MISPLACED` | `config.py#L12` | Bỏ khỏi Mục 3 (Phương pháp), chuyển sang Mục 4 (Thực nghiệm) |

### Phase 4: Hiệu chỉnh Phẫu thuật (Surgical Ground-Truth Editing)

1. Xóa bỏ hoàn toàn mọi chi tiết bị gắn nhãn `PHANTOM`.
2. Thay thế thông số sai lệch bằng giá trị chính xác từ mã nguồn (`MISMATCH`).
3. Chuẩn hóa vị trí nội dung: Đưa chi tiết API/URL/Phần cứng kiểm thử về Mục 4 (`MISPLACED`).
4. Giữ văn phong súc tích, học thuật, tuân thủ các quy chuẩn định dạng bài viết.

---

## Checklists Kiểm tra Nhanh theo Hạng mục

### 1. Mô hình AI & Thị giác máy tính (AI / Vision Pipeline)
- [ ] Tên file trọng số và dung lượng đĩa (MB) khớp với lệnh `ls -lh`.
- [ ] Kích thước tensor đầu vào (Input Resolution) khớp với code tiền xử lý (`cv2.resize`, Affine transform).
- [ ] NPU Core / Hardware acceleration (1 lõi, 3 lõi, NPU RK3588 6 TOPS) được cấu hình rõ trong code RKNN runtime.
- [ ] Bộ lọc tính sống (Anti-spoofing) có thực sự chạy hay chỉ là dummy function.

### 2. Thuật toán So khớp & Cơ sở dữ liệu Vector (Matching & Vector DB)
- [ ] Độ đo khoảng cách: Cosine distance ($1 - \cos$) hay L2 hay Inner Product.
- [ ] Công thức quy đổi độ tương đồng (%) trên giao diện người dùng.
- [ ] Cơ chế bỏ phiếu (Voting): Số lân cận $K$, số phiếu tối thiểu.
- [ ] Loại CSDL vector: ChromaDB, FAISS hay In-memory NumPy array.

### 3. Mạng & Kiến trúc Hệ thống (Networking & Resilience)
- [ ] Giao thức đồng bộ: HTTP REST Multipart, gRPC, hay ROS 2 Topic.
- [ ] Cơ chế Offline / Fallback: Hàng đợi cục bộ (SQLite/File outbox), thuật toán retry (exponential backoff).
- [ ] Web Framework: FastAPI, Flask, hay pure asyncio.
