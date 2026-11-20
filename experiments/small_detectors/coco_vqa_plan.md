# Kế hoạch pilot COCO → EAI và VQA trên PC

> Dùng checklist và giữ tất cả output trong lab hoặc thư mục tạm được chỉ định. Kiểm chứng mỗi chặng bằng Superpowers `verification-before-completion`.

**Mục tiêu**

Đo tác động của warm start COCO lên đáp án VQA trên cùng 24 ảnh EAI, với đối chứng ImageNet → EAI.

**Thiết kế**

Giữ nguyên kiến trúc 7 lớp và dùng loss mask trong chặng COCO cho các lớp thiếu. Sau đó huấn luyện cả 7 lớp trên cùng 96 ảnh EAI. Chấm oracle intent và pipeline TextCNN khóa bằng cùng bộ câu hỏi có đáp án xác định từ nhãn bbox.

**Phạm vi**

CPU, input RGB 128 × 96, một forward mỗi ảnh cho từng biến thể, ≤200.000 tham số gồm BN, xuất I/O INT8 nếu qua audit. COCO chỉ từ train2017, mỗi ảnh được chọn tải riêng. EAI chỉ dùng 96/24 do Điều phối ghi trong manifest. Cụm lab không ghi vào release, model BTC, firmware hay kit. [Thể lệ gốc, mục V và VI, trang 5–7](<../../docs/The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf>) · [COCO của Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

## 1. Chọn dữ liệu COCO

- [x] Kiểm tra label ZIP và chỉ đọc `coco/labels/train2017/*.txt`. Chọn 32 ảnh mỗi lớp theo seed `20261120`, không lặp ID giữa các quota, tổng 160 ảnh. Giữ mọi bbox thuộc 5 lớp trong từng ảnh chọn. Mapping ID YOLO COCO → EAI là `39→0, 41→1, 67→2, 73→3, 64→4`. Bỏ toàn bộ lớp khác, gồm car (ID 2). [Thứ tự nhãn COCO](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml)
- [ ] Dùng `prepare_coco.py` để lấy JPEG riêng từng ID từ `https://images.cocodataset.org/train2017/<id>.jpg`, lưu bbox đã map và manifest có URL, SHA-256 cùng quota. Lượt 160 ảnh dừng ở lỗi DNS và lưu danh sách URL trong `selection.json` của lượt chuẩn bị local.
- [x] Chạy phương án dự phòng trên COCO128 chính thức có sẵn. Chọn có seed tối đa 8 ID mỗi lớp từ phần giao với ZIP nhãn, ưu tiên phân quota cho lớp hiếm và trích đúng 22 ảnh đã chọn. Giữ cấu hình huấn luyện và EAI đối chứng. [Nguồn COCO128](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml)

## 2. Đối chứng với cùng cấu hình

- [x] Chạy `run_pilot.py` hai lượt với cùng H5 ImageNet, seed, kiến trúc, freeze backbone, Adam 0,002, batch 8 và **4 epoch EAI**. Lượt warm start thêm **4 epoch COCO**, chỉ train neck/head. FOMO softmax có background và 5 foreground được phép. YOLO mở 5 kênh class, vẫn học bbox/objectness từ bbox đã map. Giữ nguyên kernel và bias cuối của box/toy_car trong chặng COCO. Reset Adam khi bắt đầu EAI và mở cả 7 lớp. Đối chứng bắt đầu EAI từ head khởi tạo cùng seed.
- [x] Xuất INT8 với cùng 24 ảnh hiệu chuẩn thuộc EAI train. Xác nhận số tham số, SHA-256 backbone và đầu ra từng chế độ. Giữ val đến sau fit và PTQ. Lưu centroid hoặc bbox của mỗi ảnh theo phiên bản để dùng chung giữa hai chế độ hỏi đáp.

## 3. Đo VQA

- [x] Dùng `evaluate_vqa.py` tạo câu hỏi cố định từ nhãn val. COUNT hỏi một lớp có mặt, EXIST xen kẽ hỏi lớp có/không mặt khi cảnh cho phép, LIST hỏi danh sách, LOCATION chỉ hỏi lớp có vị trí vùng không mơ hồ, OOS dùng mẫu ngoài phạm vi. Lưu câu hỏi, ID, intent GT và đáp án GT. Đây là bộ câu hỏi lab sinh từ nhãn.
- [x] Dùng dictionary cố định để rút lớp từ câu hỏi. Tổng hợp COUNT chuỗi số, EXIST Co/Khong, LIST đếm theo từng lớp, LOCATION từ vùng trái/giữa/phải và OOS. Với nhiều dấu dự đoán của cùng lớp, lấy vùng có nhiều dấu nhất, hòa thì ưu tiên từ trái sang phải. Trả `Khong co` khi câu LOCATION thiếu dự đoán, và tính sai trong bộ câu hỏi đã có vật thể.
- [x] Dùng intent GT cho oracle và `release_frozen` TextCNN bất biến cho pipeline. Hai chế độ dùng cùng slot extractor và cùng đầu ra Detect. Chấm exact match theo từng intent, báo riêng accuracy TextCNN, đồng thời giữ centroid F1/count/location và bbox mAP50 như pilot EAI trước. Kết luận chỉ từ split này và báo rõ blocker tải COCO.

Lệnh chuẩn bị, huấn luyện và đánh giá nằm trong báo cáo bổ sung [report.md](report.md).

### Tập 120 ảnh do Điều phối bàn giao

Đặt `COCO120_DIR` tới bộ 120 ảnh do Điều phối bàn giao, gồm `sources.json` và `train/{images,labels}`. Runner kiểm đủ 120 cặp, đúng seed, URL S3 train2017, SHA-256 ảnh và nhãn YOLO đã ánh xạ trước khi fit. Bộ 120 ảnh và bộ 22 ảnh chọn từ COCO128 có manifest riêng. Clone thiếu fixture EAI 96/24. Dùng `TF_PY`, `LITE_PY`, `MOBILENET_H5` và `EAI_DATASET` đúng schema theo [report.md](report.md#lệnh-chạy-pilot). Chạy nhánh đối chứng trên cùng split trước khi so 119 câu VQA. Output sinh local trong `coco120_eai_artifacts` và `eai_control_artifacts`.

```bash
lab=experiments/small_detectors
out="$lab/coco120_eai_artifacts"
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 "${TF_PY:?}" -B "$lab/run_pilot.py" \
  --weights "${MOBILENET_H5:?}" --dataset "${EAI_DATASET:?}" \
  --baseline-python "${LITE_PY:?}" --coco "${COCO120_DIR:?}" --out "$out"
"$LITE_PY" -B "$lab/evaluate_vqa.py" --dataset "$EAI_DATASET" \
  --predictions "$out" --out "$out/vqa"
cmp "$lab/eai_control_artifacts/vqa/vqa_questions.json" "$out/vqa/vqa_questions.json"
```

### Tập COCO train2017 8.000 ảnh do Điều phối chuẩn bị

Manifest khóa 8.000 ID duy nhất, 1.600 ảnh theo lớp chính cho mỗi lớp bottle, cup, phone, book và mouse, seed `20261120`. Từ clone, dùng bộ tải để tái tạo ảnh, nhãn và `sources.json`. Runner dùng `sources.json` để kiểm hash, kích cỡ JPEG và nhãn YOLO đã map. `selection.json` chỉ là hiện vật của bộ canonical local. Mọi nhãn COCO chỉ thuộc ID 0–4. EAI giữ ảnh và nhãn box/toy_car của chặng 7 lớp trong split riêng. [Bộ tải COCO 8k](../../src/data_tools/coco_eai/README.md)

```bash
lab=experiments/small_detectors
coco=${COCO8K_DIR:-datasets/coco2017_eai_8k}
python3 src/data_tools/coco_eai/restore.py --dest "$coco"
"${TF_PY:?}" -B "$lab/run_pilot.py" --verify-coco "$coco"
```

Khi Điều phối giao chặng train, hướng chạy 8k cần cờ `--train-large-coco` và `--out` mới hoặc rỗng, nằm ngoài các thư mục dữ liệu. Chặng này shuffle ID theo seed ở mỗi epoch, đọc ảnh và tạo tensor đích theo batch 8, không prefetch và không giữ cache ảnh/tensor đích toàn tập. Chặng EAI 96/24 và nhánh pilot COCO nhỏ giữ luồng hiện có.

Một batch 8 có khoảng 1,125 MiB input FP32 sau resize, cộng khoảng 0,264 MiB target dense YOLO. Đây là kích thước tensor theo công thức, chưa tính JPEG giải mã, bộ đệm tạm, TensorFlow và mô hình. Nếu giữ input cả 8.000 ảnh trong RAM sẽ riêng phần input mất khoảng 1,10 GiB. Bộ đọc chỉ giữ metadata đường dẫn toàn split, rồi giải mã ảnh theo batch trong lúc chạy.

Kiểm tra fixture đã đối chiếu tensor theo batch với đường đọc toàn bộ, gồm batch cuối thiếu ảnh. Pilot đối chứng EAI tái tạo dự đoán và hai model INT8 khớp từng byte. Lượt chuẩn bị này không train tập 8k.
