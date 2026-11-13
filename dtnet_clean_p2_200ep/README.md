# DTNet Clean-Phase 200-Epoch Fine-Tuning Pipeline
> **Mục tiêu**: Bứt phá mAP50 từ 98.42% lên 99.0% - 99.5% trên vi điều khiển ESP32-S3 (Edge AI Challenge 2026).

---

## 1. Bản chất Kỹ thuật & Logic Mới (Anti-False-Positive Logic)

### Tại sao cần logic này?
Phân tích lỗi trên 600 ảnh validation (3,843 vật thể) cho thấy:
- **Recall thực tế của mô hình đã đạt 99.2% - 99.8%** (hầu hết mọi vật thể đều được nhận diện).
- Nhưng **mAP50 bị kéo tụt xuống 97.4% - 98.4%** do **Ghost Cross-Class False Positives** ở dải điểm $0.20 - 0.40$:
  - Ví dụ: Một chiếc điện thoại (`phone`) có điểm 0.85, nhưng anchor đó lại vô tình xuất hiện thêm dự đoán `toy_car` (điểm 0.28).
  - Vì chuẩn thi dùng Per-Class NMS, box `toy_car` 0.28 điểm không bị triệt tiêu và trở thành False Positive.
  - Khi tính diện tích dưới đường cong Precision-Recall, các box FP 0.28 điểm này xuất hiện trước các box GT thật có điểm 0.20, khiến Precision của `toy_car` rơi từ 96.9% xuống 85.0%!

### 3 Đòn bẩy kỹ thuật giải quyết triệt để:
1. **Cross-Class Mutual Exclusion Penalty ($w_{\text{mutual}} = 3.0, \beta = 1.0$)**:
   - Tại mọi vị trí anchor dương của lớp $c^*$ (ví dụ $c^* = \text{phone}$), 6 lớp còn lại BẮT BUỘC có target là 0.
   - Thay vì dùng Focal Loss tiêu chuẩn (vốn bị suy hao gradient khi $p \approx 0.25$), hàm loss áp dụng hình phạt tuyến tính mạnh để dập tắt dứt điểm các box ma này về 0.
2. **Scale-Adaptive EIoU Box Loss & Hard-Class Multipliers**:
   - $w_{\text{scale}} = 1.0 + 1.2 \times (1.0 - \sqrt{\text{area} / (W \times H)})$ tăng gấp đôi gradient cho các vật thể siêu nhỏ (`mouse` 9.4px, `toy_car` 9.5px).
   - Hệ số trọng số theo độ khó: `toy_car` (1.40x), `book` (1.35x), `mouse` (1.25x), `cup` (1.05x), `phone` (1.05x), `bottle` (1.00x), `box` (0.95x).
3. **Clean-Phase Native Resolution (Zero-Mosaic)**:
   - Tắt hoàn toàn Mosaic (`mosaic_p = 0.0`) trong toàn bộ 200 epochs để mạng học đường biên góc cạnh thực tế từ camera OV7670, triệt tiêu viền cắt giả tạo.

---

## 2. Cấu trúc Thư mục

```text
dtnet_clean_p2_200ep/
├── dtnet_max.py       # Kiến trúc DTNetMax (197,647 params < 200,000) & Keras synthesis
├── dataset.py         # CleanDetDataset (Native 320x240, Zero-Mosaic, OV7670 noise, Collision-free radius)
├── loss.py            # Anti-False-Positive Loss (AFPL): EIoU + Scale-Adaptive + Mutual Exclusion + QFL
├── evaluation.py      # Vectorized NMS (nms_iou=0.65, thr=0.001), VOC mAP50 evaluator, ModelEMA
├── train.py           # Pipeline 200 epochs, Cosine Annealing (1.5e-4 -> 2e-6), checkpointing
├── export_int8.py     # Xuất Full INT8 TFLite, kiểm tra sai số Keras < 0.02 và size < 1.5MB
├── calibrate.py       # Quét ngưỡng tối ưu trên mô hình INT8
├── run_train.sh       # Script bash chạy nền nohup tự động
└── README.md          # Tài liệu tái lập chi tiết
```

---

## 3. Hướng dẫn Tái lập (Reproducibility Guide)

### Bước 1: Khởi chạy Huấn luyện 200 Epochs

```bash
# Chạy nền tự động (log ghi vào qualifiers/weights/dtnet_clean_p2_200ep/train.log)
./dtnet_clean_p2_200ep/run_train.sh

# Xem tiến độ trực tiếp
tail -f qualifiers/weights/dtnet_clean_p2_200ep/train.log
```

### Bước 2: Xuất mô hình Full INT8 TFLite

```bash
python dtnet_clean_p2_200ep/export_int8.py \
  --checkpoint qualifiers/weights/dtnet_clean_p2_200ep/best_dtnet_clean_200ep.pt \
  --calib_dir qualifiers/data/train/images \
  --output qualifiers/weights/dtnet_clean_p2_200ep/dtnet_clean_200ep_int8.tflite \
  --num_calib 200
```

### Bước 3: Đánh giá & Calibrate Ngưỡng

```bash
python dtnet_clean_p2_200ep/calibrate.py \
  --tflite qualifiers/weights/dtnet_clean_p2_200ep/dtnet_clean_200ep_int8.tflite \
  --val_dir qualifiers/data/val \
  --output dtnet_clean_p2_200ep/calibrated_thresholds.json
```
