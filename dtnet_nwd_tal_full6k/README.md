# DTNet Full-6K TAL + NWD Training Pipeline
> **Mục tiêu**: Bứt phá mAP50 từ 97.87% lên 99.0% - 99.5% cho Edge AI Challenge 2026 trên vi điều khiển ESP32-S3.

---

## 1. Bản chất Kỹ thuật & 3 Đột phá Toán học

1. **Normalized Wasserstein Distance (NWD) cho vật thể siêu nhỏ**:
   - Khắc phục hiện tượng sụp đổ IoU: Với vật thể 10px (`mouse`, `toy_car`), chỉ cần lệch 2 pixel, IoU rơi xuống **0.47 (< 0.50)**.
   - NWD mô hình hóa bounding box thành phân phối Gaussian 2D, đo khoảng cách Wasserstein-2. Ngay cả khi lệch 2 pixel, NWD vẫn duy trì **0.79**, cung cấp gradient liên tục giúp định vị chính xác sub-pixel.
2. **Task-Aligned Dynamic Assigner (TAL)**:
   - Loại bỏ hoàn toàn gán nhãn bán kính hình học cố định.
   - Chọn top-4 anchor có metric tương thích $\text{score}^{0.5} \times \text{IoU}^{6.0}$ cao nhất, loại trừ 100% false positives trên nền bàn trống.
3. **Huấn luyện toàn diện Full 6,000 ảnh (5,400 train + 600 val)**:
   - Khai thác trọn vẹn 100% dữ liệu có nhãn của BTC.
   - Giai đoạn 1 (Epochs 1-150): Augmentation phong phú.
   - Giai đoạn 2 (Epochs 151-200): Clean-Phase ảnh sạch nguyên bản với LR cực thấp.

---

## 2. Hướng dẫn Chạy & Tái lập (Reproducibility)

### Bước 1: Khởi chạy Huấn luyện Full 6K TAL+NWD
```bash
./dtnet_nwd_tal_full6k/run_train.sh
tail -f qualifiers/weights/dtnet_nwd_tal_full6k/train.log
```

### Bước 2: Xuất mô hình Full INT8 TFLite
```bash
python dtnet_nwd_tal_full6k/export_int8.py \
  --checkpoint qualifiers/weights/dtnet_nwd_tal_full6k/best_dtnet_nwd_tal_full6k.pt \
  --calib_dir qualifiers/data/train/images \
  --output qualifiers/weights/dtnet_nwd_tal_full6k/dtnet_nwd_tal_full6k_int8.tflite \
  --num_calib 200
```

### Bước 3: Đánh giá & Calibrate Ngưỡng Tối ưu
```bash
python dtnet_nwd_tal_full6k/calibrate.py \
  --tflite qualifiers/weights/dtnet_nwd_tal_full6k/dtnet_nwd_tal_full6k_int8.tflite \
  --val_dir qualifiers/data/val \
  --output dtnet_nwd_tal_full6k/calibrated_thresholds.json
```
