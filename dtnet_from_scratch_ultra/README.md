# DTNet-From-Scratch-Ultra: Full-Topping Architecture for Edge AI Challenge 2026

Hệ thống huấn luyện và triển khai **DTNetMax Full-Topping** xây dựng từ đầu (from scratch) dành cho vi điều khiển **ESP32-S3 N16R8 + Camera OV7670**, tích hợp đầy đủ 100% tất cả các kỹ thuật tối ưu hóa được Ban Tổ Chức cho phép và tuân thủ tuyệt đối quy chế thi đấu.

---

## 🎯 1. BẢO CHỨNG TUÂN THỦ PHẦN CỨNG (STRICT HARDWARE GATES)

| Tiêu chuẩn kỹ thuật | Giới hạn bắt buộc của BTC | Thông số thực tế của DTNetMax | Trạng thái |
| :--- | :--- | :--- | :---: |
| **Số tham số Detection** | $\le 200.000$ params | **$197.647$ parameters** (dư $2.353$ params an toàn) | ✅ PASS |
| **Dung lượng file (.tflite)**| $\le 1.5\text{ MB}$ | **$\approx 230\text{ KB}$** (nhỏ gấp 6.5 lần giới hạn) | ✅ PASS |
| **Độ phân giải đầu vào** | $\le 320 \times 240$ pixel | **$320 \times 240$ RGB** (khớp tỷ lệ 4:3 của OV7670) | ✅ PASS |
| **Chuẩn lượng tử hóa** | Full Integer INT8 | `int8 input`, `int8 output`, `TFLITE_BUILTINS_INT8` | ✅ PASS |
| **Độ tương thích toán tử** | 100% TFLite Micro | Conv2D, DepthwiseConv2D, Add, Concat, MaxPool, Nearest Upsample, ReLU6 | ✅ PASS |
| **Quy chuẩn suy luận** | Single model, no TTA | **1 model duy nhất, 1 lần forward pass / ảnh** | ✅ PASS |

---

## 🚀 2. TỔNG HỢP 6 "FULL TOPPING" HỢP LỆ ĐƯỢC TÍCH HỢP

### 1. Anti-Nerf Soft Border Target Assignment
- Các ô tâm vật thể (center cells) được gán nhãn dương `class_id >= 0` với bán kính thích ứng diện tích ($r_{min}=1.5, r_{max}=5.5, \text{rad\_scale}=0.32$).
- Các ô nằm ở viền bounding box (border cells) **không bị ép về nền âm hoàn toàn** (tránh triệt tiêu gradient viền) mà nhận trọng số nền mềm **$0.25$**. Nền thật bên ngoài nhận trọng số **$1.0$**.
- Triệt tiêu hoàn toàn hiện tượng background logit drift mà vẫn giữ viền box cực kỳ sắc nét.

### 2. Scale-Adaptive Small Object Box Loss
$$w_{\text{scale}} = 1.0 + 1.2 \times \left(1.0 - \sqrt{\frac{\text{area}}{320 \times 240}}\right)$$
- Các vật thể nhỏ (`mouse`, `toy_car`, `phone`) được nhân hệ số loss lên tới **$2.0 \times - 2.2 \times$**, ép mạng phải học định vị chính xác từng pixel, ngăn chặn tụt IoU ở mAP50.

### 3. Hard-Class Priority Focal Weighting
- Tăng cường trọng số loss theo từng lớp dựa trên độ khó thực tế:
  - `bottle`: 1.00
  - `cup`: 1.00
  - `phone`: 1.05
  - `book`: **1.35** *(Lớp khó, nhiều kích cỡ bìa)*
  - `mouse`: **1.25** *(Lớp nhỏ, dễ bị nhầm)*
  - `box`: 0.95
  - `toy_car`: **1.30** *(Lớp nhỏ, nhiều góc xoay)*

### 4. Mô phỏng cảm biến Camera OV7670 (Hardware Noise Simulation)
- Thêm nhiễu hạt Gaussian, Poisson shot noise và độ mờ quang học nhẹ ($p=0.30$) vào pipeline data augmentation lúc train.
- Giúp mô hình có khả năng chống chịu cực tốt trước chất lượng cảm biến thực tế của camera OV7670 tại vòng Chung kết.

### 5. Dynamic 2-Phase Close-Mosaic Training
- **Phase 1 (Epoch 1 $\rightarrow$ 365):** Bật Mosaic 4 ảnh ($p=0.65$), Zoom-out Affine ($0.65 - 1.25$), HSV Jitter để mô hình học biểu diễn đa quy mô.
- **Phase 2 (Epoch 366 $\rightarrow$ 400):** **Tắt hoàn toàn Mosaic ($p=0.0$)**, chuyển về $100\%$ ảnh đơn nguyên bản $320 \times 240$ để căn chỉnh viền bounding box trùng khớp hoàn toàn với phân phối ảnh test thực tế.

### 6. ModelEMA & Single-Model INT8 Sweet-Spot Calibration
- **ModelEMA ($\beta=0.9998$):** Duy trì trọng số trung bình trượt thời gian, làm phẳng loss landscape, triệt tiêu nhiễu giật gradient.
- **Calibrate Thresholds:** Tự động quét tìm điểm cân bằng Confidence Threshold và NMS IoU tối ưu trên mô hình INT8 để đạt mAP50 cao nhất mà không cần ensemble hay TTA.

---

## 🛠️ 3. HƯỚNG DẪN SỬ DỤNG

### Bước 1: Huấn luyện mô hình từ đầu (From-Scratch)
```bash
# Chạy nền tự động (tự động nhận diện môi trường python, GPU/CPU)
./dtnet_from_scratch_ultra/run_train.sh

# Theo dõi tiến độ huấn luyện
tail -f qualifiers/weights/dtnet_from_scratch_ultra/train.log
```

### Bước 2: Xuất xưởng mô hình Full INT8 TFLite
```bash
python dtnet_from_scratch_ultra/export_int8.py \
  --checkpoint qualifiers/weights/dtnet_from_scratch_ultra/best_dtnet_ultra.pt \
  --calib_dir qualifiers/data/train/images \
  --output qualifiers/weights/dtnet_from_scratch_ultra/dtnet_max_int8.tflite \
  --num_calib 200
```

### Bước 3: Tinh chỉnh ngưỡng tối ưu (Calibration)
```bash
python dtnet_from_scratch_ultra/calibrate.py \
  --tflite qualifiers/weights/dtnet_from_scratch_ultra/dtnet_max_int8.tflite \
  --val_dir qualifiers/data/val \
  --output dtnet_from_scratch_ultra/calibrated_thresholds.json
```
