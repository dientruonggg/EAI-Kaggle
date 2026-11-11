# Edge AI Challenge 2026 (Qualifiers) - Champion Reproduction Package

Gói mã nguồn tối giản, độc lập và khép kín nhằm tái tạo 100% file kết quả `submission.csv` đạt kỷ lục điểm số **94.02+** trên bảng xếp hạng Kaggle.

---

## 1. Yêu cầu Hệ thống & Ràng buộc Phần cứng (ESP32-S3 Target)

Tất cả dự đoán được sinh ra hoàn toàn từ 2 mô hình **Full Integer INT8**, tuân thủ 100% giới hạn phần cứng vi điều khiển:
* **Mô hình BBox Detection**: `detection_champion_int8.tflite` (180,764 params, **248 KB** Flash).
* **Mô hình Intent NLP**: `text_cnn_int8.tflite` (17,408 params, **23 KB** Flash).
* **Tổng Flash Footprint**: **~271 KB** (thỏa mãn tuyệt đối giới hạn $\le 1.5$ MB).
* **Single Model Invariant**: Không sử dụng Ensemble, không sử dụng TTA, đúng 1 lần suy luận mỗi ảnh/câu hỏi.

---

## 2. Cài đặt Môi trường (Cực kỳ tối giản)

Chỉ yêu cầu Python 3.9+ và 3 thư viện cơ bản:

```bash
pip install -r reproduce/requirements.txt
```

---

## 3. Tái tạo `submission.csv` với 1 dòng lệnh

Chỉ cần chạy lệnh sau từ thư mục gốc của dự án:

```bash
python reproduce/run_reproduce.py \
  --test_images qualifiers/data/test/images \
  --qa_test qualifiers/data/qa_test.csv \
  --output submission.csv
```

### Quá trình thực thi tự động:
1. Nạp mô hình Object Detection INT8, suy luận 4,000 ảnh test.
2. Áp dụng thuật toán **Per-Class NMS** và **Sweetspot Density Calibration** (khống chế mật độ hộp chuẩn 43–47 boxes/ảnh).
3. Nạp mô hình Text-CNN INT8, phân loại 4,000 câu hỏi intent.
4. Xuất file `submission.csv` đúng 8,000 dòng dữ liệu chuẩn Kaggle.

---

## 4. Cấu trúc Thư mục

```
reproduce/
├── README.md                      # Hướng dẫn tái tạo
├── requirements.txt               # Dependencies tối giản (numpy, cv2, tensorflow)
├── run_reproduce.py               # Script suy luận duy nhất (~180 dòng code sạch)
└── weights/
    ├── detection_champion_int8.tflite  # Model phát hiện vật thể INT8 (248 KB)
    ├── text_cnn_int8.tflite            # Model phân loại ý định INT8 (23 KB)
    └── text_cnn_vocab.json             # Bảng từ vựng mã hóa câu hỏi (3.7 KB)
```
