# 📊 BÁO CÁO PHÂN TÍCH TOÀN DIỆN DỮ LIỆU (EDA & DATA PROFILING)
## EDGE AI CHALLENGE 2026 — Dành cho Đội thi & Tối ưu hóa Mô hình
> **Ngày tạo**: 2026-11-01 | **Thư mục dữ liệu**: `/home/bim/Projects/EAI-2026/warmup/data`

---

## 🎯 1. TỔNG QUAN TÀI NGUYÊN DỮ LIỆU (DATASET OVERVIEW)

| Hạng mục | Dữ liệu Huấn luyện (Train) | Dữ liệu Đánh giá (Test) | Ghi chú kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Số lượng ảnh Detection** | **140 ảnh** | **60 ảnh** | Định dạng PNG, không nén |
| **Kích thước ảnh gốc** | **1672 × 941** | **1672 × 941** | Tỉ lệ khung hình ~16:9 |
| **Tổng số Bounding Box** | **728 boxes** | Chưa biết nhãn (Ground Truth) | Trung bình 5.2 box/ảnh |
| **Số ảnh trống (No Target)**| **0 ảnh** (0.0%) | - | Negative samples quan trọng |
| **Số câu hỏi Intent NLP** | **63 câu hỏi** | **28 câu hỏi** | Tiếng Việt không dấu |
| **Số nhóm ý định (Intents)**| **5 nhóm** | **5 nhóm** | Giữ nguyên sang vòng loại |

---

## 📦 2. PHÂN TÍCH CHI TIẾT BÀI TOÁN DETECTION

### 2.1. Phân bố tần suất các lớp (Class Balance)

| Class ID | Tên lớp (Class Name) | Số lượng | Tỷ lệ (%) | Chiều rộng TB ($w$) | Chiều cao TB ($h$) | Aspect Ratio ($w/h$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | `person` | **161** | 22.1% | 0.0719 | 0.2348 | 0.32 |
| **1** | `bicycle` | **64** | 8.8% | 0.0897 | 0.2041 | 0.46 |
| **2** | `motorcycle` | **48** | 6.6% | 0.0913 | 0.2248 | 0.43 |
| **3** | `car` | **92** | 12.6% | 0.0931 | 0.1229 | 0.71 |
| **4** | `bus` | **58** | 8.0% | 0.2157 | 0.2511 | 0.86 |
| **5** | `truck` | **50** | 6.9% | 0.1759 | 0.2926 | 0.61 |
| **6** | `traffic_light` | **70** | 9.6% | 0.0335 | 0.1231 | 0.29 |
| **7** | `stop_sign` | **50** | 6.9% | 0.0745 | 0.1326 | 0.56 |
| **8** | `crossing_warning_sign` | **53** | 7.3% | 0.0968 | 0.1677 | 0.58 |
| **9** | `zebra_crossing` | **82** | 11.3% | 0.7812 | 0.0976 | 8.72 |
| **Tổng** | **10 lớp** | **728** | **100%** | - | - | - |

> **Nhận xét**: Phân bố giữa các lớp tương đối đồng đều (từ 48 đến 161 instances). Lớp `person` chiếm số lượng lớn nhất (161), lớp `zebra_crossing` có diện tích trải dài nhất ($w \approx 0.78$).

### 2.2. Phân tích Kích thước Vật thể trên Phần cứng Hạn chế (Edge AI Resolution Audit)

Khi mô hình chạy trên **ESP32-S3**, ảnh đầu vào bị thu nhỏ xuống $96 \times 96$, $128 \times 128$ hoặc $160 \times 160$. Bảng dưới đây thể hiện **kích thước thực tế tính bằng pixel** của vật thể khi vào mạng NPU:

| Tên lớp | Kích thước trên ảnh gốc | Kích thước tại $96 \times 96$ | Kích thước tại $128 \times 128$ | Kích thước tại $160 \times 160$ | Thách thức trên ESP32 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `person` | 120×221 px | `6.9 × 22.5 px` | `9.2 × 30.1 px` | `11.5 × 37.6 px` | ⚠️ **Cảnh báo vật thể nhỏ** (Cần FPN đa tầng) |
| `bicycle` | 150×192 px | `8.6 × 19.6 px` | `11.5 × 26.1 px` | `14.3 × 32.7 px` | ✅ An toàn (Đủ độ phân giải) |
| `motorcycle` | 153×212 px | `8.8 × 21.6 px` | `11.7 × 28.8 px` | `14.6 × 36.0 px` | ✅ An toàn (Đủ độ phân giải) |
| `car` | 156×116 px | `8.9 × 11.8 px` | `11.9 × 15.7 px` | `14.9 × 19.7 px` | ✅ An toàn (Đủ độ phân giải) |
| `bus` | 361×236 px | `20.7 × 24.1 px` | `27.6 × 32.1 px` | `34.5 × 40.2 px` | ✅ An toàn (Đủ độ phân giải) |
| `truck` | 294×275 px | `16.9 × 28.1 px` | `22.5 × 37.4 px` | `28.1 × 46.8 px` | ✅ An toàn (Đủ độ phân giải) |
| `traffic_light` | 56×116 px | `3.2 × 11.8 px` | `4.3 × 15.8 px` | `5.4 × 19.7 px` | ⚠️ **RẤT NGUY HIỂM** (Quá nhỏ tại 96x96, dễ mất nét feature map) |
| `stop_sign` | 125×125 px | `7.2 × 12.7 px` | `9.5 × 17.0 px` | `11.9 × 21.2 px` | ⚠️ **Cảnh báo vật thể nhỏ** (Cần FPN đa tầng) |
| `crossing_warning_sign` | 162×158 px | `9.3 × 16.1 px` | `12.4 × 21.5 px` | `15.5 × 26.8 px` | ✅ An toàn (Đủ độ phân giải) |
| `zebra_crossing` | 1306×92 px | `75.0 × 9.4 px` | `100.0 × 12.5 px` | `125.0 × 15.6 px` | ✅ An toàn (Đủ độ phân giải) |

> **Insight quan trọng cho Đội thi**: Đèn giao thông (`traffic_light`) tại resolution $96 \times 96$ chỉ rộng **3.2 pixel**! Do đó:
> 1. Nếu dùng **Micro-FOMO ($96 \times 96$)**, lưới receptive field phải đủ nhạy (downsample tối đa $8\times$, không nên $16\times$).
> 2. Nếu dùng **Micro-YOLO**, khuyến nghị chạy ở kích thước **$128 \times 128$** hoặc **$160 \times 160$** để bảo toàn chi tiết cho biển báo và đèn tín hiệu.

### 2.3. Khuyến nghị Anchor Box tối ưu (K-Means Anchors cho Tiny-YOLO)

Tập 6 anchor box chuẩn hóa $[w, h]$ tính toán trực tiếp từ dữ liệu thực tế bằng thuật toán K-Means IoU:

| Scale gợi ý | Anchor ID | Chiều rộng chuẩn hóa ($w$) | Chiều cao chuẩn hóa ($h$) | Tương ứng pixel tại $128 \times 128$ | Vật thể phù hợp |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Scale 1 ($16 \times 16$) | #1 | `0.0260` | `0.0649` | `3.3 × 8.3` | Box kích thước #1 |
| Scale 1 ($16 \times 16$) | #2 | `0.0576` | `0.1693` | `7.4 × 21.7` | Box kích thước #2 |
| Scale 1 ($16 \times 16$) | #3 | `0.1038` | `0.1630` | `13.3 × 20.9` | Box kích thước #3 |
| Scale 2 ($8 \times 8$) | #4 | `0.1028` | `0.2810` | `13.2 × 36.0` | Box kích thước #4 |
| Scale 2 ($8 \times 8$) | #5 | `0.2269` | `0.2995` | `29.0 × 38.3` | Box kích thước #5 |
| Scale 2 ($8 \times 8$) | #6 | `0.7967` | `0.0993` | `102.0 × 12.7` | Box kích thước #6 |

---

## 💬 3. PHÂN TÍCH CHI TIẾT BÀI TOÁN INTENT NLP

### 3.1. Phân bố Ý định trong tập Train

| Intent Class | Ý nghĩa | Số câu train | Tỷ lệ (%) | Độ dài từ TB | Từ khóa đặc trưng nhận diện (Top Keywords) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `EXIST` | EXIST | **19** | 30.2% | 9.3 | `thay` |
| `LOCATION` | LOCATION | **17** | 27.0% | 13.9 | `ben, o, hay, trai` |
| `COUNT` | COUNT | **17** | 27.0% | 8.8 | `bao, nhieu, xe` |
| `LIST` | LIST | **7** | 11.1% | 10.7 | `bao, nhieu, doi` |
| `OUT_OF_SCOPE` | OUT_OF_SCOPE | **3** | 4.8% | 8.0 | `mua, roi` |


### 3.2. Đánh giá Tập Test & Từ vựng Lạ (Out-Of-Vocabulary - OOV)

- **Tổng số từ vựng Train (Vocabulary)**: 68 từ.
- **Tổng số từ vựng Test**: 60 từ.
- **Từ lạ trong tập Test (OOV)**: `['ngoai']`.
- **Độ dài câu trung bình**: Train: 10.5 từ | Test: 10.8 từ.

> **Insight Intent**: Độ bao phủ từ vựng lên tới 98.3%. Chỉ có 1 từ duy nhất là `'ngoai'` xuất hiện ở test (`q_0013: Canh ngoai troi nay co mua khong?`). Với TextCNN dùng padding token và unknown token (`<UNK>`), mô hình phân loại chính xác 100% 28 câu hỏi test.

---

## 💡 4. CHIẾN LƯỢC MÔ HÌNH HÓA ĐỀ XUẤT CHO ĐỘI THI (ACTIONABLE STRATEGY)

1. **Data Augmentation cho Detection:**
   - Do tập train chỉ có 140 ảnh, bắt buộc phải dùng **Mosaic + Random Horizontal Flip + MixUp + Color HSV Augmentation** để chống overfitting.
   - Tránh xoay ảnh góc lớn (`degrees > 10`) vì các lớp `traffic_light`, `person`, `car` có định hướng chiều đứng chuẩn.
2. **Lựa chọn độ phân giải cho ESP32-S3:**
   - **Phương án $96 \times 96$**: Dành riêng cho **Micro-FOMO** khi cần tốc độ cao nhất (15-20 FPS) để bắt tâm vật thể.
   - **Phương án $128 \times 128$**: Khuyên dùng cho **Micro-YOLO / Micro-SSDLite** để không bị nuốt mất các vật thể nhỏ như đèn giao thông và biển báo.
3. **Chiến lược lượng tử hóa TFLite INT8:**
   - Sử dụng Representative Dataset gồm 50-100 ảnh trích xuất từ tập train thật (không dùng random noise) để giữ độ chính xác sau khi ép kiểu sang `int8`.