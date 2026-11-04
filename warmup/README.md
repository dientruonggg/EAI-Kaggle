# EDGE AI CHALLENGE 2026 (LẦN 2) - WARM-UP

- **Kaggle URL**: https://www.kaggle.com/competitions/edge-ai-challenge-2026-lan-2
- **Competition Slug**: `edge-ai-challenge-2026-lan-2`
- **Deadline**: 2026-11-07 16:59:59 (còn ~6 ngày)
- **Chủ đề**: Visual Question Answering (Object Detection 10 classes + Intent Classification 5 classes).

---

## 1. Cấu trúc thư mục

```
warmup/
├── data/
│   ├── classes.txt               # 10 nhãn detection
│   ├── intent_train.csv          # 63 câu train intent
│   ├── intent_test.csv           # 28 câu test intent
│   ├── sample_submission.csv     # File nộp mẫu
│   ├── train/images/             # 140 ảnh train
│   ├── train/labels/             # 140 file nhãn YOLO (cx cy w h chuẩn hóa 0-1)
│   └── test/images/              # 60 ảnh test (img_0001.png ... img_0060.png)
├── submissions/                  # Chứa các file kết quả nộp bài
├── src/                          # Source code huấn luyện & suy luận
│   ├── detection/
│   └── intent/
├── submit.py                     # Script validate định dạng & nộp trực tiếp lên Kaggle
└── README.md
```

---

## 2. Các lệnh Kaggle CLI thực chiến

### Nộp bài trực tiếp (kèm auto-validate định dạng):
```bash
python3 submit.py submissions/my_submission.csv "Mô tả bài nộp"
```

### Nộp bằng lệnh `kaggle` gốc:
```bash
kaggle competitions submit -c edge-ai-challenge-2026-lan-2 -f <path_to_csv> -m "Mô tả bài nộp"
```

### Xem lịch sử nộp & điểm số (Public / Private Score):
```bash
kaggle competitions submissions -c edge-ai-challenge-2026-lan-2
```

### Xem bảng xếp hạng (Leaderboard):
```bash
kaggle competitions leaderboard -c edge-ai-challenge-2026-lan-2 --show
```

---

## 3. Quy cách định dạng Submission

File `submission.csv` gồm đúng 89 dòng (1 header + 60 detection + 28 intent):

```csv
task_type,sample_id,prediction
DETECTION,img_0001,"[{""class_id"":0,""score"":0.91,""bbox"":[12,40,98,220]}]"
DETECTION,img_0002,[]
...
INTENT,q_0001,COUNT
INTENT,q_0002,EXIST
...
```

> **Lưu ý quan trọng**:
> - `DETECTION`: `bbox` là pixel toạ độ tuyệt đối `[x_min, y_min, x_max, y_max]` trên kích thước ảnh gốc (không phải dạng chuẩn hóa 0-1). Nếu không có vật thể ghi `[]`.
> - `INTENT`: Chỉ gồm 1 trong 5 nhãn: `COUNT`, `EXIST`, `LIST`, `LOCATION`, `OUT_OF_SCOPE`.
> - Công thức điểm: `Score = 0.70 * (mAP50 * 100) + 0.30 * (Accuracy * 100)`.
