# EAI-2026 · Baseline vòng loại 98.50673

Gói inference chạy trên máy, cố định theo notebook SOTA của gói BTC vòng loại. Trong workspace chưa có bằng chứng xác nhận cặp trọng số cuối cho chung kết. Firmware ESP32-S3 và dữ liệu chấm đầy đủ chưa được kiểm chứng với gói này.

| Tệp | Vai trò |
| --- | --- |
| `infer.py` | Entrypoint Detect, Intent và xuất CSV theo notebook |
| `requirements.txt` | Phiên bản đã kiểm tra trên Linux x86-64, CPython 3.11 |
| `weights/dtnet_nwd_tal_full6k_int8.tflite` | Detect DTNet-P2-Max NWD-TAL Full-6K, 280.856 byte |
| `weights/text_cnn_int8.tflite` | Intent TextCNN, 38.592 byte |
| `weights/text_cnn_vocab.json` | Vocab Intent của Cell 6, 3.568 byte |
| `SHA256SUMS`, `verify_provenance.py` | Kiểm tra gói và đối chiếu nguồn BTC |

## Chạy

Chạy từ gốc workspace để đặt biến đường dẫn. Đặt môi trường Python bên ngoài thư mục bàn giao. Cài dependency một lần, sau đó có thể suy luận offline.

```bash
release="$(pwd)/release_frozen"
eai_venv="$(mktemp -d)/venv"
uv venv --python 3.11 "$eai_venv"
uv pip install --python "$eai_venv/bin/python" -r "$release/requirements.txt"

cd /tmp
"$eai_venv/bin/python" "$release/infer.py" detect "/path/to/image.png"
"$eai_venv/bin/python" "$release/infer.py" intent 'Có bao nhiêu chai nước?'
"$eai_venv/bin/python" "$release/infer.py" submission \
  --images "/path/to/test/images" --questions "/path/to/questions.csv" > /tmp/eai_predictions.csv
```

Detect đọc ảnh bằng OpenCV, chuyển BGR sang RGB, resize `INTER_AREA` về 320 × 240 khi cần và lượng tử hóa input theo tensor. Giữ stride 8, score ≥ 0.020, NMS theo lớp với IoU 0.65 và tối đa 100 hộp. Đầu ra JSON gồm `class_id`, `score` làm tròn 5 chữ số và `bbox` dạng `[x1,y1,x2,y2]` làm tròn 2 chữ số trong hệ tọa độ 320 × 240. Thứ tự lớp là `bottle, cup, phone, book, mouse, box, toy_car`.

Intent trả một nhãn trong `COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE`. Chuẩn hóa tiếng Việt, bỏ stopword, cắt hoặc pad ở 18 token và áp dụng bộ lọc thực thể của notebook. Model Intent thực tế nhận ID token `int32` và trả điểm `float32`.

CSV lô giữ header `task_type,sample_id,prediction`. Detect đọc ảnh `.jpg` và `.png` theo tên đã sắp xếp và dùng stem làm ID. CSV câu hỏi dùng một cột ID (`question_id`, `sample_id` hoặc `id`) và một cột nội dung (`question` hoặc `text`). Kết quả Detect đứng trước Intent. Đường dẫn input do người chạy truyền vào được hiểu theo cwd hiện tại. Trọng số và vocab luôn được đọc theo vị trí `infer.py` và kiểm tra SHA-256 trước khi dùng.

## Nguồn và kiểm chứng

Notebook [98.50673 SOTA](../submission_package_btc/DT2_EdgeAI2026_Qualifiers_Submission_98506_SOTA.ipynb) đặt tên cặp model ở Cell 6, dùng decoder ở Cell 4, xử lý Intent ở Cell 5 và chạy inference ở Cell 9. [README gói BTC](../submission_package_btc/README.md) ghi điểm vòng loại và liệt kê Clean 200ep là bản đối sánh 98.47943. SHA-256 của Detect Clean 200ep là `87f2678f25b3e3a93261f733251661f108e21f24f9f95b17000f9f65330ab0be`.

```bash
(cd "$release" && sha256sum --check --strict SHA256SUMS)
python "$release/verify_provenance.py"
```

Lệnh thứ hai cần cấu trúc workspace gốc. Nó so từng byte giữa ba asset trong thư mục bàn giao, thư mục BTC, ZIP BTC và dữ liệu nhúng ở Cell 6. SHA-256 của ZIP gốc là `a0de165aed2ea65312222f2fc19fc4af1f9351f314b04f0c4d0a21fb83231b75`. SHA-256 của notebook SOTA là `ad2c3aa81b87d095d12590f964a84e5ce616515ec052264c9ef73477bddfebfc`.

Kiểm tra đối chiếu chạy bản bàn giao bằng LiteRT 2.1.0 và phần inference gốc của notebook bằng TensorFlow 2.19.0. Hai CSV khớp từng byte trên 3 ảnh (ảnh preview của DuyToan, một PNG 320 × 240 và một JPG cần resize được tạo bằng seed 314159) cùng 8 câu hỏi tiếng Việt phủ cả 5 nhãn. Tổng cộng 236 hộp. SHA-256 chung của CSV đối chiếu là `2da6132fa57cb3372900a318bad616808b1bd2389024b554d4c3e1df269eaa41`.
