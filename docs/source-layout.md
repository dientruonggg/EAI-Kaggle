# Đường dẫn source EAI-2026

Đường dẫn `qualifiers/`, `release_frozen/`, các tên thí nghiệm cũ, `roadmap.md` và `BENCHMARK_MATRIX.md` dùng symlink tương đối. Source chuẩn nằm ở `src/qualifiers/`, thí nghiệm cũ ở `experiments/archive/` và tài liệu ở `docs/`.

Các đường dẫn `qualifiers/data*`, `qualifiers/weights`, `qualifiers/candidate_submissions` và `qualifiers/submissions` trỏ về `datasets/qualifiers/` hoặc `artifacts/`. Khi một tiến trình ghi thay thế tệp bằng atomic rename, dùng đường dẫn đích thực để giữ symlink.

Dữ liệu thi, bộ ảnh COCO, gói ZIP nộp BTC và các output mới của máy chạy giữ ở workspace local. Các tệp đã theo dõi từ trước vẫn có trên nhánh. Pipeline ngoài phạm vi di chuyển, gồm `warmup/` và `reproduce/`, vẫn ở vị trí cũ.

## Kiểm tra baseline

    (cd artifacts/release_frozen && sha256sum --check --strict SHA256SUMS)

Chạy thử Detect, Intent hoặc CSV theo [README baseline](../artifacts/release_frozen/README.md). `verify_provenance.py` cần thư mục và ZIP BTC do đội lưu riêng bên cạnh `release_frozen/` trong `artifacts/`. Baseline này là bản vòng loại 98.50673. Kết quả firmware ESP32-S3 và phép tái huấn luyện đầy đủ cần được kiểm riêng.
