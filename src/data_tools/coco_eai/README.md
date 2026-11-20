# Tái tạo COCO EAI 8k

Gói này khóa đúng **8.000 ảnh COCO train2017** của lab EAI. Manifest giữ nguyên thứ tự ID và quota 1.600 ảnh chính cho mỗi lớp. Nhãn đã ánh xạ sang ID EAI 0–4 theo thứ tự `bottle, cup, phone, book, mouse`. `cell phone` thành `phone`. Box và toy_car thuộc chặng EAI riêng.

| Tệp | Vai trò |
| --- | --- |
| `restore.py` | Tải ảnh chọn lọc, tạo nhãn và kiểm tra bộ dữ liệu bằng Python stdlib |
| `sources.json.gz` | Bản gzip byte-identical của manifest canonical khi giải nén. Có ID, URL, kích cỡ và SHA-256 từng JPEG, từng dòng nhãn đã map và SHA-256 nhãn. |
| `README.md` | Cách chạy và bằng chứng kiểm tra |

Ảnh được GET bằng HTTPS từ endpoint S3 regional có trong manifest. Endpoint S3 global là đường thử lại cho **cùng object key**. Gói chỉ yêu cầu từng JPEG trong manifest. Nhãn được tái tạo từ các dòng khóa sẵn với ASCII và LF, gồm đúng một LF cuối tệp. Không cần tải lại gói nhãn Ultralytics để phục hồi. [Nguồn COCO](https://cocodataset.org/#download) · [Nguồn quy đổi COCO của Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

Nhãn trong manifest dựa trên annotation của **COCO Consortium**, cấp phép CC BY 4.0. Gói đã lọc còn năm lớp, đổi chỉ số lớp theo EAI và chuẩn hóa nhãn YOLO. Quyền sử dụng ảnh theo điều khoản của từng ảnh. [Điều khoản COCO](https://cocodataset.org/dataset/termsofuse.htm) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Chạy từ clone

Cần Python 3.8 trở lên, không cần cài thêm package. Đích mặc định là `datasets/coco2017_eai_8k` dưới gốc clone, được xác định theo vị trí script. Dung lượng JPEG theo manifest là **1.182.149.667 byte**, cộng metadata, nhãn và phần cấp phát của filesystem.

```bash
tool=src/data_tools/coco_eai/restore.py

python3 "$tool" --dry-run
python3 "$tool"
python3 "$tool" --verify
```

- `--dry-run` in số ảnh và URL mẫu, không ghi tệp hay truy cập mạng.
- Khi chạy lại lệnh tải, JPEG có đúng kích thước và SHA-256 được dùng lại. Nhãn sai hoặc thiếu được tạo lại từ manifest. Mỗi JPEG được thử tối đa 3 lượt, timeout socket 20 giây mỗi lượt. Chỉ công bố tệp qua rename sau khi kiểm tra kích thước và SHA-256. Mặc định có 8 tải đồng thời. Có thể chọn `--workers 1` đến `--workers 32`.
- `--verify` chỉ đọc, kiểm byte nhãn, kích thước và SHA-256 JPEG, `classes.txt` và `sources.json` nếu có. Khi phục hồi đủ 8.000 cặp, script kiểm tra toàn bộ rồi ghi `sources.json` bằng rename nguyên tử. Runner nhận cùng schema và byte manifest canonical. Nếu một bản đã ready cần sửa do có tệp sai, lượt restore gỡ marker cũ trước khi sửa và chỉ ghi lại sau khi kiểm tra đủ.
- Để thử riêng 3 ảnh, dùng đích tạm. `--limit` chọn tiền tố theo đúng thứ tự manifest. Chế độ này không phát hành marker toàn bộ.

```bash
sample="$(mktemp -d)/coco-eai"
python3 "$tool" --dest "$sample" --limit 3 --workers 3
python3 "$tool" --dest "$sample" --verify --limit 3
python3 "$tool" --dest "$sample" --limit 3
```

## Đã kiểm tra

- Manifest giải nén khớp từng byte với `sources.json` canonical. SHA-256 bản giải nén là `dc83fdfeeb9d62e772288e7a1ad94a69fe827cd2643e0e5b1d9a70f99e780227`. SHA-256 tệp gzip là `852f9d480dfad307ec194619428b4d74ba47364b2cbdefb85e8d5af10fbd4972`.
- Kiểm chỉ đọc **8.000/8.000** JPEG và nhãn cùng `classes.txt`. ID khớp `selection.json`, `retrieval.json` và tên file local.
- Đã tải thành công các ID `000000288507`, `000000375150`, `000000332836` từ S3 vào đích thử riêng. Chạy lần hai tải lại 0 JPEG. Sau khi sửa một byte của JPEG thử, verifier từ chối tệp và lệnh tiếp tục chỉ tải lại đúng JPEG lỗi. Các lượt tải thử giữ nguyên trạng thái chưa có `sources.json`.
- Fixture với dữ liệu giả lập xác nhận dry-run chỉ đọc, lỗi SHA-256 bị giới hạn 3 lần thử, lần resume tải 0 JPEG đã hợp lệ, lượt tải một phần không phát hành marker và toàn bộ fixture chỉ phát hành marker sau khi đủ. Kiểm đường dẫn mặc định từ bản sao gói ở một cây thư mục độc lập cũng đạt.
