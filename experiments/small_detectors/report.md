# EAI Lần 2 · thử nghiệm Detect dưới 200.000 tham số

**Smoke ban đầu.** FOMO có **20.952** tham số và YOLO kiểu anchor hai mức có **99.928**, tính cả mean/variance của BatchNorm. Cả hai dùng backbone ImageNet đã nạp, head Detect 7 lớp khởi tạo ngẫu nhiên, đã chạy inference FP32 và TFLite INT8 trên PC. Kết quả huấn luyện ngắn trên dữ liệu EAI được ghi riêng ở mục pilot cuối báo cáo.

## Luật, dữ liệu và nguồn trọng số

- Thể lệ gốc mục V trang 5 quy định một model Detect, một lượt mỗi ảnh, cấm ensemble/TTA, đầu vào tối đa 320 × 240, tối đa 200.000 tham số và tệp TFLite INT8 tối đa 1,5 MB. Toán tử phải được phiên bản TFLM của bộ mã nguồn BTC hỗ trợ. Hai mẫu lab giữ thứ tự 7 lớp của release gồm bottle, cup, phone, book, mouse, box và toy_car. [PDF gốc](<../../docs/The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf>) · [Danh sách lớp](../../artifacts/release_frozen/infer.py)
- `qualifiers/data`, `qualifiers/data_incoming` và `qualifiers/data_stress_test` có **0 tệp**. `DuyToan/splits` có 4.803 ID train và 1.195 ID val nhưng thiếu cặp ảnh–nhãn đọc được. ZIP BTC chứa notebook, model và submission. Các ZIP Downloads được rà có ảnh rời hoặc tài liệu, chưa có cặp ảnh–nhãn EAI xác minh được. Smoke ban đầu không fit. Ảnh thử là ba vùng `00011`, `00044`, `00066` của fixture preview DuyToan local đã vẽ hộp và chữ. [Preview DuyToan epoch 392 trong Git](../archive/DuyToan/best/val_preview.png) có byte khác fixture lịch sử, nên số dự đoán và latency khi chạy smoke mới có thể khác. Cả hai model nhận cùng ảnh RGB resize `INTER_AREA` về 128 × 96, chuẩn hóa `pixel/127.5 - 1`. Preview chỉ dùng smoke. PTQ dùng 21 tensor RGB tổng hợp.
- **Backbone dùng thực tế:** Keras MobileNetV2 **0.35**, ImageNet, no-top, checkpoint 128. Keras công khai ImageNet weights từ alpha 0.35 trở lên. Edge Impulse có FOMO alpha 0.1, nhưng chưa nạp được checkpoint 0.1 có nguồn xác minh trong phiên này. File 0.35 được đọc từ cache `~/.keras/models`. Cả ba host tải đã thử trong shell (Google Storage, GitHub, raw GitHub) đều lỗi DNS, nên checksum bên dưới xác định byte **cache đã chạy**, chưa đối chiếu với bản tải mới. [FOMO của Edge Impulse](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo) · [Cấu hình FOMO 0.1](https://studio.edgeimpulse.com/public/26544/latest/impulse/1/learning/keras-object-detection/24) · [Nguồn Keras](https://github.com/keras-team/keras/blob/master/keras/src/applications/mobilenet_v2.py) · [URL H5 chính thức](https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_128_no_top.h5)

### Sàng lọc checkpoint Detect

| Ứng viên | Bằng chứng và quyết định |
| --- | --- |
| YOLO-FastestV2 COCO | Tác giả công bố khoảng 250.000. Đếm từ graph ncnn tối ưu chính chủ được **241.344 kernel + 4.438 bias = 245.782** trên 79 convolution. Loại theo trần ngay cả sau gộp BN. Checkpoint tác giả là `coco2017-0.241078ap-model.pth`. Phiên này đọc graph và kiểm lại phép đếm, chưa tải hoặc invoke checkpoint COCO. [Repo tác giả](https://github.com/dog-qiuqiu/Yolo-FastestV2) · [Graph ncnn](https://raw.githubusercontent.com/dog-qiuqiu/Yolo-FastestV2/main/sample/ncnn/model/yolo-fastestv2-opt.param) · [Định dạng ncnn](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure) |
| Các gói YOLO công khai khác đã rà | YOLO-Fastest 1.1 khoảng 350.000 và tinyYOLO-det khoảng 260.000 theo repo tác giả. YOLO11n có 2.616.248 và YOLO26n có 2.408.932 theo số đếm fused của Ultralytics. Loại theo số công bố, chưa cân lại checkpoint trong shell. [YOLO-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest#evaluating-indicatorbenchmark) · [tinyYOLO](https://github.com/ShMazumder/tinyYOLO#model-variants) · [Đếm của Ultralytics](https://docs.ultralytics.com/guides/yolo-architecture/) |
| PhiNets + YOLOv2 | Bài báo có cấu hình detection công bố **14.300–61.200** tham số, đáng lưu ý cho trần này. Các nguồn tác giả đã kiểm tra cung cấp kiến trúc. Chưa xác định được checkpoint detector tương ứng có thể tải và đối chiếu trong phiên này. [Bảng 2 bài báo](https://arxiv.org/html/2110.00337#S4.T2) · [Mã tác giả](https://github.com/AlbertoAncilotto/phiNets) |
| DTNet-P2-Max NWD-TAL Full-6K EAI | Đọc `ema` của checkpoint local `best_dtnet_nwd_tal_full6k.pt` bằng `torch.load(..., weights_only=True)`. Đếm được 192.711 weight/bias + 4.936 BN mean/variance = **197.647**, hoặc **197.673** nếu cộng 26 bộ đếm BN. [Detect hiện hữu](../../artifacts/release_frozen/weights/dtnet_nwd_tal_full6k_int8.tflite) là tệp riêng đã khóa, 280.856 byte. Các kết quả lab bên dưới dùng hai head mới. |

## Cấu hình và số đo

| Mẫu | Backbone ImageNet gồm BN | Head/neck mới | Trainable | BN state | Tổng | TFLite INT8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FOMO, stride 8, 7 lớp + nền | 17.584 | 3.368 | 19.208 | 1.744 | **20.952** | **55.904 B** |
| YOLO-style hai mức stride 8/16, 3 anchors/mức | 91.600 | 8.328 | 93.768 | 6.160 | **99.928** | **203.688 B** |

Cả hai mẫu xây từ cùng MobileNetV2 Keras. FOMO cắt tại `block_6_expand_relu` và thêm head heatmap 1 × 1. YOLO-style dùng tap này và `block_12_add`, một nhánh ghép đặc trưng hai mức, hai head anchor với 4 giá trị tọa độ, objectness và 7 score mỗi anchor. Script băm biến backbone trước/sau `load_weights`, xác thực chính các biến đó nằm trong từng graph và ghi fingerprint trong `artifacts/results.json`. Weight/bias trainable gồm gamma/beta của BN. Hai head chưa được huấn luyện.

| Mẫu | Input chung NHWC | Output FP32 hoặc INT8 tùy bản | FP32 p50 / p95 | INT8 p50 / p95 |
| --- | --- | --- | ---: | ---: |
| FOMO | `[1,96,128,3]` | `[1,12,16,8]` | 2,113 / 3,584 ms | 0,906 / 1,179 ms |
| YOLO-style | `[1,96,128,3]` | `[1,12,16,36]` và `[1,6,8,36]` | 2,729 / 4,369 ms | 1,357 / 2,635 ms |

Intel i7-1255U, batch 1, một luồng, 5 lượt warm-up và 30 lượt đo mỗi mẫu/định dạng trên crop `00011`. FP32 dùng `tf.function`. INT8 dùng TF Lite CPU, tắt default delegates. Số đo gồm gọi runtime và đọc tensor, với INT8 gồm cả lượng tử hóa input và khử lượng tử output trên PC. Decode ảnh, resize JPEG/PNG và postprocess nằm ngoài phép đo. Sáu ca INT8 được invoke lại độc lập bằng LiteRT 2.1.0.

Kiểm tra PC bổ sung chạy entrypoint Detect của release trên cùng ba crop, lưu tạm dưới dạng PNG. DTNet trả 100 hộp mỗi crop theo cap của release. Cả ba ca qua kiểm tra ID lớp và tọa độ hộp. Đây là đối chiếu đường suy luận trên preview đã vẽ hộp. Số hộp chạm cap không dùng để xếp hạng model. Log tải dữ liệu ghi 5.400 ảnh train và 600 ảnh val từ 09/11, nhưng lần rà lại hiện tại vẫn thấy 0 tệp trong các thư mục dữ liệu khai báo. Pilot có nhãn ở phần sau dùng split riêng do Điều phối bàn giao.

FOMO gom các ô cùng lớp liền kề thành centroid rồi đếm theo lớp và theo vùng trái/giữa/phải. YOLO decode bbox `xyxy` và NMS theo lớp. Ở crop `00011`, FOMO INT8 có 26 cụm, chia 6/10/10 theo trái/giữa/phải, ví dụ `phone` tại `[60.06,53.83]` pixel đầu vào. YOLO INT8 chạm cap 100 ở cả ba crop, ví dụ hộp `mouse` `[12.06,90.27,128.0,94.47]` ở crop `00011`. Đây là đầu ra của head chưa học, có nhiều dương tính giả. Centroid và bbox phục vụ hai phép đo khác nhau. Số cụm, số hộp, ví dụ đầu ra và sai khác logit FP32–INT8 của từng crop được lưu trong `artifacts/results.json`.

### Audit INT8 và TFLM

- Cả hai file dùng I/O `int8`, graph chỉ có tensor `INT8` và `INT32` cho bias/tham số toán tử. FOMO dùng `CONV_2D v3`, `DEPTHWISE_CONV_2D v3`, `ADD v2`. YOLO thêm `CONCATENATION v2`, `RESIZE_NEAREST_NEIGHBOR v3`. NMS và centroid decoder chạy trên PC ngoài graph.
- Export YOLO ban đầu sinh `TILE v3` do cách Keras triển khai upsample bằng lặp. Phép đổi sang resize nearest kích thước tĩnh khớp **6.144/6.144** giá trị ở tensor neck mẫu. Audit graph sau đổi xác nhận `TILE` đã được thay bằng `RESIZE_NEAREST_NEIGHBOR`. TFLM upstream có resolver và kernel INT8 cho resize nearest. Chưa có xác nhận `AllocateTensors`, `Invoke`, arena hay thời gian trên phiên bản TFLM của BTC. [Resolver TFLM upstream](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) · [Kernel resize INT8](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc)

## Checksum và chạy lại

| Đầu vào/nguồn | SHA-256 |
| --- | --- |
| Keras H5 cache 0.35/128 no-top | `f555b66d2d4b2d89360b92aee6f36a17278a13596cdf9917620240507c20deef` |
| Graph ncnn YOLO-FastestV2 sau chuẩn hóa whitespace | `8626066252cd595094d5e292271d754a382cea58a69de9529421b8281201ddef` |
| Fixture preview DuyToan local của lượt đo | `c194ae385514c60b60b9df330bc8e581fbca779c54aad0f261e974aabbdd3152` |
| Checkpoint EMA DTNet EAI đã đọc | `e62a845150221087d28089480746b72287564db3e3eb48bc873fc087249ee001` |

Chạy từ gốc clone. Đặt và export `TF_PY` và `LITE_PY` trỏ tới Python của hai môi trường. Phiên bản đã dùng là Python 3.11.18, TensorFlow 2.19.0 cho `TF_PY` và LiteRT 2.1.0 cho `LITE_PY`. Cả hai dùng NumPy 1.26.4 và opencv-python 4.11.0.86. Runtime dùng [requirements của release](../../artifacts/release_frozen/requirements.txt). Đặt `MOBILENET_H5` thành đường dẫn H5 từ URL Keras ở trên. Runner kiểm SHA-256 trước khi nạp.

Smoke đọc preview qua `DuyToan/best/val_preview.png`. Để đối chiếu số đo lịch sử cần fixture đúng SHA-256 trong bảng. Hãy sinh model và `results.json` mới từ cùng preview trước khi chạy verifier. Model, artifact và dự đoán sinh local nằm ngoài Git.

```bash
lab=experiments/small_detectors
python3 -B "$lab/audit_public.py"
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 "${TF_PY:?}" -B "$lab/run_lab.py" \
  --weights "${MOBILENET_H5:?}" --out "$lab/artifacts"
"${LITE_PY:?}" -B "$lab/verify_artifacts.py" --baseline
(cd artifacts/release_frozen && sha256sum --check --strict SHA256SUMS)
```

Để đếm trực tiếp từ graph upstream, tải đúng `yolo-fastestv2-opt.param` từ URL chính chủ ở bảng trên rồi chạy `python3 -B "$lab/audit_public.py" --param "${YOLO_NCNN_PARAM:?}"` với `YOLO_NCNN_PARAM` trỏ tới tệp đã tải. Audit sẽ đối chiếu checksum sau chuẩn hóa whitespace và số convolution với bằng chứng đã ghi. Lượt chạy lại với fixture lịch sử cho hai FlatBuffer khớp từng byte và cùng kết quả smoke. Model INT8 và SHA-256 đầu ra được sinh trong `artifacts/` của lab.

## Pilot CPU có nhãn · 96 train, 24 val

Nguồn là 120 cặp từ tập **train công khai của EAI Lần 2 trên Kaggle** do Điều phối chuẩn bị, seed `20261120`. Đọc `sources.json` sau khi `failures: []`, kiểm tra đường dẫn nguồn của cả hai split nằm dưới `train/`, đủ cặp ảnh–nhãn, và hai split không trùng ID hoặc SHA-256 ảnh. Tập train có **655** vật thể, val có **163**, cả hai đủ 7 lớp. SHA-256 manifest và của từng split gồm byte ảnh/nhãn được ghi vào `pilot_artifacts/pilot_results.json` của lượt lịch sử. Split này dùng để đánh giá hai model lab. DTNet Full-6K đã học trên pool chứa các ảnh này, nên chỉ là mốc tham chiếu có khả năng trùng dữ liệu huấn luyện. [Cách gộp pool DTNet](../archive/dtnet_nwd_tal_full6k/train.py)

- Cả hai mẫu giữ đầu vào RGB **128 × 96**, backbone MobileNetV2 0.35 ImageNet và các thông số model từ smoke. Chạy CPU **4 epoch**, batch 8, 48 bước cập nhật mỗi model, Adam 0,002, thứ tự shuffle từng epoch dùng cùng seed. Chỉ train head FOMO và neck/head YOLO. SHA-256 backbone khớp trước/sau fit, SHA-256 phần huấn luyện đổi đúng dự kiến.
- FOMO dùng trọng số loss 12 cho ô có vật thể, giữ vật thể nhỏ hơn khi trùng ô. **13/655** gán bị trùng. YOLO ghép với anchor có IoU kích thước cao nhất, giữ gán có IoU cao hơn khi trùng slot. **4/655** gán bị trùng. Loss train từ **2,515 → 0,875** (FOMO) và **5,134 → 2,091** (YOLO). Fit tương ứng **0,949 s** và **1,849 s**, toàn lượt pilot **6,520 s** trên CPU.
- PTQ chỉ dùng **24 ảnh đầu của split train theo thứ tự manifest**. Cả hai artifact có I/O INT8, tensor nội bộ chỉ INT8/INT32, toán tử và phiên bản giữ như smoke. FOMO **55.968 byte**, YOLO **204.080 byte**. Mỗi biến thể được suy luận một lượt mỗi ảnh trong lần đánh giá cuối. Val chỉ dùng sau khi chốt epoch và PTQ, không chọn epoch hay ngưỡng theo val.

### FOMO · centroid, count và location

Ghép centroid 1:1 theo khoảng cách tăng dần, cùng lớp, bán kính tối đa **8 px** trên ảnh 128 × 96. Ngưỡng heatmap **0,18**. Vùng location là ba phần bằng nhau theo chiều ngang. Count MAE tính trị tuyệt đối trên tổng số vật thể mỗi ảnh. Cột location trên toàn bộ GT tính cả các vật thể bị bỏ sót.

| Định dạng | Centroid đúng / 163 | Precision / Recall / F1 | Count MAE/ảnh | Ảnh đếm đúng / 24 | Location trên cặp ghép | Location trên toàn bộ GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 39 | 20,00% / 23,93% / 21,79% | 2,417 | 2 | 97,44% | 23,31% |
| INT8 | 41 | 19,62% / 25,15% / 22,04% | 2,667 | 2 | 97,56% | 24,54% |

### YOLO và DTNet · bbox và count

mAP50 dùng [evaluator của repo](../../src/qualifiers/src/evaluation.py), trung bình 7 lớp ở IoU 0,50. Bbox quy về pixel 320 × 240. YOLO giữ score từ 0,02, NMS theo lớp IoU 0,50 và cap 100. DTNet dùng nguyên decoder release, sàn 0,02, NMS 0,65 và cap 100. Count dùng **score ≥ 0,30** cho cả hai detector bbox. Hai model có điểm score và NMS khác nhau, nên ngưỡng chung là quy ước đo của pilot.

| Định dạng | BBox mAP50 | Hộp vào phép chấm AP | Dự đoán ở ngưỡng count | Count MAE/ảnh | Ảnh đếm đúng / 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO FP32 | 0,0650% | 2.400 | 832 | 27,875 | 0 |
| YOLO INT8 | 0,0284% | 2.400 | 704 | 22,542 | 0 |
| DTNet INT8 · tham chiếu có trùng pool | 96,6983% | 799 | 165 | 0,333 | 17 |

YOLO chạm cap trên mọi ảnh và còn nhiều false positive sau 4 epoch. FOMO có độ đúng vùng cao khi đã ghép được centroid, trong khi mới bao phủ khoảng một phần tư số vật thể. Đây là pilot nhỏ với backbone đóng băng. Centroid F1 và bbox mAP50 là hai thước đo khác nhau. Kiểm tra này chạy trên PC. TFLM, kit, camera OV7670 và split test ẩn nằm ngoài phạm vi.

### Lệnh chạy pilot

Clone thiếu split EAI 96/24 của lượt đo. Đặt `EAI_DATASET` tới fixture được bàn giao với `train/{images,labels}`, `val/{images,labels}` và `sources.json` có `failures: []`, seed `20261120`, source `egde-ai-challenge-2026-lan-2-qualifiers` và danh sách cặp ở hai khóa `train` và `val`. Trong manifest, hai danh sách đều ghi đường dẫn gốc dưới `train/images/` và `train/labels/`. Runner ánh xạ từng cặp về thư mục của split. Cần đúng fixture lịch sử để đối chiếu số đo.

Runner dùng [evaluator trong Git](../../src/qualifiers/src/evaluation.py) để tính mAP50. Workspace phục hồi có thể dùng bản tương thích tại `qualifiers/evaluation.py`.

```bash
lab=experiments/small_detectors
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
  "${TF_PY:?}" -B "$lab/run_pilot.py" --weights "${MOBILENET_H5:?}" \
  --dataset "${EAI_DATASET:?}" --baseline-python "${LITE_PY:?}" \
  --out "$lab/pilot_artifacts"
(cd artifacts/release_frozen && sha256sum --check --strict SHA256SUMS)
```

Kết quả nằm trong `pilot_artifacts/`, tách khỏi `artifacts/` của smoke. Dùng `--out` nếu cần tạo một thư mục kết quả riêng. Hai tệp sinh local là `pilot_results.json` và `pilot_predictions.json`.

## Warm start COCO → EAI · đối chứng VQA

Đã chạy cả hai nhánh trên CPU. Nhánh 160 ảnh chọn trực tiếp từ COCO train2017 **chưa tải được ảnh**. Shell trả `curl: (6) Could not resolve host: github.com` với [URL ZIP nhãn](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip). ZIP nhãn sau đó có sẵn trong thư mục tạm do Điều phối chuẩn bị. Sau khi đọc ZIP, yêu cầu ảnh đầu tiên [000000500568.jpg](https://images.cocodataset.org/train2017/000000500568.jpg) trả `[Errno -3] Temporary failure in name resolution`. Danh sách 160 ID được lưu trong `selection.json` của lượt chuẩn bị local.

**Pilot thực tế dùng 22 ảnh COCO train2017** chọn có seed từ bản cache **COCO128** chính thức. Quy trình chỉ trích các JPEG được chọn từ archive có sẵn. Seed `20261120`, quota tối đa 8 ảnh/lớp, chọn lớp hiếm trước và loại trùng ID. Quota theo lớp EAI là 5 bottle, 7 cup, 5 phone, 3 book, 2 mouse. Sau khi giữ mọi bbox của 5 lớp, có 93 bbox, trong đó 2 mouse. Mapping ID YOLO COCO → EAI là `39→0, 41→1, 67→2, 73→3, 64→4`. `car` bị lọc cùng các lớp ngoài mapping. [COCO128 chính thức](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml) Manifest của lượt lịch sử được lưu local tại `coco_eai_artifacts/coco128_sources.json`.

Hai nhánh có cùng SHA-256 H5, seed, head khởi tạo, backbone đóng băng, split EAI 96/24, 128 × 96, Adam 0,002, batch 8 và 4 epoch EAI. Nhánh warm thêm 4 epoch COCO với softmax FOMO giới hạn ở background và 5 lớp tương ứng, còn YOLO chỉ tính class loss ở 5 lớp đó. Assertion xác nhận kernel và bias của hai kênh box/toy_car giữ nguyên trong COCO. Sang EAI, Adam khởi tạo lại và cả 7 lớp có loss. Cả hai nhánh dùng cùng 24 ảnh EAI train để hiệu chuẩn INT8. Tổng thời gian lượt đối chứng **7,081 s**, lượt warm **10,227 s**. Hai model giữ **20.952 / 99.928** tham số gồm BN, I/O INT8 và kích cỡ **55.968 / 204.080 byte**. Cả hai nhánh sinh `pilot_results.json` trong thư mục output tương ứng.

### Đáp án trên 24 ảnh EAI

Có **119 câu mẫu**, sinh cố định từ nhãn bbox EAI val. Mỗi intent có 24 câu trừ LOCATION có 23 do một ảnh thiếu lớp có vị trí vùng nhất quán. EXIST gồm 17 câu có và 7 câu không. Đáp án theo mục VI.3, COUNT là chuỗi số, EXIST là `Co/Khong`, LIST theo thứ tự lớp chuẩn trong lab, LOCATION chia ba phần theo tâm và OOS là `OOS`. Slot lớp được trích xuất từ câu hỏi bằng từ điển cố định. YOLO và DTNet dùng các bbox có score ≥0,30 để trả lời và FOMO dùng centroid từ ngưỡng heatmap 0,18. Với nhiều tâm cùng lớp, dùng vùng có nhiều tâm nhất, hòa thì ưu tiên từ trái sang phải. [PDF gốc, mục VI.1 và VI.3](<../../docs/The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf>) Câu hỏi, intent và đáp án GT được sinh trong `vqa/vqa_questions.json`.

**TextCNN đã khóa đạt 119/119 intent trên các mẫu câu này.** Vì intent khớp hết, số đúng của oracle intent và pipeline giống nhau trong từng ô. Hai chế độ được lưu riêng trong JSON. Kết quả này đo bài toán lab trên câu hỏi sinh từ nhãn. Điểm có trọng số dùng tỷ trọng mục VI.5 để mô phỏng cách chấm.

| INT8 | COUNT /24 | EXIST /24 | LIST /24 | LOCATION /23 | OOS /24 | Exact match oracle / pipeline | Điểm trọng số oracle / pipeline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FOMO ImageNet → EAI | 5 | 6 | 0 | 3 | 24 | 31,93% / 31,93% | 21,93% / 21,93% |
| FOMO ImageNet → COCO → EAI | 5 | 12 | 0 | 9 | 24 | **42,02% / 42,02%** | 34,13% / 34,13% |
| YOLO ImageNet → EAI | 3 | 17 | 0 | 9 | 24 | 44,54% / 44,54% | 36,32% / 36,32% |
| YOLO ImageNet → COCO → EAI | 5 | 17 | 0 | 10 | 24 | **47,06% / 47,06%** | 39,09% / 39,09% |
| DTNet frozen · tham chiếu trùng pool | 23 | 24 | 14 | 23 | 24 | 90,76% / 90,76% | 88,85% / 88,85% |

Ở FP32, exact match VQA của FOMO từ **32,77% → 40,34%** và YOLO từ **47,06% → 45,38%**. Mỗi nhánh sinh hai tệp local `vqa/vqa_results.json` và `vqa/vqa_answers.json`.

| Kiểm tra detector | ImageNet → EAI | ImageNet → COCO → EAI |
| --- | ---: | ---: |
| FOMO FP32 centroid F1 / count MAE | 21,79% / 2,417 | 30,86% / 1,750 |
| FOMO INT8 centroid F1 / count MAE | 22,04% / 2,667 | 31,07% / 2,000 |
| FOMO INT8 đúng vùng trên toàn bộ GT | 24,54% | 33,13% |
| YOLO FP32 bbox mAP50 / count MAE | 0,0650% / 27,875 | 0,0608% / 22,333 |
| YOLO INT8 bbox mAP50 / count MAE | 0,0284% / 22,542 | 0,0885% / 17,167 |

Trên split này, warm start cải thiện cả centroid và VQA của FOMO. YOLO vẫn chạm cap 100 hộp trên mọi ảnh khi chấm AP, và VQA FP32 giảm. LIST của cả hai mẫu lab vẫn 0/24. Mẫu COCO nhỏ, lệch lớp và câu hỏi theo mẫu giới hạn khả năng suy rộng. Các số đo này phục vụ sàng lọc trên PC. Chưa có xác nhận TFLM của BTC hoặc hiệu năng trên kit.

### Lệnh tái chạy hai nhánh

Dùng cùng môi trường, H5 và `EAI_DATASET` như trên. Đặt `COCO_LABELS_ZIP` tới `coco2017labels.zip`, `COCO128_ZIP` tới ZIP COCO128 chính thức và `COCO22_DIR` tới đích local mới cho 22 ảnh được chọn.

```bash
lab=experiments/small_detectors
python3 -B "$lab/prepare_coco.py" --labels "${COCO_LABELS_ZIP:?}" \
  --image-archive "${COCO128_ZIP:?}" --per-class 8 --out "${COCO22_DIR:?}"
export CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3
"$TF_PY" -B "$lab/run_pilot.py" --dataset "${EAI_DATASET:?}" --weights "${MOBILENET_H5:?}" \
  --baseline-python "${LITE_PY:?}" --out "$lab/eai_control_artifacts"
"$TF_PY" -B "$lab/run_pilot.py" --dataset "$EAI_DATASET" --weights "$MOBILENET_H5" \
  --baseline-python "$LITE_PY" --coco "$COCO22_DIR" --out "$lab/coco_eai_artifacts"
for arm in eai_control_artifacts coco_eai_artifacts; do
  "$LITE_PY" -B "$lab/evaluate_vqa.py" --dataset "$EAI_DATASET" \
    --predictions "$lab/$arm" --out "$lab/$arm/vqa"
done
(cd artifacts/release_frozen && sha256sum --check --strict SHA256SUMS)
```

Khi kết nối tới COCO khả dụng, dùng `prepare_coco.py` với cùng `--labels`, đích riêng và bỏ `--image-archive` lẫn `--per-class 8` để chuẩn bị tập 160 ảnh. Giữ cùng đối chứng và bộ câu hỏi nếu chạy phép đo đó. [Kế hoạch](coco_vqa_plan.md) · [Bộ tải COCO 8k đã khóa](../../src/data_tools/coco_eai/README.md)
