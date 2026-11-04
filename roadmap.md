Lộ trình các Kaggle Notebook và Competition dưới đây nối tiếp danh sách kiến thức trong bảng Notion của bạn — đi từ xử lý ảnh truyền thống, trích xuất đặc trưng, các kiến trúc CNN làm Backbone (VGG-16, ResNet, MobileNet, ConvNeXt) và Transfer Learning, cho đến các mô hình Phát hiện vật thể (Object Detection) từ cổ điển đến hiện đại kèm kỹ thuật tối ưu thực chiến trên Kaggle.

---

## 1. Giai đoạn "Cổ thật cổ": Xử lý ảnh & Machine Learning truyền thống (Pre-Deep Learning)

Giai đoạn này khớp trực tiếp với các mục **Image Processing**, **Feature Extraction** (Edge Detection, Corner Detection, Keypoint Descriptors, Haar feature, HOG) và **Project** (Face Detection, Object Detection using Background Subtraction) trong lộ trình của bạn:

* **Xử lý ảnh nền tảng, Thresholding, Edge Detection & Background Subtraction:**
* **Link Kaggle:** [OPENCV IN-DEPTH COURSE - Image Processing, Haar & HOG](https://www.kaggle.com/code/mlvprasad/opencv-in-depth-course-2023-for-indian-kaggler)
* **Trọng tâm:** Thực hành từ cân bằng sáng, lọc nhiễu, tìm cạnh (Sobel, Canny) đến phát hiện đối tượng bằng Haar Cascades và HOG (Histogram of Oriented Gradients).




* **Tách nền chuyển động (Background Subtraction bằng Frame Difference):**
* **Link Kaggle:** [Background Subtraction Using Frame Difference](https://www.kaggle.com/code/mdaqdouqi/assignment-1)
* **Trọng tâm:** Ứng dụng `cv2.absdiff()` và `cv2.threshold()` để tách vật thể chuyển động khỏi nền tĩnh — đúng với bài tập *Object Detection using Background Subtraction* trong bảng của bạn.




* **Phân đoạn ảnh nâng cao & Trích xuất đặc trưng (Morphology, Watershed, ORB):**
* **Link Kaggle:** [OpenCV ORB Detector, BFMatcher & Image Segmentation](https://www.kaggle.com/code/shiblinomani/opencv-orb-detector-bfmatcher-image-segmentation)
* **Trọng tâm:** Kết hợp xử lý hình thái học (Morphology), phân đoạn ảnh (Otsu, Watershed) và Keypoint Descriptors.




* **Thuật toán HOG + SVM (Đỉnh cao thời kỳ trước CNN):**
* **Link Kaggle:** [Support Vector Machine Object Detection](https://www.kaggle.com/code/mehmetlaudatekman/support-vector-machine-object-detection)
* **Trọng tâm:** Dùng cửa sổ trượt (Sliding Window), trích xuất đặc trưng hình dáng bằng HOG và đưa vào bộ phân loại SVM.





---

## 2. Giai đoạn Two-Stage Detectors: Kỷ nguyên R-CNN & CNN Backbones

Sau khi nắm vững các kiến trúc CNN (LeNet, AlexNet, VGG-16, ResNet, GoogLeNet) và Transfer Learning, người ta dùng các mạng CNN này làm **Backbone** trích xuất đặc trưng kết hợp với **Region Proposal Network (RPN)** để sinh vùng chứa vật thể:

* **Faster R-CNN lý thuyết & thực hành từ gốc:**
* **Link Kaggle:** [Faster RCNN: A Gentle Introduction!](https://www.kaggle.com/code/debanga/faster-rcnn-a-gentle-introduction)
* **Trọng tâm:** Giải thích sự tiến hóa từ R-CNN (2014) $\rightarrow$ Fast R-CNN $\rightarrow$ Faster R-CNN và cách dùng Transfer Learning tinh chỉnh đầu ra (Box Predictor) trên PyTorch.




* **Faster R-CNN với Backbone ResNet-50 (Baseline chuẩn trên Kaggle):**
* **Link Kaggle:** [Object Detection Starter - FasterRCNN with ResNet50](https://www.kaggle.com/code/cevangelist/object-detection-starter-fasterrcnn-with-resnet50)
* **Trọng tâm:** Cách ghép nối kiến trúc ResNet kèm Feature Pyramid Network (FPN) vào bài toán phát hiện vật thể thực tế.





---

## 3. Giai đoạn One-Stage Detectors & Tối ưu hóa thực chiến (EfficientDet, YOLOv5)

Thay vì tách làm 2 bước (tìm vùng + phân loại), các mô hình One-Stage dự đoán thẳng tọa độ hộp giới hạn (Bounding Box) và nhãn trong 1 lần chạy. Đây cũng là giai đoạn xuất hiện nhiều kỹ thuật **tối ưu hóa mô hình** đặc trưng trên Kaggle:

* **EfficientDet + PyTorch Lightning + Các kỹ thuật Tối ưu hóa (All-in-One):**
* **Link Kaggle:** [EfficientDet meets Pytorch Lightning](https://www.kaggle.com/code/yassinealouini/efficientdet-meets-pytorch-lightning)
* **Trọng tâm:** Tổng quan hệ sinh thái Object Detection, kiến trúc BiFPN kết hợp đa tỉ lệ, và chuỗi kỹ thuật tối ưu điểm số trên Kaggle (Mosaic Augmentation, TTA, WBF).


* **Pipeline Huấn luyện & Tối ưu hóa thực chiến của Kaggle Grandmaster (Alex Shonenkov):**
* **Link Kaggle (Training):** [[Training] EfficientDet - Global Wheat Detection](https://www.kaggle.com/code/shonenkov/training-efficientdet)
* **Link Kaggle (Inference & WBF):** [[Inference] EfficientDet](https://www.kaggle.com/code/shonenkov/inference-efficientdet)
* **Trọng tâm:** Cách thiết lập Albumentations (Cutout, RandomSizedBBoxSafeCrop), lịch trình giảm Learning Rate, và hậu xử lý bằng **Weighted Boxes Fusion (WBF)** thay thế cho NMS truyền thống để gộp kết quả từ nhiều mô hình.



---

## 4. Giai đoạn Hiện đại (SOTA): YOLO thế hệ mới & Detection Transformers (DETR)

Bỏ qua cơ chế Anchor-based phức tạp hoặc NMS thủ công, các mô hình hiện đại hướng tới kiến trúc **Anchor-free**, **End-to-End (NMS-free)** và ứng dụng **Vision Transformer**:

* **DETR (DEtection TRansformer - Facebook AI):**
* **Link Kaggle (Hướng dẫn chi tiết + K-Fold + WBF):** [End to End Object Detection with Transformers: DETR](https://www.kaggle.com/code/tanulsingh077/end-to-end-object-detection-with-transformers-detr)
* **Link Kaggle (Ứng dụng trên tập dữ liệu Great Barrier Reef):** [DETR - Detection Transformer - Train](https://www.kaggle.com/code/julian3833/detr-detection-transformer-train-0-189)
* **Trọng tâm:** Kết hợp Backbone CNN (như ResNet) với bộ Encoder-Decoder của Transformer và hàm mất mát Bipartite Matching Loss (Hungarian Algorithm) để dự đoán trực tiếp tập hợp các Bounding Box mà không cần NMS.




* **YOLOv10 (Kiến trúc Real-time End-to-End không cần NMS):**
* **Link Kaggle:** [Vehicle Detection using YOLOv10](https://www.kaggle.com/code/universalus/vehicle-detection-using-yolov10)
* **Trọng tâm:** Huấn luyện mô hình phát hiện phương tiện giao thông với cơ chế Dual Label Assignments giúp loại bỏ bước NMS, tối ưu tốc độ suy luận (Inference latency).


* **YOLO11 (Thế hệ Ultralytics mới, đa tác vụ & tối ưu phần cứng):**
* **Link Kaggle:** [YOLO11 - Object Detection, Segmentation & Pose](https://www.kaggle.com/code/keshabkkumar/yolo11)
* **Trọng tâm:** Quy trình chuẩn bị `data.yaml`, huấn luyện và đánh giá mô hình YOLO hiện đại với tốc độ cao và ít tham số hơn các bản tiền nhiệm.



---

## Bảng tóm tắt lộ trình thực hành & Kỹ thuật tối ưu trên Kaggle

| Cấp độ | Đại diện tiêu biểu | Kỹ thuật Tối ưu hóa cần tập trung trên Kaggle |
| --- | --- | --- |
| **1. Cổ điển (Classical CV)** | Haar Cascade, HOG + SVM, Background Subtraction

 | Cân bằng Histogram, lọc nhiễu (Smoothing), tìm ngưỡng (Otsu Thresholding), xử lý hình thái học (Morphology).

 |
| **2. Two-Stage (R-CNN)** | Faster R-CNN (Backbone VGG-16, ResNet)

 | Transfer Learning, đóng băng tầng (Freeze layers), tinh chỉnh tỷ lệ Anchor Boxes, Feature Pyramid Network (FPN).

 |
| **3. One-Stage thực chiến** | EfficientDet, YOLOv5/v8 | Tăng cường dữ liệu nâng cao (**Mosaic, MixUp, CutMix** qua Albumentations), **TTA** (Test-Time Augmentation), gộp hộp bằng **WBF** (Weighted Boxes Fusion). |
| **4. Hiện đại (SOTA)** | DETR, YOLOv10, YOLO11 | Bipartite Matching, **NMS-free** training, lượng tử hóa mô hình (FP16 / INT8) và xuất sang **ONNX / TensorRT** để tăng tốc độ chạy. |

Bạn đang dự định áp dụng chuỗi mô hình này cho đồ án môn học hay một bài toán cụ thể nào (ví dụ: nhận diện phương tiện giao thông, y tế, hay lỗi sản phẩm)?