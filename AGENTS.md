# Edge AI Challenge 2026 (Lần 2) - Hardware & Competition Invariants

Source of truth: `Edge AI Challenge 2026 - Training 1.pdf` (Kickoff 08/11/2026). If this file and the PDF disagree, the PDF wins; update this file.

## 0. Timeline
- Vòng loại (Kaggle): 09/11/2026 - 18/11/2026.
- Công bố đội vào chung kết + phát kit: 20/11/2026.
- Chung kết (chạy trên ESP32-S3): 29/11/2026.

## 1. Task Definition
- **Object detection, 7 classes** (ID fixed): 0 `bottle`, 1 `cup`, 2 `phone`, 3 `book`, 4 `mouse`, 5 `box`, 6 `toy_car`. Output: class + bbox + confidence.
- **Intent TextCNN, 5 labels**: `COUNT`, `EXIST`, `LIST`, `LOCATION`, `OUT_OF_SCOPE` (Vietnamese questions).
- Finals logic: intent + object class from question, detection on camera frame -> đếm / tồn tại / danh sách / vị trí -> JSON answer.

## 2. Scoring
- **Vòng bảng score = 70 x mAP50 (detection) + 30 x Accuracy (intent).** Detection is the main lever.
- Local metrics must mirror this: mAP@IoU0.5 for detection, accuracy for intent, on `data/val_split.txt`.

## 3. Hard Constraints (violations = disqualification)
- **Detection parameters: <= 200,000** (hard limit).
- **Quantized model size: <= 1.5 MB** (full INT8).
- **Input image: <= 320 x 240** (camera OV7670 delivers 320x240 RGB565, 4:3). Prefer 4:3 inputs (e.g. 256x192, 320x240) over square to avoid distortion; square sizes (128/160/192) are allowed.
- **No ensemble, no TTA** for detection inference. One model, one forward pass per image. (WBF / multi-model / flip-TTA pipelines from warm-up are banned.)
- **Pretrained weights allowed, but must be declared** (record source + license in the submission notes / README).
- **Do not use hidden test data** for training, pseudo-labeling, or self-training.
- **Finals latency: <= 10 s** end-to-end (capture + preprocess + inference + logic). All inference on-device; no offloading image/question to a server.

## 4. Microcontroller Target (ESP32-S3 N16R8)
- Xtensa dual-core LX7 @ 240 MHz, 512 KB SRAM, 8 MB PSRAM, 16 MB Flash.
- Runtime: TensorFlow Lite Micro via ESP-IDF (organizer repos tested on ESP-IDF 5.2.6). Tensor arena goes in PSRAM.
- Only TFLM-supported builtin ops: Conv2D, DepthwiseConv2D, Add, Concat, Reshape, ReLU/ReLU6, MaxPool/AvgPool, ResizeNearestNeighbor, FullyConnected, Softmax, Logistic, Mean. Avoid custom ops, large attention, dilated conv with non-unit stride.
- Keep decode + NMS outside the model graph (Python for Kaggle, C for device).
- Preprocessing on device must exactly match training (resize method, normalization, INT8 scale/zero-point). Read dtype/scale/zero-point from the `.tflite`, never infer from file name; for TextCNN also verify the token-ID input dtype.

## 5. Quantization Protocol (PTQ vs QAT)
- Full integer INT8: `inference_input_type = tf.int8`, `inference_output_type = tf.int8`, `TFLITE_BUILTINS_INT8`.
- For Keras 3 models use Post-Training Quantization with a representative dataset of 100+ training images.
- Do NOT use `tfmot.quantization.keras.quantize_model` on Keras 3 weights (partial layer matching silently uninitializes up to ~30% of layers; mAP 60% -> 31%).
- Always compare FP32 vs INT8 mAP50 / accuracy on local validation before submitting.

## 6. Submission Rules (Kaggle)
- CSV columns: **`task_type, sample_id, prediction`**.
  - `DETECTION` rows: `prediction` = JSON list of bboxes.
  - `INTENT` rows: `prediction` = one of the 5 labels.
  - Each `sample_id` appears exactly once.
- **Max 4 submissions / 24 h.** Never submit without the user's explicit go-ahead.
- Gate before every upload:
  1. Validate against `sample_submission.csv` (headers, IDs, row count, JSON well-formed, class IDs in 0..6).
  2. Predictions must come from the INT8 `.tflite` that fits Section 3 (log param count + file size alongside the CSV).
  3. Box-count sanity: report total boxes and boxes/image; calibrate the confidence threshold on local validation (the warm-up "2,600-2,800 boxes" rule applied only to the old 60-image test set and is NOT valid here).

## 7. Kaggle T4x2 Remote Workflow
- Large training runs go to Kaggle via `warmup/kaggle_runner/run_remote_train.py` (or `qualifiers/kaggle_runner`), competition slug `edge-ai-challenge-2026-lan-2`.
- Kaggle CLI must be configured for the current machine/account (`~/.kaggle/kaggle.json` or access token); do not hardcode `/home/bim/...` paths or another user's kernel slug.
- Pull weights to `warmup/weights/kaggle_t4/` (or the qualifiers equivalent) and validate locally on `data/val_split.txt` before submitting.
- Project skills live in `.agents/skills/` (`edge-ai-competition`, `kaggle`, `fiftyone-*`); see `.agents/skills/THIRD_PARTY_SKILLS.md`.
