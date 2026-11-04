# Feature Specification: TextCNN Vietnamese Intent Classifier (100% Accuracy)

**Feature Branch**: `002-textcnn-vietnamese-100`

**Created**: 2026-11-04

**Status**: Ready for Implementation (Fast-Track)

**Input**: User description: "Tìm bản TextCNN train trên bộ dữ liệu tiếng Việt, đảm bảo khả năng hiểu 100% cho mọi trường hợp"

---

## 1. Executive Summary

This feature delivers an ultra-robust, high-accuracy TextCNN intent classification engine for the Edge AI Challenge 2026. The classifier must correctly understand Vietnamese queries across all 5 competition intents (`COUNT`, `EXIST`, `LIST`, `LOCATION`, `OUT_OF_SCOPE`) with:
1. **100% Validation & Test Accuracy** on official benchmark splits (`intent_val.csv`, `intent_test.csv`).
2. **Zero-Degradation on Accented / Unaccented Text**: Seamlessly understands formal Vietnamese with diacritics, unaccented typing, and colloquial chat variants ("ko", "k", "dc", "nhé", "vậy").
3. **High Discriminative Power on Hard Negatives**: Correctly distinguishes subtle semantic boundaries:
   - "Ở đây có chuột máy tính ko?" $\rightarrow$ `EXIST` (not `LOCATION`).
   - "Vị trí của con chuột" $\rightarrow$ `LOCATION` (not `EXIST`).
   - "1 cộng 1 bằng mấy?" $\rightarrow$ `OUT_OF_SCOPE` (not `COUNT`).
4. **Hardware-Ready INT8 TFLite Micro Model**: Model size $\le 1.5$ MB (budget $< 60$ KB), $< 20,000$ parameters, Full INT8 quantization for ESP32-S3 deployment.

---

## 2. User Scenarios & Acceptance Criteria

### User Story 1 - 100% Accurate Intent Recognition Across All Grammatical Forms (Priority: P1)

As a competition system evaluating live user queries, when an arbitrary Vietnamese question is received, the TextCNN model classifies the query into the exact target intent without failing on diacritics or sentence structures.

**Acceptance Scenarios**:

1. **Given** any counting query with or without accents ("Có bao nhiêu chai?", "Co may cuon sach?", "Dem so luong coc tren ban giup toi"), **When** classified, **Then** output is `COUNT`.
2. **Given** any existence check ("Trong hình có cốc không?", "O day co chuot khong ban?"), **When** classified, **Then** output is `EXIST`.
3. **Given** any listing query ("Trong ảnh có những gì?", "Liet ke cac do vat", "Ke ten cac mon do tren ban"), **When** classified, **Then** output is `LIST`.
4. **Given** any spatial location query ("Sách nằm ở đâu?", "Cuon sach o ben trai hay ben phai?", "Vi tri cua chiec coc la gi?"), **When** classified, **Then** output is `LOCATION`.
5. **Given** any irrelevant, conversational, mathematical, or general knowledge query ("Hom nay troi mua khong?", "1 + 1 = may?", "Ban ten la gi?"), **When** classified, **Then** output is `OUT_OF_SCOPE`.

---

### User Story 2 - Full Integer INT8 TFLite Micro Quantization (Priority: P1)

As an embedded engineer, I need the trained TextCNN weights exported directly to an INT8 TFLite file compatible with TensorFlow Lite Micro on ESP32-S3.

**Acceptance Scenarios**:

1. **Given** the trained model, **When** quantized with representative calibration samples, **Then** file size is $\le 1.5$ MB (strictly $< 100$ KB).
2. **Given** evaluation of the exported INT8 `.tflite` model, **When** tested on the evaluation set, **Then** accuracy matches the floating-point model with zero degradation.
3. **Given** inference on ESP32-S3, **When** executed, **Then** latency is $< 10$ milliseconds.
