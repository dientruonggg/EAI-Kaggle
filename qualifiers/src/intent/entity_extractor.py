#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - OBJECT CLASS ENTITY EXTRACTOR
======================================================
BTC Permitted Rule:
  "Được sử dụng từ điển và logic để chuẩn hóa câu hỏi, trích xuất lớp vật
   thể và định dạng đáp án; không được thay thế hoàn toàn TextCNN bằng
   luật từ khóa"

This module extracts the target object class (0..6) from a Vietnamese question:
  0: bottle   (chai, bình nước, lọ, chai nhựa...)
  1: cup      (cốc, ly, tách, ca nước...)
  2: phone    (điện thoại, smartphone, đt...)
  3: book     (sách, cuốn sách, quyển sách, tập vở...)
  4: mouse    (chuột, con chuột, chuột máy tính...)
  5: box      (hộp, cái hộp, thùng, hộp giấy, thùng carton...)
  6: toy_car  (ô tô đồ chơi, xe đồ chơi, xe con, ô tô...)
  -1: None    (for LIST and OUT_OF_SCOPE questions)
"""

import os
import re
import csv
import unicodedata

# 7 Competition Object Classes (Fixed ID Order)
CLASS_ID_TO_NAME = {
    0: "bottle",
    1: "cup",
    2: "phone",
    3: "book",
    4: "mouse",
    5: "box",
    6: "toy_car"
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}

# Mapping from class_id to all canonical aliases / synonyms (accented & unaccented)
CLASS_DICTIONARY = {
    0: [
        "chai", "binh nuoc", "chai nuoc", "binh", "lo", "chai nhua", "chai thuy tinh",
        "bình nước", "chai nước", "bình", "lọ", "chai nhựa", "chai thuỷ tinh", "bottle"
    ],
    1: [
        "coc", "ly", "tach", "ca nuoc", "coc uong nuoc", "ly nuoc",
        "cốc", "tách", "ca nước", "cốc uống nước", "ly nước", "cup"
    ],
    2: [
        "dien thoai", "smartphone", "dien thoai thong minh", "dien thoai di dong", "dt", "phone",
        "điện thoại", "điện thoại thông minh", "điện thoại di động", "đt"
    ],
    3: [
        "sach", "cuon sach", "quyen sach", "tap vo", "tap sach", "tap", "giao trinh", "tai lieu", "book",
        "sách", "cuốn sách", "quyển sách", "tập vở", "tập sách", "tập", "giáo trình", "tài liệu"
    ],
    4: [
        "chuot", "chuot may tinh", "con chuot", "chuot khong day", "chuot quang", "mouse",
        "chuột", "chuột máy tính", "con chuột", "chuột không dây", "chuột quang"
    ],
    5: [
        "hop", "cai hop", "thung", "hop giay", "thung giay", "hop carton", "hop qua", "box",
        "hộp", "cái hộp", "thùng", "hộp giấy", "thùng giấy", "hộp carton", "hộp quà"
    ],
    6: [
        "oto do choi", "xe do choi", "xe con do choi", "xe hoi do choi", "xe mo hinh", "oto", "xe con", "xe hoi", "toy_car",
        "ô tô đồ chơi", "xe đồ chơi", "xe con đồ chơi", "xe hơi đồ chơi", "xe mô hình", "ô tô", "xe con", "xe hơi"
    ]
}


def remove_accents(text):
    """Strips diacritics / tones from Vietnamese text."""
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)


def normalize_text(text):
    """Normalizes text for robust dictionary phrase lookup."""
    text = remove_accents(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


# Precompile pattern matchers sorted by token length descending (longest phrase first)
_COMPILED_PATTERNS = []
for cid, aliases in CLASS_DICTIONARY.items():
    for alias in aliases:
        norm_alias = normalize_text(alias)
        word_count = len(norm_alias.split())
        char_count = len(norm_alias)
        pattern = re.compile(r"\b" + re.escape(norm_alias) + r"\b")
        _COMPILED_PATTERNS.append((word_count, char_count, cid, pattern))

# Sort: multi-word phrases first, then longer string length first
_COMPILED_PATTERNS.sort(key=lambda x: (x[0], x[1]), reverse=True)


def extract_target_class(question_text):
    """
    Extracts the targeted class_id (0..6) from a Vietnamese question.
    Returns:
      int: 0..6 if a target class is mentioned.
      -1: if no class is mentioned (e.g. LIST, OUT_OF_SCOPE questions).
    """
    norm = normalize_text(question_text)
    for _, _, cid, pat in _COMPILED_PATTERNS:
        if pat.search(norm):
            return cid
    return -1


def evaluate_extractor(csv_path):
    """Evaluates target class extraction accuracy against ground-truth QA dataset."""
    total, correct = 0, 0
    mismatches = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total += 1
            true_cid = int(r["target_class"])
            pred_cid = extract_target_class(r["question"])
            if pred_cid == true_cid:
                correct += 1
            else:
                mismatches.append((r["question"], true_cid, pred_cid))

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    return correct, total, acc, mismatches


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    # Go up 4 levels: qualifiers/src/intent/entity_extractor.py -> qualifiers/src/intent -> qualifiers/src -> qualifiers -> repo_root
    intent_dir = os.path.dirname(current_file)
    src_dir = os.path.dirname(intent_dir)
    qual_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(qual_dir, "data")

    print("=" * 70)
    print("   EVALUATING OBJECT CLASS ENTITY EXTRACTOR ON CONTEST QA DATASETS")
    print(f"   Data Directory: {data_dir}")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"qa_{split}.csv")
        if os.path.exists(path):
            c, t, acc, m = evaluate_extractor(path)
            print(f"[{split.upper()}] Accuracy: {c}/{t} ({acc:.2f}%)")
            if m:
                for q, tr, pr in m[:5]:
                    print(f"   Mismatch: '{q}' -> True: {tr}, Pred: {pr}")
    print("=" * 70)
