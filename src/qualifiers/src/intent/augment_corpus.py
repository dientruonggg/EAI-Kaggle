#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - VIETNAMESE INTENT CORPUS & DICTIONARY GENERATOR
========================================================================
BTC Permitted Rule:
  "Được sử dụng từ điển và logic để chuẩn hóa câu hỏi, trích xuất lớp vật
   thể và định dạng đáp án; không được thay thế hoàn toàn TextCNN bằng
   luật từ khóa"

This module provides:
  1. Comprehensive dictionary of 7 competition object classes and their
     Vietnamese synonyms/aliases (accented, unaccented, colloquial).
  2. Canonical question normalization logic (diacritics stripping, polite
     social fillers filtering, punctuation cleaning).
  3. Template-based augmented corpus generator for high-diversity training.
"""

import os
import re
import csv
import random
import unicodedata

random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

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

# Comprehensive Class Synonyms & Aliases (Unaccented)
CLASSES_UNACCENTED = [
    ("bottle", ["chai", "binh nuoc", "chai nuoc", "binh", "lo", "chai nhua", "chai thuy tinh", "lo nuoc", "bottle"]),
    ("cup", ["coc", "ly", "tach", "coc uong nuoc", "ly nuoc", "ca nuoc", "ly uong nuoc", "tach uong nuoc", "cup"]),
    ("phone", ["dien thoai", "smartphone", "dien thoai thong minh", "dien thoai di dong", "dt", "phone"]),
    ("book", ["sach", "cuon sach", "quyen sach", "tap vo", "tap sach", "giao trinh", "tai lieu", "vo", "book"]),
    ("mouse", ["chuot", "chuot may tinh", "con chuot", "chuot khong day", "chuot quang", "chuot day", "mouse"]),
    ("box", ["hop", "cai hop", "thung", "hop giay", "thung giay", "hop carton", "hop qua", "box"]),
    ("toy_car", ["oto do choi", "xe do choi", "xe con do choi", "xe hoi do choi", "xe mo hinh", "oto", "xe con", "xe hoi", "toy_car"])
]

# Comprehensive Class Synonyms & Aliases (Accented)
CLASSES_ACCENTED = [
    ("bottle", ["chai", "bình nước", "chai nước", "bình", "lọ", "chai nhựa", "chai thuỷ tinh", "lọ nước", "bottle"]),
    ("cup", ["cốc", "ly", "tách", "cốc uống nước", "ly nước", "ca nước", "ly uống nước", "tách uống nước", "cup"]),
    ("phone", ["điện thoại", "smartphone", "điện thoại thông minh", "điện thoại di động", "đt", "phone"]),
    ("book", ["sách", "cuốn sách", "quyển sách", "tập vở", "tập sách", "giáo trình", "tài liệu", "vở", "book"]),
    ("mouse", ["chuột", "chuột máy tính", "con chuột", "chuột không dây", "chuột quang", "chuột dây", "mouse"]),
    ("box", ["hộp", "cái hộp", "thùng", "hộp giấy", "thùng giấy", "hộp carton", "hộp quà", "box"]),
    ("toy_car", ["ô tô đồ chơi", "xe đồ chơi", "xe con đồ chơi", "xe hơi đồ chơi", "xe mô hình", "ô tô", "xe con", "xe hơi", "toy_car"])
]

# BTC-Approved Polite Social Filler Particles (Stopwords)
STOPWORDS = {
    'ban', 'oi', 'cho', 'toi', 'hoi', 'xin', 'minh',
    'vay', 'nhi', 'the', 'a', 'gium', 'giup', 'nhe', 'nha'
}

# 5 Intent Classes
INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
INTENT_MAP = {idx: name for idx, name in enumerate(INTENTS)}
LABEL_MAP = {name: idx for idx, name in enumerate(INTENTS)}

# Question Templates (Unaccented)
TEMPLATES_UNACCENTED = {
    "COUNT": [
        "Co bao nhieu {item} trong hinh?",
        "Co bao nhieu {item} trong anh?",
        "Co bao nhieu {item} tren ban?",
        "Trong anh co may {item}?",
        "Trong hinh co may {item}?",
        "Trong anh co bao nhieu {item} vay?",
        "Dem so luong {item} trong anh giup toi.",
        "Dem so {item} tren ban gium toi.",
        "Ban dem duoc bao nhieu {item} tat ca?",
        "Tong cong co bao nhieu {item}?",
        "Co may {item} xuat hien tren ban?",
        "Hay dem xem co bao nhieu {item}.",
        "Co tat ca may {item} trong hinh?",
        "Dem so {item} dang co tren ban.",
        "So luong {item} trong anh la bao nhieu?",
        "Ban dem giup toi so {item} tren ban.",
        "Trong hinh co tong cong bao nhieu {item}?"
    ],
    "EXIST": [
        "Co {item} trong hinh khong?",
        "Co {item} trong anh hay khong?",
        "Trong anh co {item} nao khong?",
        "Trong hinh co {item} khong ban?",
        "O day co {item} khong?",
        "Ban co thay {item} nao tren ban khong?",
        "Co {item} nao xuat hien tren ban khong?",
        "Hinh nay co {item} khong?",
        "Lieu co {item} trong anh hay khong?",
        "Kiem tra xem co {item} khong.",
        "Tren ban co {item} khong vay?",
        "O trong anh co thay {item} nao ko?",
        "Ban co nhin thay {item} trong anh khong?",
        "Trong canh co {item} khong?",
        "Co xuat hien {item} khong?"
    ],
    "LIST": [
        "Trong anh co nhung gi?",
        "Trong hinh co nhung vat the nao?",
        "Hay liet ke toan bo cac vat the trong hinh.",
        "Liet ke cac do vat tren ban giup toi.",
        "Ke ten cac do vat xuat hien trong anh.",
        "Tren ban dang co nhung thu gi?",
        "Cho toi danh sach cac vat the trong anh.",
        "Trong anh gom co nhung do vat gi?",
        "Liet ke tat ca cac mon do dang co.",
        "Ke ten tat ca vat the tren ban.",
        "Nhung vat gi dang co mat trong hinh anh?",
        "Danh sach cac do vat trong hinh la gi?",
        "Liet ke cac vat the co trong anh.",
        "Co nhung loai do vat nao trong hinh nay?"
    ],
    "LOCATION": [
        "{item} nam o dau?",
        "{item} o dau vay?",
        "Vi tri cua {item} la o dau?",
        "{item} dang nam o vi tri nao tren ban?",
        "Xac dinh vi tri cua {item} giup toi.",
        "{item} o ben trai hay ben phai?",
        "Tim vi tri cua {item} trong anh.",
        "{item} dang o ben nao cua hinh anh?",
        "Hay cho biet {item} nam o ben trai, o giua hay ben phai.",
        "Vi tri cu the cua {item} tren ban.",
        "{item} nam o dau tren ban the?",
        "Hay xac dinh {item} nam ben trai, o giua hay ben phai anh.",
        "{item} xuat hien o phia nao khung hinh?"
    ],
    "OUT_OF_SCOPE": [
        "Hom nay troi co mua khong?",
        "Thoi tiet hom nay the nao?",
        "Nhiet do hien tai la bao nhieu do?",
        "Ban ten la gi?",
        "Ban co biet hat hay nhay khong?",
        "Ban co biet hat khong?",
        "May gio roi ban?",
        "1 cong 1 bang may?",
        "Hai nhan nam bang bao nhieu?",
        "Gia xang hom nay tang hay giam?",
        "Gia vang hien tai bao nhieu mot chi?",
        "Thu do cua nuoc Phap la gi?",
        "Viet Nam co bao nhieu tinh thanh?",
        "Ai la tong thong dau tien cua nuoc My?",
        "Ke cho toi nghe mot cau chuyen cuoi di.",
        "Ban bao nhieu tuoi roi?",
        "Lam the nao de hoc gioi toan?",
        "Nuoc soi o bao nhieu do C?",
        "Mat troi moc o huong nao?",
        "Hom nay la thu may trong tuan?",
        "Ban an com chua?",
        "Doc tin tuc hom nay di.",
        "Chao ban, ban khoe khong?",
        "Xin chao, chuc ban mot ngay tot lanh.",
        "Tam biet nhe, hen gap lai sau.",
        "Cam on ban rat nhieu.",
        "Toi dang cam thay rat buon."
    ]
}

# Question Templates (Accented)
TEMPLATES_ACCENTED = {
    "COUNT": [
        "Có bao nhiêu {item} trong hình?",
        "Có bao nhiêu {item} trong ảnh?",
        "Có bao nhiêu {item} trên bàn?",
        "Trong ảnh có mấy {item}?",
        "Trong hình có mấy {item}?",
        "Trong ảnh có bao nhiêu {item} vậy?",
        "Đếm số lượng {item} trong ảnh giúp tôi.",
        "Đếm số {item} trên bàn giùm tôi.",
        "Bạn đếm được bao nhiêu {item} tất cả?",
        "Tổng cộng có bao nhiêu {item}?",
        "Có mấy {item} xuất hiện trên bàn?",
        "Hãy đếm xem có bao nhiêu {item}.",
        "Có tất cả mấy {item} trong hình?",
        "Đếm số {item} đang có trên bàn.",
        "Số lượng {item} trong ảnh là bao nhiêu?",
        "Hãy đếm số {item} xuất hiện.",
        "Bạn đếm giúp tôi số {item} trên bàn."
    ],
    "EXIST": [
        "Có {item} trong hình không?",
        "Có {item} trong ảnh hay không?",
        "Trong ảnh có {item} nào không?",
        "Trong hình có {item} không bạn?",
        "Ở đây có {item} không?",
        "Bạn có thấy {item} nào trên bàn không?",
        "Có {item} nào xuất hiện trên bàn không?",
        "Hình này có {item} không?",
        "Liệu có {item} trong ảnh hay không?",
        "Kiểm tra xem có {item} không.",
        "Trên bàn có {item} không vậy?",
        "Ở trong ảnh có thấy {item} nào ko?",
        "Bạn có nhìn thấy {item} trong ảnh không?",
        "Trong cảnh có {item} không?",
        "Có xuất hiện {item} hay không?"
    ],
    "LIST": [
        "Trong ảnh có những gì?",
        "Trong hình có những vật thể nào?",
        "Hãy liệt kê toàn bộ các vật thể trong hình.",
        "Liệt kê các đồ vật trên bàn giúp tôi.",
        "Kể tên các đồ vật xuất hiện trong ảnh.",
        "Trên bàn đang có những thứ gì?",
        "Cho tôi danh sách các vật thể trong ảnh.",
        "Trong ảnh gồm có những đồ vật gì?",
        "Liệt kê tất cả các món đồ đang có.",
        "Kể tên tất cả vật thể trên bàn.",
        "Những vật gì đang có mặt trong hình ảnh?",
        "Danh sách các đồ vật trong hình là gì?",
        "Liệt kê các vật thể có trong ảnh.",
        "Có những loại đồ vật nào trong hình này?"
    ],
    "LOCATION": [
        "{item} nằm ở đâu?",
        "{item} ở đâu vậy?",
        "Vị trí của {item} là ở đâu?",
        "{item} đang nằm ở vị trí nào trên bàn?",
        "Xác định vị trí của {item} giúp tôi.",
        "{item} ở bên trái hay bên phải?",
        "Tìm vị trí của {item} trong ảnh.",
        "{item} đang ở bên nào của hình ảnh?",
        "Hãy cho biết {item} nằm ở bên trái, ở giữa hay bên phải.",
        "Vị trí cụ thể của {item} trên bàn.",
        "{item} nằm ở đâu trên bàn thế?",
        "Hãy xác định {item} nằm bên trái, ở giữa hay bên phải ảnh.",
        "{item} xuất hiện ở phía nào khung hình?"
    ],
    "OUT_OF_SCOPE": [
        "Hôm nay trời có mưa không?",
        "Thời tiết hôm nay thế nào?",
        "Nhiệt độ hiện tại là bao nhiêu độ?",
        "Bạn tên là gì?",
        "Bạn có biết hát hay nhảy không?",
        "Bạn có biết hát không?",
        "Mấy giờ rồi bạn?",
        "1 cộng 1 bằng mấy?",
        "Hai nhân năm bằng bao nhiêu?",
        "Giá xăng hôm nay tăng hay giảm?",
        "Giá vàng hiện tại bao nhiêu một chỉ?",
        "Thủ đô của nước Pháp là gì?",
        "Việt Nam có bao nhiêu tỉnh thành?",
        "Ai là tổng thống đầu tiên của nước Mỹ?",
        "Kể cho tôi nghe một câu chuyện cười đi.",
        "Bạn bao nhiêu tuổi rồi?",
        "Làm thế nào để học giỏi toán?",
        "Nước sôi ở bao nhiêu độ C?",
        "Mặt trời mọc ở hướng nào?",
        "Hôm nay là thứ mấy trong tuần?",
        "Bạn ăn cơm chưa?",
        "Đọc tin tức hôm nay đi.",
        "Chào bạn, bạn khỏe không?",
        "Xin chào, chúc bạn một ngày tốt lành.",
        "Tạm biệt nhé, hẹn gặp lại sau.",
        "Cảm ơn bạn rất nhiều.",
        "Tôi đang cảm thấy rất buồn."
    ]
}


def remove_accents(text):
    """Normalizes string and strips all Vietnamese diacritics / tones."""
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)


def normalize_question(text):
    """
    Standardizes question according to BTC normalization protocol:
    1. Strip diacritics / accents.
    2. Lowercase and strip punctuation.
    3. Filter polite social fillers (STOPWORDS).
    4. Canonicalize whitespace.
    """
    text = remove_accents(text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [w for w in text.split() if w and w not in STOPWORDS]
    return " ".join(tokens)


def generate_augmented_corpus(base_train_csv, output_augmented_csv):
    """Generates a high-diversity, balanced augmented corpus for TextCNN training."""
    rows = []
    if os.path.exists(base_train_csv):
        with open(base_train_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({"question": r["question"], "intent": r["intent"]})
    print(f"[*] Base training samples: {len(rows)}")

    generated = []

    # 1. Class-dependent intents (COUNT, EXIST, LOCATION)
    for intent in ["COUNT", "EXIST", "LOCATION"]:
        # Unaccented templates
        for tpl in TEMPLATES_UNACCENTED[intent]:
            for _, syns in CLASSES_UNACCENTED:
                for syn in syns:
                    q = tpl.format(item=syn)
                    generated.append({"question": q, "intent": intent})
        # Accented templates
        for tpl in TEMPLATES_ACCENTED[intent]:
            for _, syns in CLASSES_ACCENTED:
                for syn in syns:
                    q = tpl.format(item=syn)
                    generated.append({"question": q, "intent": intent})

    # 2. LIST (General query, no specific target class)
    for _ in range(15):
        for tpl in TEMPLATES_UNACCENTED["LIST"]:
            generated.append({"question": tpl, "intent": "LIST"})
        for tpl in TEMPLATES_ACCENTED["LIST"]:
            generated.append({"question": tpl, "intent": "LIST"})

    # 3. OUT_OF_SCOPE
    for _ in range(10):
        for tpl in TEMPLATES_UNACCENTED["OUT_OF_SCOPE"]:
            generated.append({"question": tpl, "intent": "OUT_OF_SCOPE"})
        for tpl in TEMPLATES_ACCENTED["OUT_OF_SCOPE"]:
            generated.append({"question": tpl, "intent": "OUT_OF_SCOPE"})

    print(f"[*] Generated synthetic variations: {len(generated)}")

    all_rows = rows + generated
    random.shuffle(all_rows)

    os.makedirs(os.path.dirname(output_augmented_csv), exist_ok=True)
    with open(output_augmented_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question", "intent"])
        writer.writeheader()
        for i, r in enumerate(all_rows):
            writer.writerow({
                "question_id": f"q_aug_{i+1:05d}",
                "question": r["question"],
                "intent": r["intent"]
            })

    print(f"[✔] Total augmented training samples: {len(all_rows)} written to {output_augmented_csv}")
    return output_augmented_csv


if __name__ == "__main__":
    out_path = os.path.join(PROJECT_ROOT, "qualifiers/data/intent_train_augmented.csv")
    in_path = os.path.join(PROJECT_ROOT, "qualifiers/data/intent_train.csv")
    generate_augmented_corpus(in_path, out_path)
