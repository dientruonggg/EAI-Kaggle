#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - AUTOMATED INTENT & QA ANNOTATOR
Generates ground-truth Question-Answer pairs and Intent labels for every image in the dataset,
strictly adhering to Section VI.3 of the Competition Rules:
  - COUNT: "Co bao nhieu chai?" -> e.g. "2" or "0"
  - EXIST: "Co dien thoai hay khong?" -> "Co" or "Khong"
  - LIST: "Trong anh co gi?" -> "2 bottle, 1 cup" or "Khong co"
  - LOCATION: "Cuon sach nam o dau?" -> "ben trai", "o giua", or "ben phai"
  - OUT_OF_SCOPE: "Hom nay troi co mua khong?" -> "OOS"
"""
import os
import sys
import glob
import random
import csv

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
ORIG_W = 320.0

CLASS_NAMES_EN = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]

# Phrasing variations (both accented and unaccented Vietnamese)
CLASS_SYNONYMS_VN = {
    0: ["chai", "chai nuoc", "binh nuoc", "chai"],
    1: ["coc", "ly", "cai coc", "tach nuoc"],
    2: ["dien thoai", "chiec dien thoai", "smartphone", "dien thoai"],
    3: ["cuon sach", "sach", "quyen sach", "tap sach"],
    4: ["chuot", "chuot may tinh", "con chuot", "chuot"],
    5: ["hop", "cai hop", "hop giay", "thung hop"],
    6: ["oto do choi", "xe do choi", "xe oto do choi", "oto do choi"]
}

COUNT_TEMPLATES = [
    "Co bao nhieu {obj} trong hinh?",
    "Co bao nhieu {obj}?",
    "Dem so {obj} trong anh.",
    "Trong anh co may {obj}?",
    "Ban dem duoc bao nhieu {obj} tren ban?",
    "Hay dem so {obj} xuat hien."
]

EXIST_TEMPLATES = [
    "Co {obj} hay khong?",
    "Co {obj} trong hinh khong?",
    "Trong anh co {obj} khong?",
    "Ban co nhin thay {obj} trong anh khong?",
    "Hinh anh co {obj} nao khong?",
    "Trong canh co {obj} khong?"
]

LIST_TEMPLATES = [
    "Trong anh co gi?",
    "Trong hinh co nhung do vat gi?",
    "Liet ke cac vat the co trong anh.",
    "Ke ten cac do vat xuat hien tren ban.",
    "Co nhung loai do vat nao trong hinh nay?",
    "Hay liet ke toan bo cac vat the trong hinh."
]

LOCATION_TEMPLATES = [
    "{obj} nam o dau?",
    "Vi tri cua {obj} la o dau?",
    "{obj} nam o ben trai, o giua hay ben phai?",
    "{obj} xuat hien o phia nao khung hinh?",
    "Hay xac dinh {obj} nam ben trai, o giua hay ben phai anh."
]

OOS_QUESTIONS = [
    "Hom nay troi co mua khong?",
    "Thoi tiet hom nay the nao?",
    "May gio roi?",
    "Ban ten la gi?",
    "Nhiet do phong bay gio la bao nhieu?",
    "Thu do cua Viet Nam la gi?",
    "Chieu nay an gi?",
    "Ai la nguoi tao ra ban?",
    "Phuong trinh bac hai giai nhu the nao?",
    "Hom nay la ngay may?",
    "Troi dang nang hay mua?",
    "Ban co biet hat khong?",
    "Ti gia do la hom nay bao nhieu?",
    "Doc tin tuc hom nay di.",
    "May tinh cua ban chay he dieu hanh gi?",
    "Gia xang hom nay tang hay giam?",
    "Ban co thich xem phim khong?",
    "Chi duong den san bay Da Nang giup toi.",
    "Cong thuc tinh chu vi hinh tron la gi?",
    "Hom nay la thu may?"
]

def determine_zone(cx_normalized):
    cx_px = cx_normalized * ORIG_W
    if cx_px < (ORIG_W / 3.0):
        return "ben trai"
    elif cx_px <= (2.0 * ORIG_W / 3.0):
        return "o giua"
    else:
        return "ben phai"

def generate_qa_for_image(img_id, label_path):
    """
    Parses bounding boxes for an image and generates QA pairs across all 5 intents.
    """
    boxes = []
    class_counts = {c: 0 for c in range(7)}
    class_boxes = {c: [] for c in range(7)}

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    class_counts[cid] += 1
                    class_boxes[cid].append((cx, cy, w, h))

    qa_list = []

    # 1. COUNT QA (1 positive count + 1 zero count)
    present_classes = [c for c in range(7) if class_counts[c] > 0]
    absent_classes = [c for c in range(7) if class_counts[c] == 0]

    if present_classes:
        pos_cid = random.choice(present_classes)
        obj_name = random.choice(CLASS_SYNONYMS_VN[pos_cid])
        q_text = random.choice(COUNT_TEMPLATES).format(obj=obj_name)
        ans = str(class_counts[pos_cid])
        qa_list.append({"intent": "COUNT", "question": q_text, "answer": ans, "target_class": pos_cid})

    if absent_classes:
        zero_cid = random.choice(absent_classes)
        obj_name = random.choice(CLASS_SYNONYMS_VN[zero_cid])
        q_text = random.choice(COUNT_TEMPLATES).format(obj=obj_name)
        qa_list.append({"intent": "COUNT", "question": q_text, "answer": "0", "target_class": zero_cid})

    # 2. EXIST QA (1 positive + 1 negative)
    if present_classes:
        pos_cid = random.choice(present_classes)
        obj_name = random.choice(CLASS_SYNONYMS_VN[pos_cid])
        q_text = random.choice(EXIST_TEMPLATES).format(obj=obj_name)
        qa_list.append({"intent": "EXIST", "question": q_text, "answer": "Co", "target_class": pos_cid})

    if absent_classes:
        zero_cid = random.choice(absent_classes)
        obj_name = random.choice(CLASS_SYNONYMS_VN[zero_cid])
        q_text = random.choice(EXIST_TEMPLATES).format(obj=obj_name)
        qa_list.append({"intent": "EXIST", "question": q_text, "answer": "Khong", "target_class": zero_cid})

    # 3. LIST QA
    q_list = random.choice(LIST_TEMPLATES)
    if not present_classes:
        list_ans = "Khong co"
    else:
        parts = [f"{class_counts[c]} {CLASS_NAMES_EN[c]}" for c in sorted(present_classes)]
        list_ans = ", ".join(parts)
    qa_list.append({"intent": "LIST", "question": q_list, "answer": list_ans, "target_class": None})

    # 4. LOCATION QA
    if present_classes:
        loc_cid = random.choice(present_classes)
        obj_name = random.choice(CLASS_SYNONYMS_VN[loc_cid])
        # Pick the most prominent box (largest)
        best_box = max(class_boxes[loc_cid], key=lambda b: b[2] * b[3])
        zone = determine_zone(best_box[0])
        q_loc = random.choice(LOCATION_TEMPLATES).format(obj=obj_name)
        qa_list.append({"intent": "LOCATION", "question": q_loc, "answer": zone, "target_class": loc_cid})
    elif absent_classes:
        zero_cid = random.choice(absent_classes)
        obj_name = random.choice(CLASS_SYNONYMS_VN[zero_cid])
        q_loc = random.choice(LOCATION_TEMPLATES).format(obj=obj_name)
        qa_list.append({"intent": "LOCATION", "question": q_loc, "answer": "Khong co", "target_class": zero_cid})

    # 5. OUT_OF_SCOPE QA (Attach an OOS question with probability 0.5)
    if random.random() < 0.6:
        oos_q = random.choice(OOS_QUESTIONS)
        qa_list.append({"intent": "OUT_OF_SCOPE", "question": oos_q, "answer": "OOS", "target_class": None})

    return qa_list

def generate_dataset_qa(split_names=("train", "val", "test")):
    print("=" * 70)
    print("   AUTOMATED QA ANNOTATION PIPELINE FOR QUALIFIERS")
    print("=" * 70)

    q_counter = 1

    for split in split_names:
        img_dir = os.path.join(PROJECT_ROOT, "data", split, "images")
        lbl_dir = os.path.join(PROJECT_ROOT, "data", split, "labels")

        imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                      glob.glob(os.path.join(img_dir, "*.jpg")))

        print(f"[*] Processing split '{split}': {len(imgs)} images...")

        all_qa_rows = []
        intent_rows = []

        for img_p in imgs:
            img_id = os.path.splitext(os.path.basename(img_p))[0]
            lbl_p = os.path.join(lbl_dir, f"{img_id}.txt")

            qas = generate_qa_for_image(img_id, lbl_p)
            for qa in qas:
                qid = f"q_{split}_{q_counter:05d}"
                q_counter += 1

                all_qa_rows.append({
                    "question_id": qid,
                    "image_id": img_id,
                    "question": qa["question"],
                    "intent": qa["intent"],
                    "target_class": qa["target_class"] if qa["target_class"] is not None else -1,
                    "answer": qa["answer"]
                })

                intent_rows.append({
                    "question_id": qid,
                    "question": qa["question"],
                    "intent": qa["intent"]
                })

        # Save comprehensive QA table
        qa_csv = os.path.join(PROJECT_ROOT, "data", f"qa_{split}.csv")
        with open(qa_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question_id", "image_id", "question", "intent", "target_class", "answer"])
            writer.writeheader()
            writer.writerows(all_qa_rows)

        # Save official intent format
        intent_csv = os.path.join(PROJECT_ROOT, "data", f"intent_{split}.csv")
        with open(intent_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question_id", "question", "intent"])
            writer.writeheader()
            writer.writerows(intent_rows)

        print(f"  [✔] Generated {len(all_qa_rows)} QA pairs for '{split}' -> {qa_csv}")

    print("\n[✔] ALL SPLITS ANNOTATED SUCCESSFULLY!")

if __name__ == "__main__":
    generate_dataset_qa()
