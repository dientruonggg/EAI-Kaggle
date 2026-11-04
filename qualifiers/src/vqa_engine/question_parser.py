import re
import unicodedata

# Mapping from Vietnamese keywords / aliases to standard class_ids (0..6)
CLASS_KEYWORDS = {
    0: ["chai", "chai nuoc", "binh nuoc", "binh", "bottle"],
    1: ["coc", "ly", "tach", "ca nuoc", "cup"],
    2: ["dien thoai", "dt", "smartphone", "phone"],
    3: ["sach", "cuon sach", "quyen sach", "tap", "book"],
    4: ["chuot", "chuot may tinh", "con chuot", "mouse"],
    5: ["hop", "cai hop", "thung", "hop giay", "box"],
    6: ["oto do choi", "xe do choi", "oto", "xe con", "xe hoi do choi", "toy_car"]
}

CLASS_ID_TO_NAME = {
    0: "bottle",
    1: "cup",
    2: "phone",
    3: "book",
    4: "mouse",
    5: "box",
    6: "toy_car"
}

def remove_vietnamese_tones(text):
    """Removes diacritics / tones from Vietnamese string."""
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = text.replace("đ", "d").replace("Đ", "D")
    return text

def normalize_text(text):
    text = text.lower()
    text = remove_vietnamese_tones(text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def extract_target_class(question_text):
    """
    Extracts the targeted class_id from question using aliases/rules.
    Returns: class_id (int) or None if generic (e.g. LIST questions).
    """
    norm = normalize_text(question_text)
    
    # Check specific multi-word patterns first (e.g. "chuot may tinh", "oto do choi")
    for cid, aliases in sorted(CLASS_KEYWORDS.items(), key=lambda x: -max(len(a) for a in x[1])):
        for alias in sorted(aliases, key=lambda a: -len(a)):
            alias_norm = normalize_text(alias)
            if re.search(r"\b" + re.escape(alias_norm) + r"\b", norm):
                return cid
    return None
