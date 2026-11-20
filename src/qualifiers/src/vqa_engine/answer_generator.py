from .question_parser import extract_target_class, CLASS_ID_TO_NAME
from .spatial_reasoner import determine_spatial_zone

def generate_answer(intent, question_text, detections):
    """
    Synthesizes standardized answer JSON according to Competition Section VI.3:
      - COUNT: Integer, e.g.: '2'; if none '0'
      - EXIST: 'Co' or 'Khong'
      - LIST: '2 bottle, 1 cup'; if none 'Khong co'
      - LOCATION: 'ben trai', 'o giua' or 'ben phai'
      - OUT_OF_SCOPE: 'OOS'
    detections: list of dicts with keys {'class_id': int, 'score': float, 'bbox': [x1, y1, x2, y2]}
    """
    intent = intent.upper().strip()

    if intent == "OUT_OF_SCOPE":
        return "OOS"

    target_cid = extract_target_class(question_text)

    # Filter detections for target class if specified
    if target_cid is not None:
        target_boxes = [d for d in detections if d["class_id"] == target_cid]
    else:
        target_boxes = detections

    if intent == "COUNT":
        count = len(target_boxes)
        return str(count)

    elif intent == "EXIST":
        return "Co" if len(target_boxes) > 0 else "Khong"

    elif intent == "LOCATION":
        if len(target_boxes) == 0:
            return "Khong co"
        # Take the detection with highest confidence score
        best_det = max(target_boxes, key=lambda x: x.get("score", 0.0))
        zone = determine_spatial_zone(best_det["bbox"])
        return zone

    elif intent == "LIST":
        if len(detections) == 0:
            return "Khong co"
        # Count frequency of each detected class
        counts = {}
        for d in detections:
            cid = d["class_id"]
            cname = CLASS_ID_TO_NAME.get(cid, f"class_{cid}")
            counts[cname] = counts.get(cname, 0) + 1

        parts = [f"{cnt} {cname}" for cname, cnt in sorted(counts.items())]
        return ", ".join(parts)

    else:
        return "OOS"
