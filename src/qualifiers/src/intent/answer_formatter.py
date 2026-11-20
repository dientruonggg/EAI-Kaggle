#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - STANDARDIZED ANSWER FORMATTER & SPATIAL REASONER
========================================================================
BTC Permitted Rule:
  "Được sử dụng từ điển và logic để chuẩn hóa câu hỏi, trích xuất lớp vật
   thể và định dạng đáp án; không được thay thế hoàn toàn TextCNN bằng
   luật từ khóa"

This module formats the final competition answer JSON according to the
official rules (Kickoff & Finals Specification Section VI.3):
  1. COUNT:
     - Returns integer count of target class: "4", "1", "0".
  2. EXIST:
     - Returns "Co" if target class appears in image, else "Khong".
  3. LIST:
     - Returns comma-separated list of detected classes with their counts:
       e.g. "4 book, 1 mouse", "1 bottle".
       If no objects are detected, returns "Khong co".
  4. LOCATION:
     - Divides camera frame horizontally into 3 zones (320px width):
       * Zone 1 (Left):   cx < 320 / 3 (~106.67px)       -> "ben trai"
       * Zone 2 (Middle): 106.67px <= cx <= 213.33px     -> "o giua"
       * Zone 3 (Right):  cx > 213.33px                  -> "ben phai"
     - Evaluates the detection with the highest confidence score for the target class.
     - If target class is absent, returns "Khong co".
  5. OUT_OF_SCOPE:
     - Returns "OOS".
"""

from .entity_extractor import CLASS_ID_TO_NAME

IMAGE_WIDTH = 320.0


def determine_spatial_zone(bbox, image_width=IMAGE_WIDTH):
    """
    Determines whether a bounding box [xmin, ymin, xmax, ymax]
    is in 'ben trai', 'o giua', or 'ben phai' based on box horizontal center.
    Zone 1 (Left):   cx < W / 3
    Zone 2 (Middle): W / 3 <= cx <= 2 * W / 3
    Zone 3 (Right):  cx > 2 * W / 3
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0

    zone_boundary_1 = image_width / 3.0           # ~106.67 px
    zone_boundary_2 = (2.0 * image_width) / 3.0   # ~213.33 px

    if cx < zone_boundary_1:
        return "ben trai"
    elif cx <= zone_boundary_2:
        return "o giua"
    else:
        return "ben phai"


def format_answer(intent, target_class, detections, image_width=IMAGE_WIDTH):
    """
    Formats standardized answer string for competition submission / finals.

    Args:
      intent: str - One of ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
      target_class: int - Targeted class ID (0..6) or -1 if none
      detections: list of dicts with:
                  {'class_id': int, 'score': float, 'bbox': [xmin, ymin, xmax, ymax]}
      image_width: float - Image width (default: 320.0)

    Returns:
      str: Standardized answer string.
    """
    intent = intent.upper().strip()

    if intent == "OUT_OF_SCOPE":
        return "OOS"

    # Filter detections for the requested target class
    if target_class is not None and target_class >= 0:
        target_boxes = [d for d in detections if d["class_id"] == target_class]
    else:
        target_boxes = detections

    if intent == "COUNT":
        return str(len(target_boxes))

    elif intent == "EXIST":
        return "Co" if len(target_boxes) > 0 else "Khong"

    elif intent == "LOCATION":
        if len(target_boxes) == 0:
            return "Khong co"
        # Select target detection with highest confidence score
        best_det = max(target_boxes, key=lambda x: x.get("score", 0.0))
        return determine_spatial_zone(best_det["bbox"], image_width=image_width)

    elif intent == "LIST":
        if len(detections) == 0:
            return "Khong co"
        # Count frequency of each detected class
        counts = {}
        for d in detections:
            cid = d["class_id"]
            cname = CLASS_ID_TO_NAME.get(cid, f"class_{cid}")
            counts[cname] = counts.get(cname, 0) + 1

        # Format sorted alphabetically by class name
        parts = [f"{cnt} {cname}" for cname, cnt in sorted(counts.items())]
        return ", ".join(parts)

    else:
        return "OOS"
