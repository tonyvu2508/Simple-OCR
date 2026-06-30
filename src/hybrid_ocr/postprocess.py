"""
Post-processing and field extraction logic for the auction OCR pipeline.
"""

from __future__ import annotations

import re
from typing import Dict, List, Any


def extract_structured_data(detections: List[Dict]) -> Dict[str, Any]:
    """
    Extract structured data from raw detections.
    
    Args:
        detections: List of detection dictionaries containing bounding boxes,
                    class labels, and recognized text.
    
    Returns:
        Structured dictionary with extracted fields.
    """
    structured_data = {
        "car_name": None,
        "model": None,
        "year": None,
        "mileage": None,
        "color": None,
        "grade": None,
        "engine": None,
        "rating": None,
        "start_price": None,
        "sold_price": None,
        "chassis_number": None,
        "registration_number": None,
        "notes": [],
    }
    
    # Sort detections by y-coordinate to process top-to-bottom
    sorted_dets = sorted(detections, key=lambda d: d["bbox"][1])
    
    for det in sorted_dets:
        class_name = det.get("class_name")
        text = det.get("recognized_text", "").strip()
        if not text:
            continue
            
        # Basic heuristic mapping based on class and text content
        # Note: In a real production system, this would use a more sophisticated
        # layout analysis or an LLM to map text to specific fields.
        
        if class_name == "header_field":
            if "万円" in text:
                if not structured_data["start_price"]:
                    structured_data["start_price"] = text
                else:
                    structured_data["sold_price"] = text
            elif "千km" in text or "万km" in text:
                structured_data["mileage"] = text
            elif "年" in text and len(text) <= 5:
                structured_data["year"] = text
            elif "点" in text:
                structured_data["rating"] = text
            elif re.search(r"[0-9]{4}cc", text):
                structured_data["engine"] = text
        
        elif class_name == "car_info":
            if "-" in text and len(text) > 8:
                structured_data["chassis_number"] = text
                
        elif class_name == "handwritten" or class_name == "printed_text":
            if len(text) > 5:
                structured_data["notes"].append(text)

    return structured_data
