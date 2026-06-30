"""
PaddleOCR-based text region detection for auction documents.

Handles the first stage of the OCR pipeline: detecting text regions
in document page images. Extracts crops with optional deskewing
for downstream recognition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Workaround for PaddlePaddle 3.x MKLDNN/onednn crash on CPU
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
import numpy as np

from ..dataset.augmentation import deskew_image


# Class names for detection compatibility
DETECTION_CLASSES = {
    0: "header_field",
    1: "auction_form",
    2: "form_field",
    3: "handwritten",
    4: "printed_text",
    5: "car_info",
}


class TextDetector:
    """
    PaddleOCR-based text region detector for auction documents.
    
    Detects text regions on page images, returning bounding boxes.
    
    Args:
        model_path: Ignored, kept for API compatibility.
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: Ignored, kept for API compatibility.
        device: Device to run inference on ('cpu', 'cuda', 'mps', 'auto').
        imgsz: Ignored, kept for API compatibility.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        device: str = "auto",
        imgsz: int = 1280,
    ):
        self.confidence_threshold = confidence_threshold
        
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "paddleocr is required for detection: pip install paddleocr"
            )
            
        use_gpu = False
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                use_gpu = True
        elif device == "cuda":
            use_gpu = True
            
        # Initialize PaddleOCR detector only
        self.model = PaddleOCR(
            lang="japan",
        )

    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = True,
        apply_deskew: bool = True,
        crop_padding: int = 5,
    ) -> List[Dict]:
        """
        Detect text regions in a page image using PaddleOCR.
        
        Args:
            image: Page image (H, W, 3), BGR format.
            return_crops: Whether to include cropped images in results.
            apply_deskew: Whether to deskew cropped text regions.
            crop_padding: Padding around detected regions (pixels).
        
        Returns:
            List of detection dictionaries:
                - "bbox": (x1, y1, x2, y2) bounding box coordinates
                - "class_id": Integer class ID (always 4 for printed_text)
                - "class_name": String class name
                - "confidence": Detection confidence score
                - "crop": Cropped image (if return_crops=True)
        """
        # Run PaddleOCR prediction
        results = self.model.predict(image)
        
        detections = []
        h, w = image.shape[:2]
        
        if results and len(results) > 0:
            result = results[0]
            dt_polys = result.get("dt_polys", [])
            for box in dt_polys:
                # box structure is 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                pts = np.array(box, dtype=np.int32)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                
                # Apply padding (clamped to image bounds)
                x1 = max(0, x1 - crop_padding)
                y1 = max(0, y1 - crop_padding)
                x2 = min(w, x2 + crop_padding)
                y2 = min(h, y2 + crop_padding)
                
                detection = {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": 4, # Defaults to printed_text
                    "class_name": DETECTION_CLASSES[4],
                    "confidence": 0.99, # Dummy confidence
                }
                
                if return_crops:
                    crop = image[y1:y2, x1:x2].copy()
                    
                    # Optionally deskew the crop
                    if apply_deskew and crop.size > 0:
                        crop = deskew_image(crop)
                    
                    detection["crop"] = crop
                
                detections.append(detection)
        
        # Sort by position: top-to-bottom, left-to-right
        detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        
        return detections

    def detect_from_pdf(
        self,
        pdf_path: str,
        dpi: int = 200,
        page_range: Optional[Tuple[int, int]] = None,
        **detect_kwargs,
    ) -> Dict[int, List[Dict]]:
        """
        Detect text regions in all pages of a PDF.
        
        Args:
            pdf_path: Path to PDF file.
            dpi: Resolution for rendering PDF pages.
            page_range: Optional (start, end) page range (0-indexed, inclusive).
            **detect_kwargs: Additional arguments passed to detect().
        
        Returns:
            Dictionary mapping page number to list of detections.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required: pip install PyMuPDF")
        
        doc = fitz.open(pdf_path)
        
        start = page_range[0] if page_range else 0
        end = page_range[1] + 1 if page_range else len(doc)
        
        all_detections = {}
        
        for page_num in range(start, min(end, len(doc))):
            page = doc[page_num]
            
            # Render page to image
            pix = page.get_pixmap(dpi=dpi)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            
            if pix.n == 4:  # RGBA
                img = img_data.reshape(pix.h, pix.w, 4)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                img = img_data.reshape(pix.h, pix.w, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:  # Grayscale
                img = img_data.reshape(pix.h, pix.w)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Detect text regions
            detections = self.detect(img, **detect_kwargs)
            all_detections[page_num] = detections
            
            print(f"  Page {page_num + 1}/{len(doc)}: {len(detections)} regions detected")
        
        doc.close()
        
        return all_detections

    def visualize(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on image for visualization.
        
        Args:
            image: Original image (H, W, 3), BGR.
            detections: List of detection dictionaries.
            output_path: Optional path to save visualization.
        
        Returns:
            Image with bounding boxes drawn.
        """
        vis = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0) # Green for all text detections
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, vis)
        
        return vis


def render_pdf_pages(
    pdf_path: str,
    output_dir: str,
    dpi: int = 200,
    page_range: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """
    Render PDF pages as individual images.
    
    Args:
        pdf_path: Path to PDF file.
        output_dir: Directory to save page images.
        dpi: Rendering resolution.
        page_range: Optional (start, end) page range (0-indexed).
    
    Returns:
        List of saved image file paths.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required: pip install PyMuPDF")
    
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    start = page_range[0] if page_range else 0
    end = page_range[1] + 1 if page_range else len(doc)
    
    saved_paths = []
    
    for page_num in range(start, min(end, len(doc))):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        
        img_path = os.path.join(output_dir, f"page_{page_num:04d}.png")
        pix.save(img_path)
        saved_paths.append(img_path)
    
    doc.close()
    
    return saved_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect text regions in PDF documents using PaddleOCR")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output", default="results/detections", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200, help="Rendering DPI")
    args = parser.parse_args()
    
    detector = TextDetector()
    detections = detector.detect_from_pdf(args.pdf, dpi=args.dpi)
    
    print(f"\nTotal pages processed: {len(detections)}")
    for page_num, dets in detections.items():
        print(f"  Page {page_num}: {len(dets)} detections")
