"""
End-to-end Hybrid OCR Pipeline.

Orchestrates the entire process:
1. PDF/Image reading
2. YOLO detection (find text regions)
3. Preprocessing (crop, letterbox, deskew)
4. Recognition (ConvNeXt + Transformer)
5. Post-processing (structure extraction)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

try:
    import torch
except ImportError:
    pass

from .detection.detect import TextDetector, render_pdf_pages
from .recognition.model import HybridOCR, build_model_from_config
from .dataset.vocabulary import Vocabulary
from .dataset.augmentation import letterbox_image
from .train.config import load_config
from .postprocess import extract_structured_data


class AuctionOCRPipeline:
    """
    End-to-end OCR pipeline for Japanese auction documents.
    """

    def __init__(
        self,
        yolo_model_path: str,
        rec_model_path: str,
        rec_config_path: str,
        device: str = "auto",
    ):
        print("Initializing Hybrid OCR Pipeline...")
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"  Using device: {self.device}")
        
        # Load YOLO detector
        print("  Loading Detection Model (YOLO)...")
        self.detector = TextDetector(
            model_path=yolo_model_path,
            device=self.device,
        )
        
        # Load Recognition model
        print("  Loading Recognition Model (ConvNeXt + Transformer)...")
        config = load_config(rec_config_path)
        
        # Try to find vocabulary alongside checkpoint or config
        vocab_path = Path(rec_model_path).parent / "vocab.json"
        if not vocab_path.exists():
            vocab_path = Path(rec_config_path).parent / "vocab.json"
            
        if vocab_path.exists():
            self.vocab = Vocabulary.load(str(vocab_path))
        else:
            print("  WARNING: vocab.json not found, building default vocabulary")
            self.vocab = Vocabulary.build_japanese_auction_vocab()
            
        self.recognizer = HybridOCR.load_checkpoint(
            rec_model_path,
            vocab_size=self.vocab.size,
            device=self.device,
        )
        self.recognizer.eval()
        
        # Target image size for recognition
        self.img_h = config.get("input", {}).get("image_height", 64)
        self.img_w = config.get("input", {}).get("image_width", 256)
        
        # Normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Preprocess image crop for recognition model."""
        # Convert BGR to RGB
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Letterbox resize
        img, _, _ = letterbox_image(img, self.img_h, self.img_w, pad_value=0)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # To tensor
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return tensor

    def process_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single document page image.
        """
        img_name = Path(image_path).name
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # 1. Detection
        detections = self.detector.detect(img, return_crops=True, apply_deskew=True)
        
        # 2. Recognition
        # Process in batches for speed if many detections
        batch_size = 16
        all_tensors = []
        
        for det in detections:
            crop = det["crop"]
            if crop.size > 0:
                tensor = self._preprocess_crop(crop)
                all_tensors.append(tensor)
            else:
                det["recognized_text"] = ""
                det["confidence"] = 0.0
                
        if all_tensors:
            with torch.no_grad():
                for i in range(0, len(all_tensors), batch_size):
                    batch_tensors = all_tensors[i:i+batch_size]
                    batch_stack = torch.stack(batch_tensors).to(self.device)
                    
                    results = self.recognizer.predict(
                        batch_stack, self.vocab, decoding="greedy"
                    )
                    
                    # Update detection dictionaries
                    for j, res in enumerate(results):
                        idx = i + j
                        detections[idx]["recognized_text"] = res["text"]
                        detections[idx]["rec_confidence"] = res["confidence"]
                        # Remove crop from dict to make it JSON serializable
                        if "crop" in detections[idx]:
                            del detections[idx]["crop"]
        
        # 3. Visualization
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            vis = self.detector.visualize(img, detections)
            cv2.imwrite(str(Path(output_dir) / f"vis_{img_name}"), vis)
            
        # 4. Post-processing (Structure Extraction)
        structured_data = extract_structured_data(detections)
        
        return {
            "file": img_name,
            "raw_detections": detections,
            "structured_data": structured_data,
        }

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str = "results",
        page_range: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process an entire PDF document.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing PDF: {pdf_path}")
        
        # Render PDF to images
        tmp_dir = Path(output_dir) / "tmp_pages"
        image_paths = render_pdf_pages(
            pdf_path, str(tmp_dir), dpi=200, page_range=page_range
        )
        
        results = []
        json_path = Path(output_dir) / f"{Path(pdf_path).stem}_results.json"
        
        for i, img_path in enumerate(image_paths):
            print(f"  Processing page {i+1}/{len(image_paths)}...")
            page_result = self.process_image(img_path, output_dir=output_dir)
            results.append(page_result)
            
            # Save progress incrementally after each page
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"\nDone! Results saved to {json_path}")
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-End Hybrid OCR Pipeline")
    parser.add_argument("--pdf", required=True, help="Input PDF file")
    parser.add_argument("--yolo-model", required=True, help="YOLO model .pt path")
    parser.add_argument("--rec-model", required=True, help="Recognition model .pt path")
    parser.add_argument("--rec-config", default="configs/recognition.yaml")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--pages",
        default="all",
        help="Pages to process, e.g. '0' (page 1), '0-4' (pages 1 to 5), or 'all' (default)"
    )
    
    args = parser.parse_args()
    
    # Parse page range
    page_range = None
    if args.pages and args.pages != "all":
        if "-" in args.pages:
            try:
                start, end = map(int, args.pages.split("-"))
                page_range = (start, end)
            except ValueError:
                raise ValueError("Invalid page range format. Use e.g. '0-4' or '0'")
        else:
            try:
                page_idx = int(args.pages)
                page_range = (page_idx, page_idx)
            except ValueError:
                raise ValueError("Invalid page format. Use e.g. '0-4' or '0'")
    
    pipeline = AuctionOCRPipeline(
        yolo_model_path=args.yolo_model,
        rec_model_path=args.rec_model,
        rec_config_path=args.rec_config,
    )
    
    pipeline.process_pdf(args.pdf, output_dir=args.output, page_range=page_range)
