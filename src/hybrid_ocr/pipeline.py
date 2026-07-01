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


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Cân bằng độ tương phản cục bộ CLAHE trên ảnh màu BGR bằng cách chuyển đổi qua hệ LAB.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


class ONNXHybridOCR:
    """
    ONNX wrapper for HybridOCR model to run inference using onnxruntime.
    """
    def __init__(self, model_path: str, vocab: Vocabulary):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required to run ONNX inference: pip install onnxruntime")
        
        # Load ONNX session
        self.session = ort.InferenceSession(model_path)
        self.vocab = vocab
        
        # Đọc độ dài chuỗi cố định từ input shape thứ 2 (target_input) của ONNX
        try:
            self.max_len = int(self.session.get_inputs()[1].shape[1])
        except (IndexError, ValueError, TypeError):
            self.max_len = 100

    def predict(
        self,
        images: torch.Tensor,
        vocab: Vocabulary,
        decoding: str = "greedy",
        beam_width: int = 5,
        max_len: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if decoding != "greedy":
            print(f"  WARNING: Only 'greedy' decoding is supported for ONNX. Falling back from '{decoding}' to 'greedy'.")
            
        max_len = max_len or self.max_len
        
        # Convert torch Tensor to numpy
        images_np = images.cpu().numpy()
        batch_size = images_np.shape[0]
        
        sos_idx = vocab.sos_idx
        eos_idx = vocab.eos_idx
        pad_idx = vocab.pad_idx
        
        # Initialize target_input (B, max_len) filled with PAD and set SOS at index 0
        target_input = np.ones((batch_size, max_len), dtype=np.int64) * pad_idx
        target_input[:, 0] = sos_idx
        
        finished = np.zeros(batch_size, dtype=bool)
        char_confidences = [[] for _ in range(batch_size)]
        
        for step in range(max_len):
            # Run session with the constant-sized target_input (B, max_len)
            inputs = {
                "images": images_np,
                "target_input": target_input,
            }
            outputs = self.session.run(["logits"], inputs)
            logits = outputs[0]  # (B, max_len, vocab_size)
            
            # Get logits at the current step (step)
            next_step_logits = logits[:, step, :]  # (B, vocab_size)
            
            # Apply softmax to logits to get probabilities
            exp_logits = np.exp(next_step_logits - np.max(next_step_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            next_tokens = np.argmax(probs, axis=-1)  # (B,)
            
            # Update target_input and finished status
            for i in range(batch_size):
                if not finished[i]:
                    token = next_tokens[i]
                    char_confidences[i].append(float(probs[i, token]))
                    if token == eos_idx:
                        finished[i] = True
            
            # Append token to the next index in target_input (if not at the end)
            if step < max_len - 1:
                target_input[:, step + 1] = next_tokens
            
            if finished.all():
                break
                
        results = []
        for i in range(batch_size):
            tokens = target_input[i].tolist()
            clean_tokens = []
            for t in tokens:
                clean_tokens.append(t)
                if t == eos_idx:
                    break
            text = vocab.decode(clean_tokens)
            avg_confidence = sum(char_confidences[i]) / len(char_confidences[i]) if char_confidences[i] else 0.0
            
            results.append({
                "text": text,
                "tokens": clean_tokens,
                "confidence": avg_confidence,
                "char_confidences": char_confidences[i],
            })
            
        return results


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
        use_clahe: bool = False,
        detector_type: str = "paddle",
    ):
        print("Initializing Hybrid OCR Pipeline...")
        self.use_clahe = use_clahe
        self.detector_type = detector_type
        
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
        
        # Load detector
        print(f"  Loading Detection Model ({detector_type.upper()})...")
        self.detector = TextDetector(
            model_path=yolo_model_path,
            device=self.device,
            detector_type=self.detector_type,
        )
        
        # Load Recognition model
        if rec_model_path.lower() == "mangaocr":
            local_dir = "models/manga-ocr-base"
            if not os.path.exists(local_dir):
                print(f"  Local model not found. Downloading MangaOCR model from Hugging Face to: {local_dir}...")
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id="kha-white/manga-ocr-base", local_dir=local_dir)
                except ImportError:
                    print("  WARNING: huggingface_hub not found. Will let manga-ocr library download it automatically.")
            
            print(f"  Loading Recognition Model (MangaOCR from local path: {local_dir})...")
            try:
                from manga_ocr import MangaOcr
            except ImportError:
                raise ImportError(
                    "manga-ocr is required for MangaOCR recognition: pip install manga-ocr"
                )
            self.mocr = MangaOcr(pretrained_model_name_or_path=local_dir)
            self.rec_model_type = "mangaocr"
            # Set dummy defaults for compatibility
            self.img_h = 64
            self.img_w = 256
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            print("  Loading Recognition Model (ConvNeXt + Transformer)...")
            self.rec_model_type = "hybrid"
            config = load_config(rec_config_path)
            
            # Try to find vocabulary alongside checkpoint or config
            vocab_path = Path(rec_model_path).parent / "vocab.json"
            if not vocab_path.exists():
                vocab_path = Path(rec_config_path).parent / "vocab.json"
            if not vocab_path.exists():
                vocab_path = Path("runs/finetune") / "vocab.json"
            if not vocab_path.exists():
                vocab_path = Path("runs/recognition") / "vocab.json"
                
            if vocab_path.exists():
                print(f"  Loaded vocabulary from: {vocab_path}")
                self.vocab = Vocabulary.load(str(vocab_path))
            else:
                print("  WARNING: vocab.json not found, building default vocabulary")
                self.vocab = Vocabulary.build_japanese_auction_vocab()
                
            if rec_model_path.endswith(".onnx"):
                print("  Detected ONNX format. Initializing ONNX Runtime Session...")
                self.recognizer = ONNXHybridOCR(rec_model_path, self.vocab)
            else:
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
        # Apply CLAHE if enabled
        if self.use_clahe:
            crop = apply_clahe(crop)
            
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

    def _visualize_predictions(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Vẽ bounding box màu đỏ và văn bản dự đoán tiếng Nhật kèm độ tin cậy màu đỏ lên ảnh sử dụng Pillow.
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # 1. Vẽ bounding box màu đỏ bằng OpenCV trước
        vis = image.copy()
        color_bgr = (0, 0, 255)  # Màu đỏ trong hệ BGR của OpenCV
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)
            
        # 2. Chuyển đổi sang ảnh PIL để vẽ chữ Unicode (tiếng Nhật)
        img_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Danh sách đường dẫn các font tiếng Nhật phổ biến trên macOS và Linux (RunPod)
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
            "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, 13)
                    break
                except IOError:
                    continue
        
        if font is None:
            font = ImageFont.load_default()
            
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            text = det.get("recognized_text", "")
            conf = det.get("rec_confidence", 0.0)
            
            label = f"{text} ({conf*100:.0f}%)" if text else ""
            if label:
                # Tính toán kích thước chữ để tạo background che khuất nét box bên dưới
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                    w = right - left
                    h = bottom - top
                except AttributeError:
                    w, h = draw.textsize(label, font=font)
                
                # Vẽ chữ màu đỏ trực tiếp lên ảnh dạng trong suốt (hệ PIL RGB là (255, 0, 0))
                bg_y1 = max(0, y1 - h - 5)
                draw.text((x1 + 2, bg_y1), label, font=font, fill=(255, 0, 0))
                
                
        # 3. Chuyển đổi ngược về OpenCV BGR
        vis = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return vis


    def process_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single document page image.
        """
        import time
        t_start = time.perf_counter()
        
        img_name = Path(image_path).name
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # 1. Detection
        t0 = time.perf_counter()
        detections = self.detector.detect(img, return_crops=True, apply_deskew=True)
        t_det = time.perf_counter() - t0
        
        # 2. Recognition
        # Preprocessing crops
        t0 = time.perf_counter()
        
        if self.rec_model_type == "mangaocr":
            # MangaOCR handles PIL images directly and doesn't use the standard preprocessing pipeline
            t_prep = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            from PIL import Image
            for det in detections:
                crop = det["crop"]
                if crop.size > 0:
                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    # Predict using MangaOcr
                    text = self.mocr(pil_crop)
                    det["recognized_text"] = text
                    det["rec_confidence"] = 1.0  # MangaOCR does not output confidence scores
                else:
                    det["recognized_text"] = ""
                    det["rec_confidence"] = 0.0
                    
                # Clean up crops to make it JSON serializable
                if "crop" in det:
                    del det["crop"]
            t_rec = time.perf_counter() - t0
            num_crops = len(detections)
        else:
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
            t_prep = time.perf_counter() - t0
                    
            # Running model inference
            t0 = time.perf_counter()
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
            t_rec = time.perf_counter() - t0
            num_crops = len(all_tensors)
        
        # 3. Visualization
        t0 = time.perf_counter()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Ảnh 1: Bounding Box màu xanh (mặc định)
            vis = self.detector.visualize(img, detections)
            cv2.imwrite(str(Path(output_dir) / f"vis_{img_name}"), vis)
            
            # Ảnh 2: Bounding Box màu đỏ + Text nhận diện kèm Confidence màu đỏ
            vis_pred = self._visualize_predictions(img, detections)
            cv2.imwrite(str(Path(output_dir) / f"vis_text_{img_name}"), vis_pred)
        t_vis = time.perf_counter() - t0
            
        # 4. Post-processing (Structure Extraction)
        t0 = time.perf_counter()
        structured_data = extract_structured_data(detections)
        t_struct = time.perf_counter() - t0
        
        t_total = time.perf_counter() - t_start
        print(f"    - [Time Summary] Detection: {t_det:.3f}s | Preprocess: {t_prep:.3f}s | Recognition (x{num_crops}): {t_rec:.3f}s | Visualization: {t_vis:.3f}s | Structure: {t_struct:.3f}s | Total: {t_total:.3f}s")
        
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
        import time
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing PDF: {pdf_path}")
        
        # Render PDF to images
        t0 = time.perf_counter()
        tmp_dir = Path(output_dir) / "tmp_pages"
        image_paths = render_pdf_pages(
            pdf_path, str(tmp_dir), dpi=200, page_range=page_range
        )
        t_render = time.perf_counter() - t0
        print(f"  Rendered PDF to {len(image_paths)} images in {t_render:.3f}s")
        
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
    parser.add_argument("--use-clahe", action="store_true", help="Use CLAHE local contrast enhancement on crops")
    parser.add_argument(
        "--detector",
        choices=["paddle", "yolo", "surya"],
        default="paddle",
        help="Detector backend to use for finding text boxes ('paddle', 'yolo', or 'surya')"
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
        use_clahe=args.use_clahe,
        detector_type=args.detector,
    )
    
    pipeline.process_pdf(args.pdf, output_dir=args.output, page_range=page_range)
