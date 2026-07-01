import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.hybrid_ocr.detection.detect import TextDetector, render_pdf_pages
from src.hybrid_ocr.recognition.model import HybridOCR
from src.hybrid_ocr.dataset.vocabulary import Vocabulary
from src.hybrid_ocr.dataset.augmentation import letterbox_image
from src.hybrid_ocr.train.config import load_config

def preprocess_crop(crop, img_h, img_w, mean, std):
    """Preprocess image crop for recognition model."""
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img, _, _ = letterbox_image(img, img_h, img_w, pad_value=0)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return tensor

def main():
    parser = argparse.ArgumentParser(description="Extract crops from PDF and auto-label for Fine-tuning")
    parser.add_argument("--pdf", default="pdfs/2026年6月25日-JU愛知-2163-通常車-151-200.pdf", help="Path to input PDF")
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="Path to YOLO model (if used, else PaddleOCR)")
    parser.add_argument("--rec-model", default="runs/recognition/model_last.pt", help="Current best recognition checkpoint")
    parser.add_argument("--rec-config", default="configs/recognition.yaml", help="Path to recognition config")
    parser.add_argument("--output-dir", default="data/real_fine_tune", help="Output directory for dataset")
    parser.add_argument("--max-pages", type=int, default=5, help="Maximum number of pages to process for fine-tune data")
    parser.add_argument(
        "--labeler",
        choices=["paddle", "hybrid", "glmocr"],
        default="paddle",
        help="Model backend to use for auto-labeling ('paddle', 'hybrid', or 'glmocr')"
    )
    args = parser.parse_args()

    # Create directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    labeler_type = args.labeler.lower()
    
    # Load vocab if using hybrid
    vocab = None
    if labeler_type == "hybrid":
        vocab_path = Path(args.rec_model).parent / "vocab.json"
        if not vocab_path.exists():
            vocab_path = Path(args.rec_config).parent / "vocab.json"
        
        if vocab_path.exists():
            vocab = Vocabulary.load(str(vocab_path))
            print(f"Loaded vocabulary from {vocab_path} (size: {vocab.size})")
        else:
            vocab = Vocabulary.build_japanese_auction_vocab()
            print(f"Built default vocabulary (size: {vocab.size})")

    # Load Labeler Model
    recognizer = None
    glm_processor = None
    glm_model = None
    
    if labeler_type == "hybrid":
        print("Loading Custom Recognition Model (Hybrid)...")
        config = load_config(args.rec_config)
        recognizer = HybridOCR.load_checkpoint(
            args.rec_model,
            vocab_size=vocab.size,
            device=device,
        )
        recognizer.eval()

        img_h = config.get("input", {}).get("image_height", 64)
        img_w = config.get("input", {}).get("image_width", 256)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
    elif labeler_type == "glmocr":
        print("Loading GLM-OCR Model for auto-labeling...")
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError("transformers is required for GLM-OCR. Please install it.")
            
        local_dir = "models/glm-ocr"
        if not os.path.exists(local_dir):
            print(f"  Local GLM-OCR not found. Downloading to {local_dir}...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="zai-org/GLM-OCR", local_dir=local_dir)
            
        glm_processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        glm_model = AutoModelForImageTextToText.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32 if device == "mps" else torch.bfloat16
        ).to(device)
        glm_model.eval()

    # Load Detector
    print("Loading Text Detector (PaddleOCR)...")
    detector = TextDetector(model_path=args.yolo_model, device=device)

    # Render PDF pages
    print(f"Rendering PDF pages (max {args.max_pages} pages)...")
    tmp_pdf_dir = output_dir / "tmp_pdf_pages"
    page_paths = render_pdf_pages(args.pdf, str(tmp_pdf_dir), dpi=200, page_range=(0, args.max_pages - 1))

    # Load existing labels if they exist
    labels_path = output_dir / "labels.json"
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            print(f"Loaded {len(labels)} existing labels from {labels_path}")
        except Exception:
            labels = {}
    else:
        labels = {}
    
    crop_counter = 0

    print("Processing pages and extracting text regions...")
    for page_idx, page_path in enumerate(page_paths):
        print(f"  Page {page_idx + 1}/{len(page_paths)}...")
        img = cv2.imread(page_path)
        if img is None:
            continue
        
        # Detect text boxes
        detections = detector.detect(img, return_crops=True, apply_deskew=True)
        print(f"    Detected {len(detections)} text regions.")

        if labeler_type == "paddle":
            # Use PaddleOCR's own recognized text directly
            for crop_idx, det in enumerate(detections):
                crop = det["crop"]
                text = det.get("text", "")
                
                if crop.size > 0:
                    crop_name = f"page_{page_idx:04d}_crop_{crop_idx:04d}.png"
                    crop_path = images_dir / crop_name
                    
                    # Save image crop
                    cv2.imwrite(str(crop_path), crop)
                    
                    # Save label mapping
                    labels[crop_name] = text
                    crop_counter += 1
                    
        elif labeler_type == "glmocr":
            # Use GLM-OCR VLM for high-quality labeling
            from PIL import Image
            for crop_idx, det in enumerate(detections):
                crop = det["crop"]
                if crop.size > 0:
                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_crop},
                                {"type": "text", "text": "Text Recognition:"},
                            ],
                        }
                    ]
                    inputs = glm_processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(glm_model.device)
                    
                    with torch.no_grad():
                        output = glm_model.generate(**inputs, max_new_tokens=128)
                        
                    prompt_len = inputs["input_ids"].shape[1]
                    text = glm_processor.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
                    
                    crop_name = f"page_{page_idx:04d}_crop_{crop_idx:04d}.png"
                    crop_path = images_dir / crop_name
                    cv2.imwrite(str(crop_path), crop)
                    labels[crop_name] = text
                    crop_counter += 1
                    
        elif labeler_type == "hybrid":
            # Use custom recognition model
            all_crops = []
            all_tensors = []
            
            for det in detections:
                crop = det["crop"]
                if crop.size > 0:
                    all_crops.append(crop)
                    tensor = preprocess_crop(crop, img_h, img_w, mean, std)
                    all_tensors.append(tensor)

            if all_tensors:
                predicted_texts = []
                batch_size = 16
                with torch.no_grad():
                    for i in range(0, len(all_tensors), batch_size):
                        batch_tensors = all_tensors[i:i+batch_size]
                        batch_stack = torch.stack(batch_tensors).to(device)
                        
                        results = recognizer.predict(
                            batch_stack, vocab, decoding="greedy"
                        )
                        for res in results:
                            predicted_texts.append(res["text"])

                # Save crops and predictions
                for crop_idx, (crop, text) in enumerate(zip(all_crops, predicted_texts)):
                    crop_name = f"page_{page_idx:04d}_crop_{crop_idx:04d}.png"
                    crop_path = images_dir / crop_name
                    
                    # Save image crop
                    cv2.imwrite(str(crop_path), crop)
                    
                    # Save label mapping
                    labels[crop_name] = text
                    crop_counter += 1

        # Save labels.json incrementally after each page
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"    Saved progress for Page {page_idx + 1} to {labels_path}")

    print(f"\nDone! Extracted {crop_counter} new crops in this run.")
    print(f"Dataset generated at: {output_dir}")
    print(f"Images are saved in: {images_dir}")
    print(f"Initial labels.json saved at: {labels_path}")
    print("\nNext step: Open labels.json, review and correct the auto-generated texts, then start fine-tuning!")

if __name__ == "__main__":
    main()
