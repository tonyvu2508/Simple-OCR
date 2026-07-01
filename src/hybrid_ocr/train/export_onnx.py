import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path
from src.hybrid_ocr.train.config import load_config
from src.hybrid_ocr.recognition.model import build_model_from_config
from src.hybrid_ocr.dataset.vocabulary import Vocabulary

def export_onnx(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
):
    # 1. Load config
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    input_cfg = config.get("input", {})
    
    # 2. Determine device
    device = torch.device("cpu") # Exporting on CPU is recommended
    
    # 3. Load Vocabulary
    # Find vocab in the log_dir of configs
    log_dir = config.get("training", {}).get("log_dir", "runs/recognition")
    # Try finding it in checkpoint dir or log dir
    vocab_path = Path(checkpoint_path).parent / "vocab.json"
    if not vocab_path.exists():
        vocab_path = Path(log_dir) / "vocab.json"
    if not vocab_path.exists():
        vocab_path = Path("runs/finetune") / "vocab.json"
        
    if not vocab_path.exists():
        print(f"⚠️ Không tìm thấy vocab.json tại {vocab_path}, đang khởi tạo bộ từ vựng mặc định...")
        vocab = Vocabulary.build_japanese_auction_vocab()
    else:
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load(str(vocab_path))
        
    # 4. Build Model
    print("Building model...")
    model = build_model_from_config(config, vocab)
    
    # 5. Load Checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 6. Create Dummy Inputs
    img_h = input_cfg.get("image_height", 64)
    img_w = input_cfg.get("image_width", 256)
    max_seq_len = model_cfg.get("decoder", {}).get("max_seq_length", 100)
    
    dummy_images = torch.randn(1, 3, img_h, img_w, device=device)
    dummy_targets = torch.ones(1, max_seq_len, dtype=torch.long, device=device) * vocab.sos_idx
    
    # Ensure output dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 7. Export Entire Model
    print(f"Exporting model to ONNX format at {output_path} (max_seq_length={max_seq_len})...")
    torch.onnx.export(
        model,
        (dummy_images, dummy_targets),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["images", "target_input"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "target_input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }
    )
    print("✅ Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch Hybrid OCR to ONNX format")
    parser.add_argument("--config", default="configs/recognition.yaml", help="Path to recognition config yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", default="models/recognition/model.onnx", help="Path to output ONNX file")
    
    args = parser.parse_args()
    
    export_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
