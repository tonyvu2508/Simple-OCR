"""
Training script for the ConvNeXt + Transformer OCR recognition model.

Supports both synthetic pre-training and real data fine-tuning.
Implements label smoothing cross entropy loss and learning rate scheduling.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset.augmentation import OCRAugmentor
from ..dataset.dataset import OCRDataset, collate_fn
from ..dataset.vocabulary import Vocabulary
from ..recognition.model import HybridOCR, build_model_from_config
from .config import load_config
from .utils import AverageMeter, calculate_metrics


class LabelSmoothingLoss(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Helps prevent the model from becoming overly confident, which is
    especially useful in OCR where characters can look very similar
    (e.g., 0/O, 1/l, ア/了).
    
    Args:
        classes: Number of classes (vocabulary size).
        smoothing: Smoothing factor (epsilon).
        ignore_index: Target index to ignore (e.g., padding token).
    """

    def __init__(
        self,
        classes: int,
        smoothing: float = 0.1,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pred: Logits (N, C) where C = number of classes.
            target: Ground truth indices (N,).
            
        Returns:
            Scalar loss.
        """
        assert pred.dim() == 2
        assert target.dim() == 1
        
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 2))  # Exclude target and padding
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
            if mask.dim() > 0 and mask.size(0) > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
                
        return self.criterion(pred, true_dist)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    vocab: Vocabulary,
    scaler: torch.cuda.amp.GradScaler,
    clip_grad: float = 5.0,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move batch to device
        images = batch["image"].to(device)
        target_input = batch["target_input"].to(device)
        target_output = batch["target_output"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            logits = model(images, target_input)
            
            # Calculate loss
            # Flatten logits and targets: (B, seq_len, vocab_size) -> (B*seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_output.view(-1)
            loss = criterion(logits_flat, target_flat)
        
        # Calculate sequence accuracy (Teacher Forcing)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # (B, seq_len)
            mask = target_output != vocab.pad_idx
            # Sequence is correct if all non-pad tokens match exactly
            correct_tokens = (preds == target_output) | ~mask
            seq_correct = correct_tokens.all(dim=-1).sum().item()
            acc_meter.update(seq_correct / images.size(0), images.size(0))
        
        # Backward pass and optimize with AMP scaler
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")
        
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab: Vocabulary,
) -> Tuple[float, float, float]:
    """
    Evaluate the model on validation data.
    
    Returns:
        Tuple of (loss, accuracy, CER).
    """
    model.eval()
    loss_meter = AverageMeter()
    
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        images = batch["image"].to(device)
        target_input = batch["target_input"].to(device)
        target_output = batch["target_output"].to(device)
        
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            # Forward pass for loss
            logits = model(images, target_input)
            
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_output.view(-1)
            loss = criterion(logits_flat, target_flat)
            
        loss_meter.update(loss.item(), images.size(0))
        
        # Greedy decoding for accuracy/CER
        # We use a subset of validation for actual decoding if it's too slow
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            generated = model.predict(
                images, vocab, decoding="greedy", max_len=target_output.size(1)
            )
        
        # Convert targets to strings
        for i in range(images.size(0)):
            # Target
            tgt_tokens = target_output[i].cpu().tolist()
            tgt_text = vocab.decode(tgt_tokens)
            all_targets.append(tgt_text)
            
            # Prediction
            pred_text = generated[i]["text"]
            all_preds.append(pred_text)
            
    # Calculate metrics
    accuracy, cer, wer = calculate_metrics(all_preds, all_targets)
    
    return loss_meter.avg, accuracy, cer


def train(
    config_path: str,
    train_data_path: str,
    val_data_path: str,
    stage: str = "pretrain",
    checkpoint: Optional[str] = None,
    output_dir: str = "runs/recognition",
    num_workers: int = 0,
) -> None:
    """
    Main training loop.
    
    Args:
        config_path: Path to configuration YAML.
        train_data_path: Path to training annotations or directory.
        val_data_path: Path to validation annotations or directory.
        stage: Training stage ('pretrain' or 'finetune').
        checkpoint: Path to resume from or finetune from.
        output_dir: Output directory for checkpoints.
    """
    # Load config
    config = load_config(config_path)
    stage_cfg = config.get("training", {}).get(stage, {})
    if not stage_cfg:
        raise ValueError(f"No configuration found for training stage: {stage}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Setup Vocabulary
    vocab_cfg = config.get("vocabulary", {})
    vocab_path = Path(output_dir) / "vocab.json"
    
    if vocab_path.exists():
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load(str(vocab_path))
    else:
        print("Building vocabulary...")
        vocab = Vocabulary.build_japanese_auction_vocab(
            include_hiragana=vocab_cfg.get("include_hiragana", True),
            include_katakana=vocab_cfg.get("include_katakana", True),
            include_kanji=vocab_cfg.get("include_kanji", True),
            include_digits=vocab_cfg.get("include_digits", True),
            include_latin=vocab_cfg.get("include_latin", True),
            include_symbols=vocab_cfg.get("include_symbols", True),
            extra_kanji_path=vocab_cfg.get("kanji_source", None),
        )
        vocab.save(str(vocab_path))
    
    print(f"Vocabulary size: {vocab.size}")
    
    # Build model
    print("Building model...")
    model = build_model_from_config(config["model"], vocab)
    
    start_epoch = 1
    checkpoint_data = None
    if checkpoint:
        print(f"Loading checkpoint weights from: {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        
    model = model.to(device)
    
    # Setup Data
    img_h = config.get("input", {}).get("image_height", 64)
    img_w = config.get("input", {}).get("image_width", 256)
    
    aug_cfg = config.get("augmentation", {})
    augmentor = OCRAugmentor(
        perspective_scale=aug_cfg.get("perspective_distortion", 0.05),
        rotation_range=aug_cfg.get("rotation_range", 3.0),
        noise_var=aug_cfg.get("gaussian_noise_var", 0.01),
        brightness_range=tuple(aug_cfg.get("brightness_range", [0.8, 1.2])),
        contrast_range=tuple(aug_cfg.get("contrast_range", [0.8, 1.2])),
        blur_prob=aug_cfg.get("blur_probability", 0.1),
        blur_kernel_range=tuple(aug_cfg.get("blur_kernel_range", [3, 5])),
        jpeg_quality_range=tuple(aug_cfg.get("jpeg_quality_range", [70, 95])),
        augment_prob=0.8 if stage == "finetune" else 0.5,
    )
    
    print("Loading datasets...")
    train_dataset = OCRDataset(
        data_path=train_data_path,
        vocab=vocab,
        img_height=img_h,
        img_width=img_w,
        augmentor=augmentor,
        is_train=True,
    )
    
    val_dataset = OCRDataset(
        data_path=val_data_path,
        vocab=vocab,
        img_height=img_h,
        img_width=img_w,
        augmentor=None,
        is_train=False,
    )
    
    # Resolve num_workers (CLI overrides config, config overrides default 0)
    cfg_num_workers = config.get("training", {}).get("num_workers", 0)
    final_num_workers = num_workers if num_workers != 0 else cfg_num_workers
    
    batch_size = stage_cfg.get("batch_size", 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=final_num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=final_num_workers,
        pin_memory=True,
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Optimizer & Loss
    learning_rate = stage_cfg.get("learning_rate", 1e-4)
    weight_decay = stage_cfg.get("weight_decay", 1e-5)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    epochs = stage_cfg.get("epochs", 100)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    label_smoothing = stage_cfg.get("label_smoothing", 0.1)
    criterion = LabelSmoothingLoss(
        classes=vocab.size,
        smoothing=label_smoothing,
        ignore_index=vocab.pad_idx,
    ).to(device)
    
    # Restore optimizer/scheduler states if resuming
    if checkpoint_data:
        if "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            print("  Restored optimizer state")
        if "scheduler_state_dict" in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            print("  Restored scheduler state")
        if "epoch" in checkpoint_data:
            start_epoch = checkpoint_data["epoch"] + 1
            print(f"  Resuming from epoch {start_epoch}")
            
    clip_grad = config.get("training", {}).get("gradient_clip", 5.0)
    
    # Initialize AMP GradScaler
    # Enable scaler only when running on CUDA and using float16 (bfloat16 does not need scaling)
    use_amp = (device.type == "cuda")
    is_bf16 = use_amp and torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and not is_bf16))
    
    # Training Loop
    print(f"\nStarting {stage} from epoch {start_epoch} to {epochs}...")
    best_cer = float("inf")
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            vocab=vocab,
            scaler=scaler,
            clip_grad=clip_grad,
        )
        
        val_loss, val_acc, val_cer = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            vocab=vocab,
        )
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc (TF): {train_acc:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f} | Val CER: {val_cer:.4f}")
        
        # Save checkpoints
        save_path = Path(output_dir) / f"model_last.pt"
        extra_info = {
            "epoch": epoch,
            "cer": val_cer,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        model.save_checkpoint(str(save_path), extra_info)
        
        if val_cer < best_cer:
            best_cer = val_cer
            best_path = Path(output_dir) / f"model_best.pt"
            model.save_checkpoint(str(best_path), extra_info)
            print(f"New best model saved! (CER: {best_cer:.4f})")
            
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid OCR Recognition Model")
    parser.add_argument("--config", default="configs/recognition.yaml")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--checkpoint", help="Path to checkpoint to resume/finetune from")
    parser.add_argument("--output", default="runs/recognition")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes for data loader")
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        stage=args.stage,
        checkpoint=args.checkpoint,
        output_dir=args.output,
        num_workers=args.num_workers,
    )
