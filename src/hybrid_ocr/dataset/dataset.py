"""
PyTorch Dataset for OCR text recognition training.

Provides OCRDataset for loading image-label pairs from directories or
annotation files, with support for augmentation and vocabulary encoding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import OCRAugmentor, letterbox_image
from .vocabulary import Vocabulary


class OCRDataset(Dataset):
    """
    Dataset for OCR text recognition training.
    
    Supports two data formats:
    
    1. **Annotation file** (recommended):
       A JSON file with list of {"image": "path/to/img.png", "label": "テキスト"}
    
    2. **Directory structure**:
       root/
         images/
           img_001.png
           img_002.png
         labels.json  # {"img_001.png": "テキスト", ...}
    
    Each sample is a text crop image paired with its ground truth text label.
    
    Args:
        data_path: Path to annotation JSON file or data directory.
        vocab: Vocabulary instance for encoding labels.
        img_height: Target image height after resize.
        img_width: Target image width after resize.
        max_seq_length: Maximum sequence length (labels longer than this are skipped).
        augmentor: Optional augmentation pipeline.
        is_train: Whether this is a training dataset (enables augmentation).
    """

    def __init__(
        self,
        data_path: str,
        vocab: Vocabulary,
        img_height: int = 64,
        img_width: int = 256,
        max_seq_length: int = 100,
        augmentor: Optional[OCRAugmentor] = None,
        is_train: bool = True,
    ):
        self.vocab = vocab
        self.img_height = img_height
        self.img_width = img_width
        self.max_seq_length = max_seq_length
        self.augmentor = augmentor if is_train else None
        self.is_train = is_train
        
        # ImageNet normalization (for pretrained ConvNeXt)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Load data samples
        self.samples = self._load_samples(data_path)

    def _load_samples(self, data_path: str) -> List[Dict[str, str]]:
        """Load samples from annotation file or directory."""
        path = Path(data_path)
        samples = []
        
        if path.is_file() and path.suffix == ".json":
            # Format 1: Annotation JSON file
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            base_dir = path.parent
            for item in data:
                img_path = str(base_dir / item["image"])
                label = item["label"]
                # Skip samples with labels too long for the model
                if len(label) <= self.max_seq_length - 2:  # -2 for SOS/EOS
                    samples.append({"image": img_path, "label": label})
        
        elif path.is_dir():
            # Format 2: Directory with images/ and labels.json
            labels_file = path / "labels.json"
            if labels_file.exists():
                with open(labels_file, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                
                images_dir = path / "images"
                if not images_dir.exists():
                    images_dir = path
                
                for img_name, label in labels.items():
                    img_path = str(images_dir / img_name)
                    if len(label) <= self.max_seq_length - 2:
                        samples.append({"image": img_path, "label": label})
        else:
            raise ValueError(
                f"data_path must be a JSON file or directory, got: {data_path}"
            )
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary with:
                - "image": (3, H, W) float tensor, normalized
                - "target_input": (seq_len,) long tensor, decoder input (SOS + label)
                - "target_output": (seq_len,) long tensor, decoder target (label + EOS)
                - "target_length": scalar, length of target sequence
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample["image"])
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Augmentation (only during training)
        if self.augmentor is not None:
            image = self.augmentor(image)
        
        # Resize with letterboxing (preserve aspect ratio)
        image, _, _ = letterbox_image(
            image, self.img_height, self.img_width, pad_value=0
        )
        
        # Normalize to [0, 1], then apply ImageNet normalization
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Convert to tensor (H, W, C) → (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Encode label
        label = sample["label"]
        # Decoder input: [SOS, char1, char2, ...] (teacher forcing input)
        target_input = self.vocab.encode(label, add_sos=True, add_eos=False)
        # Decoder target: [char1, char2, ..., EOS] (what the decoder should predict)
        target_output = self.vocab.encode(label, add_sos=False, add_eos=True)
        
        target_input = torch.tensor(target_input, dtype=torch.long)
        target_output = torch.tensor(target_output, dtype=torch.long)
        target_length = torch.tensor(len(target_output), dtype=torch.long)
        
        return {
            "image": image_tensor,
            "target_input": target_input,
            "target_output": target_output,
            "target_length": target_length,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    Pads target sequences to the maximum length in the batch.
    
    Args:
        batch: List of sample dictionaries from OCRDataset.__getitem__.
    
    Returns:
        Batched dictionary with padded tensors.
    """
    images = torch.stack([item["image"] for item in batch])
    
    # Find max sequence length in this batch
    max_len = max(item["target_input"].size(0) for item in batch)
    
    # Pad sequences
    batch_size = len(batch)
    target_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_outputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_lengths = torch.stack([item["target_length"] for item in batch])
    
    for i, item in enumerate(batch):
        length = item["target_input"].size(0)
        target_inputs[i, :length] = item["target_input"]
        target_outputs[i, :length] = item["target_output"]
    
    return {
        "image": images,
        "target_input": target_inputs,
        "target_output": target_outputs,
        "target_length": target_lengths,
    }
