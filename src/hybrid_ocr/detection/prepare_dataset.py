"""
Dataset preparation utilities for YOLO detection.

Provides tools to convert PDFs and manual annotations into the
standard YOLO dataset format for fine-tuning the detection model.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def create_yolo_dataset(
    annotations_json: str,
    images_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Convert a generic JSON annotation file into YOLO format.
    
    YOLO format requires:
    dataset/
      images/
        train/
        val/
      labels/
        train/
        val/
    
    Expected input JSON format:
    [
        {
            "image": "path/to/img.png",
            "boxes": [
                {"bbox": [x1, y1, x2, y2], "class_id": 0},
                ...
            ]
        },
        ...
    ]
    
    Args:
        annotations_json: Path to JSON annotation file.
        images_dir: Base directory where images are stored.
        output_dir: Output YOLO dataset directory.
        val_split: Fraction of data to use for validation.
        seed: Random seed for splitting.
    """
    out_dir = Path(output_dir)
    img_dir_in = Path(images_dir)
    
    # Create YOLO directory structure
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Shuffle and split
    np.random.seed(seed)
    np.random.shuffle(data)
    
    split_idx = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Creating YOLO dataset in {output_dir}")
    print(f"  Train: {len(train_data)} images")
    print(f"  Val:   {len(val_data)} images")
    
    def process_split(split_data: List[Dict], split_name: str):
        for item in split_data:
            img_rel_path = item["image"]
            img_path = img_dir_in / img_rel_path
            
            if not img_path.exists():
                print(f"WARNING: Image not found: {img_path}")
                continue
            
            # Read image to get dimensions (needed for YOLO normalized coords)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"WARNING: Could not read image: {img_path}")
                continue
            
            h, w = img.shape[:2]
            
            # Copy image
            img_out_name = f"{img_path.stem}.jpg"  # Convert to JPG to save space
            img_out_path = out_dir / "images" / split_name / img_out_name
            cv2.imwrite(str(img_out_path), img)
            
            # Create label file
            label_out_name = f"{img_path.stem}.txt"
            label_out_path = out_dir / "labels" / split_name / label_out_name
            
            with open(label_out_path, "w", encoding="utf-8") as f:
                for box_info in item.get("boxes", []):
                    cls_id = box_info["class_id"]
                    x1, y1, x2, y2 = box_info["bbox"]
                    
                    # Convert to YOLO format: center_x, center_y, width, height (normalized 0-1)
                    center_x = ((x1 + x2) / 2) / w
                    center_y = ((y1 + y2) / 2) / h
                    box_w = (x2 - x1) / w
                    box_h = (y2 - y1) / h
                    
                    # Ensure within bounds
                    center_x = max(0.0, min(1.0, center_x))
                    center_y = max(0.0, min(1.0, center_y))
                    box_w = max(0.0, min(1.0, box_w))
                    box_h = max(0.0, min(1.0, box_h))
                    
                    f.write(f"{cls_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n")
    
    process_split(train_data, "train")
    process_split(val_data, "val")
    
    print("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("--json", required=True, help="Input annotations JSON")
    parser.add_argument("--images", required=True, help="Input images directory")
    parser.add_argument("--output", required=True, help="Output YOLO dataset directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    create_yolo_dataset(
        args.json,
        args.images,
        args.output,
        args.val_split,
    )
