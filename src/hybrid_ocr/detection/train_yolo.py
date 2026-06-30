"""
YOLO detection fine-tuning script.

Fine-tunes a pretrained YOLOv8 model on custom auction document data
using ultralytics API.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def train_yolo(
    config_path: str,
    pretrained_weights: str = "yolov8s.pt",
    project_name: str = "runs/detection",
    experiment_name: str = "train",
) -> None:
    """
    Fine-tune YOLO model.
    
    Args:
        config_path: Path to dataset YAML configuration (e.g., configs/yolo_detection.yaml).
        pretrained_weights: Path to starting weights.
        project_name: Directory for saving runs.
        experiment_name: Name of this specific training run.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for training: pip install ultralytics")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Starting YOLO fine-tuning...")
    print(f"  Config:  {config_path}")
    print(f"  Weights: {pretrained_weights}")
    print(f"  Output:  {project_name}/{experiment_name}")
    
    # Load model
    model = YOLO(pretrained_weights)
    
    # Start training
    # The training hyperparameters are largely defined in the YAML config,
    # but we pass project/name to override the output destination.
    results = model.train(
        data=config_path,
        project=project_name,
        name=experiment_name,
        # Allow passing through config parameters that might not be in YAML
        plots=True,
        save=True,
    )
    
    print(f"\nTraining completed!")
    print(f"Best weights saved to: {project_name}/{experiment_name}/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for document text detection")
    parser.add_argument("--config", default="configs/yolo_detection.yaml", help="Path to config YAML")
    parser.add_argument("--weights", default="yolov8s.pt", help="Pretrained weights to start from")
    parser.add_argument("--project", default="runs/detection", help="Output project directory")
    parser.add_argument("--name", default="auction_detector", help="Experiment name")
    
    args = parser.parse_args()
    
    train_yolo(
        args.config,
        args.weights,
        args.project,
        args.name,
    )
