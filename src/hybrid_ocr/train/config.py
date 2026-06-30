"""
Configuration parsing for training scripts.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
    
    Returns:
        Dictionary containing configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config or {}
