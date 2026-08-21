"""
Configuration and runtime utility helpers for TruthLens Visual AI package.
"""

import os
from pathlib import Path
import yaml
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent


def load_config(config_path: str = None) -> dict:
    """Loads the model release configuration YAML."""
    if config_path is None:
        config_path = PACKAGE_ROOT / "config" / "model_config.yaml"
    
    config_file = Path(config_path)
    if not config_file.exists():
        return {
            "model": {"name": "truthlens_dinov2_forensic_net", "device": "cuda", "input_size": [252, 252]},
            "checkpoints": {"primary_weights": "checkpoints/truthlens_dinov2_model.pth"},
            "storage": {"heatmap_output_dir": "storage/heatmaps"}
        }
    
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(preferred: str = "cuda") -> torch.device:
    """Selects compute device with automatic CUDA / CPU fallback."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
