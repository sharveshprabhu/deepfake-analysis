"""
Configuration loader and validator for Temporal Deepfake Detection Model.
"""
import os
import yaml
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads and returns the YAML configuration as a dictionary."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    if config_path is None:
        candidates = [
            os.path.join(base_dir, "config", "model_config.yaml"),
            os.path.join(base_dir, "config", "training_config.yaml"),
            "config/model_config.yaml",
            "config/training_config.yaml"
        ]
        resolved_path = None
        for cand in candidates:
            if os.path.exists(cand):
                resolved_path = cand
                break
        if resolved_path is None:
            resolved_path = os.path.join(base_dir, "config", "model_config.yaml")
        config_path = resolved_path
    elif not os.path.isabs(config_path):
        cand = os.path.join(base_dir, config_path)
        if os.path.exists(cand):
            config_path = cand

    if not os.path.exists(config_path):
        # Fallback search if path was relative to workspace root
        if os.path.exists(os.path.join(base_dir, "config", "model_config.yaml")):
            config_path = os.path.join(base_dir, "config", "model_config.yaml")
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_device(requested_device: str = "cuda") -> str:
    """Returns torch device string, safely falling back to cpu if requested cuda is unavailable."""
    import torch
    if requested_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
