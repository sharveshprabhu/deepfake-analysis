"""
Global Spatial Branch for Person 1 (Spatial AI)
Captures global scene structure, object relationships, geometry, lighting patterns, and shadows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GlobalSpatialBranch(nn.Module):
    """
    Processes full-frame spatial representations to capture scene-wide consistency.
    """
    def __init__(self, in_features: int = 1792, out_dim: int = 1024, dropout: float = 0.2):
        super(GlobalSpatialBranch, self).__init__()
        
        # Spatial attention over full scene feature map
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_features, in_features // 4, kernel_size=1),
            nn.BatchNorm2d(in_features // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global projection
        self.projection = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, feature_map: torch.Tensor, pooled_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            feature_map: (B, C, H, W)
            pooled_features: (B, C)
        Returns:
            global_features: (B, out_dim)
            attention_map: (B, 1, H, W)
        """
        att_map = self.spatial_att(feature_map) # (B, 1, H, W)
        weighted_fmap = feature_map * att_map
        weighted_pool = F.adaptive_avg_pool2d(weighted_fmap, (1, 1)).flatten(1)
        
        combined = weighted_pool + pooled_features
        global_features = self.projection(combined)
        
        return global_features, att_map
