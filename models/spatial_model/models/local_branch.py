"""
Local Spatial Branch for Person 1 (Spatial AI)
Extracts fine-grained manipulation artifacts, texture inconsistencies, blending boundaries,
and patch-level irregularities across general regions (NOT face-only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LocalSpatialBranch(nn.Module):
    """
    Extracts multi-patch fine-grained spatial representations.
    Operates on general spatial regions across the entire image.
    """
    def __init__(self, in_features: int = 1792, out_dim: int = 1024, num_patches: int = 4, dropout: float = 0.2):
        super(LocalSpatialBranch, self).__init__()
        self.num_patches = num_patches
        
        # High-frequency / Boundary artifact extractor (1x1 and 3x3 conv)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 2, in_features // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features // 4),
            nn.ReLU(inplace=True)
        )
        
        # Multi-region grid pooling: 2x2 grid = 4 spatial quadrants
        self.grid_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Patch aggregation
        patch_dim = (in_features // 4) * 4
        self.projection = nn.Sequential(
            nn.Linear(patch_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Input: feature_map (B, C, H, W)
        Output: local_features (B, out_dim)
        """
        # Extract fine spatial details
        local_fmap = self.local_conv(feature_map) # (B, C//4, H, W)
        
        # 2x2 regional grid pooling across quadrants
        grid_features = self.grid_pool(local_fmap) # (B, C//4, 2, 2)
        grid_flat = grid_features.flatten(1) # (B, (C//4)*4)
        
        local_features = self.projection(grid_flat) # (B, out_dim)
        return local_features
