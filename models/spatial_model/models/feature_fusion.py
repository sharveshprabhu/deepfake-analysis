"""
Feature Fusion Module for Person 1 (Spatial AI)
Combines Global Scene Features and Local Artifact Features via Gated Cross-Branch Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureFusion(nn.Module):
    """
    Fuses global scene-level representation with local patch-level representation.
    """
    def __init__(
        self,
        global_dim: int = 1024,
        local_dim: int = 1024,
        embedding_dim: int = 2048,
        fusion_type: str = "gated", # "gated", "concat", or "add"
        dropout: float = 0.2
    ):
        super(MultiScaleFeatureFusion, self).__init__()
        self.fusion_type = fusion_type
        
        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(global_dim + local_dim, global_dim),
            nn.Sigmoid()
        )
        
        # Projection to final unified spatial embedding
        self.proj = nn.Sequential(
            nn.Linear(global_dim + local_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            global_features: (B, global_dim)
            local_features: (B, local_dim)
        Returns:
            spatial_embedding: (B, embedding_dim)
        """
        cat_feat = torch.cat([global_features, local_features], dim=-1)
        
        if self.fusion_type == "gated":
            alpha = self.gate(cat_feat)
            gated_global = alpha * global_features
            gated_local = (1.0 - alpha) * local_features
            fused_input = torch.cat([gated_global, gated_local], dim=-1)
            embedding = self.proj(fused_input)
        else:
            embedding = self.proj(cat_feat)
            
        # L2-normalize spatial embedding
        norm_embedding = F.normalize(embedding, p=2, dim=-1)
        return norm_embedding
