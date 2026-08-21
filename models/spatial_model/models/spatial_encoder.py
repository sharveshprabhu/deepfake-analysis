"""
Whole-Frame Spatial Encoder for Person 1 (Spatial AI)
Extracts multi-scale spatial representations (global scene context + fine local artifacts).
Exposes standard interface for Person 2 (Image Classification + Video Temporal Analysis).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

try:
    from .backbone import VisionBackbone
    from .global_branch import GlobalSpatialBranch
    from .local_branch import LocalSpatialBranch
    from .feature_fusion import MultiScaleFeatureFusion
except ImportError:
    from backbone import VisionBackbone
    from global_branch import GlobalSpatialBranch
    from local_branch import LocalSpatialBranch
    from feature_fusion import MultiScaleFeatureFusion


class SpatialEncoder(nn.Module):
    """
    Person 1 Spatial Encoder.
    Outputs rich whole-frame spatial representations:
      - global_features (scene structure, lighting, geometry)
      - local_features (fine artifacts, textures, blending boundaries)
      - spatial_embedding (2048-dim fused normalized embedding)
      - feature_map (spatial feature grid)
      - attention_map (spatial saliency heatmap)
    """
    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        freeze_stages: int = 2,
        global_dim: int = 1024,
        local_dim: int = 1024,
        embedding_dim: int = 2048,
        mode: str = "full", # "full", "global_only", "local_only"
        include_aux_head: bool = True
    ):
        super(SpatialEncoder, self).__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim
        
        # 1. Vision Backbone
        self.backbone = VisionBackbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages
        )
        in_features = self.backbone.feature_dim
        
        # 2. Global Branch
        self.global_branch = GlobalSpatialBranch(
            in_features=in_features,
            out_dim=global_dim
        )
        
        # 3. Local Branch
        self.local_branch = LocalSpatialBranch(
            in_features=in_features,
            out_dim=local_dim
        )
        
        # 4. Multi-Scale Feature Fusion
        self.fusion = MultiScaleFeatureFusion(
            global_dim=global_dim,
            local_dim=local_dim,
            embedding_dim=embedding_dim,
            fusion_type="gated"
        )
        
        # 5. Optional Auxiliary Classification Head (for representation training only)
        if include_aux_head:
            head_in_dim = global_dim if mode == "global_only" else (local_dim if mode == "local_only" else embedding_dim)
            self.aux_classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(head_in_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.15),
                nn.Linear(512, 1)
            )
        else:
            self.aux_classifier = None

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Processes image tensor and returns comprehensive spatial features.
        Input: image (B, 3, H, W)
        """
        # Backbone feature extraction
        feature_map, pooled = self.backbone(image)
        
        # Global scene branch
        global_feat, att_map = self.global_branch(feature_map, pooled)
        
        # Local artifact branch
        local_feat = self.local_branch(feature_map)
        
        # Branch fusion based on ablation mode
        if self.mode == "global_only":
            spatial_emb = nn.functional.normalize(global_feat, p=2, dim=-1)
        elif self.mode == "local_only":
            spatial_emb = nn.functional.normalize(local_feat, p=2, dim=-1)
        else: # "full"
            spatial_emb = self.fusion(global_feat, local_feat)
            
        outputs = {
            "global_features": global_feat,
            "local_features": local_feat,
            "spatial_embedding": spatial_emb,
            "feature_map": feature_map,
            "attention_map": att_map
        }
        
        if self.aux_classifier is not None:
            outputs["logits"] = self.aux_classifier(spatial_emb)
            outputs["prob"] = torch.sigmoid(outputs["logits"])
            
        return outputs

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Pure feature extraction interface for Person 2 (Temporal & Image Models).
        """
        self.eval()
        out = self.forward(image)
        return {
            "global_features": out["global_features"],
            "local_features": out["local_features"],
            "spatial_embedding": out["spatial_embedding"],
            "feature_map": out["feature_map"],
            "attention_map": out["attention_map"]
        }
