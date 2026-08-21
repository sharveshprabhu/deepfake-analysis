"""
Visual Spatial Frame Backbone Feature Extractor.
Supports Swin-T / ViT / CNN backbones with temporal batch folding (B*T, C, H, W) -> (B, T, D).
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple

class FrameVisualEncoder(nn.Module):
    """
    Extracts spatial embeddings for each frame in a video sequence.
    Folds (B, T, C, H, W) -> (B*T, C, H, W) through backbone, then unfolds to (B, T, D).
    """
    def __init__(
        self,
        backbone_name: str = "swin_tiny",
        embedding_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        
        if backbone_name == "swin_tiny":
            try:
                weights = models.Swin_T_Weights.DEFAULT if pretrained else None
                base_model = models.swin_t(weights=weights)
            except Exception:
                base_model = models.swin_t(weights=None)
            in_features = base_model.head.in_features
            base_model.head = nn.Identity()
            self.backbone = base_model
        elif backbone_name == "vit_base":
            try:
                weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
                base_model = models.vit_b_16(weights=weights)
            except Exception:
                base_model = models.vit_b_16(weights=None)
            in_features = base_model.heads.head.in_features
            base_model.heads.head = nn.Identity()
            self.backbone = base_model
        elif backbone_name == "efficientnet_b4":
            try:
                weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
                base_model = models.efficientnet_b4(weights=weights)
            except Exception:
                base_model = models.efficientnet_b4(weights=None)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            try:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                base_model = models.resnet50(weights=weights)
            except Exception:
                base_model = models.resnet50(weights=None)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model

        # Projection head to target temporal feature dimension
        self.proj = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        if freeze_backbone:
            self.freeze_backbone_layers()

    def freeze_backbone_layers(self):
        """Freezes all backbone feature extraction parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, num_layers: int = 2):
        """Selectively unfreezes top N layers for fine-tuning without memorization."""
        # Unfreeze projection head
        for param in self.proj.parameters():
            param.requires_grad = True
            
        # Unfreeze top layers in backbone
        params = list(self.backbone.parameters())
        for p in params[-num_layers * 10:]:
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns: (B, T, D)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        feat_flat = self.backbone(x_flat) # (B*T, in_features)
        emb_flat = self.proj(feat_flat)   # (B*T, D)
        
        return emb_flat.view(B, T, self.embedding_dim)
