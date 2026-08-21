"""
Backbone Module for Person 1 (Spatial AI)
Provides pretrained vision feature extractors with partial freezing support.
Supported: ConvNeXt-Tiny, EfficientNet-B4, ResNet-50.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class VisionBackbone(nn.Module):
    """
    Wraps standard vision backbones to expose feature maps and feature vectors.
    """
    def __init__(
        self,
        name: str = "efficientnet_b4",
        pretrained: bool = True,
        freeze_stages: int = 2
    ):
        super(VisionBackbone, self).__init__()
        self.name = name.lower()
        
        if "convnext" in self.name:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            m = models.convnext_tiny(weights=weights)
            self.features = m.features
            self.feature_dim = 768
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif "resnet" in self.name:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            m = models.resnet50(weights=weights)
            self.features = nn.Sequential(
                m.conv1, m.bn1, m.relu, m.maxpool,
                m.layer1, m.layer2, m.layer3, m.layer4
            )
            self.feature_dim = 2048
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else: # Default: EfficientNet-B4
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            m = models.efficientnet_b4(weights=weights)
            self.features = m.features
            self.feature_dim = 1792
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Partial freezing
        if freeze_stages > 0:
            self._freeze_initial_layers(freeze_stages)

    def _freeze_initial_layers(self, stages: int):
        child_layers = list(self.features.children())
        freeze_count = min(stages, len(child_layers))
        for layer in child_layers[:freeze_count]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_top_stages(self, stages: int = 4):
        """Unfreezes top/deeper feature stages while keeping shallow initial layers frozen."""
        child_layers = list(self.features.children())
        unfreeze_start = max(0, len(child_layers) - stages)
        for layer in child_layers[unfreeze_start:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feature_map: (B, C, H', W')
            pooled_vector: (B, C)
        """
        fmap = self.features(x)
        pooled = self.pool(fmap).flatten(1)
        return fmap, pooled
