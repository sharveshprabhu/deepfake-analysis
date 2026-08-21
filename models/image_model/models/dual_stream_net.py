"""
TruthLens Dual-Stream Neural Network Backbone.
Combines Deep Spatial RGB Features (ResNet18) + Spatial Rich Model (SRM) Noise Residual Stream
for robust image manipulation and deepfake detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TruthLensDualStreamNet(nn.Module):
    """
    Dual-stream neural network combining Deep RGB Feature Backbone + SRM Noise Residual Stream.
    """

    def __init__(self):
        super().__init__()
        # RGB Backbone (ResNet18)
        self.rgb_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        rgb_feat_dim = self.rgb_backbone.fc.in_features  # 512
        self.rgb_backbone.fc = nn.Identity()

        # SRM Noise Residual Stream (Shallow CNN)
        self.srm_stream = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )  # 64-dim

        # Fusion Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feat_dim + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Logit
        )

    def forward(self, rgb_img: torch.Tensor, srm_res: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgb_img: (B, 3, 256, 256) normalized RGB image tensor
            srm_res: (B, 1, 256, 256) normalized SRM noise residual tensor
            
        Returns:
            logit: (B, 1) unnormalized manipulation logit
        """
        feat_rgb = self.rgb_backbone(rgb_img)  # (B, 512)
        feat_srm = self.srm_stream(srm_res)    # (B, 64)
        fused = torch.cat([feat_rgb, feat_srm], dim=1)  # (B, 576)
        logit = self.classifier(fused)
        return logit
