"""
3D-ResNet Lip Viseme Visual Encoder with Squeeze-and-Excitation (SE) Attention.
Captures continuous spatiotemporal dynamics, mouth contour velocities, and micro-visemes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.GELU(),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResidualBlock2D(nn.Module):
    """2D Residual block with BatchNorm and SE Attention."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.gelu(out + res)
        return out


class VisualLipEncoder(nn.Module):
    """
    Spatiotemporal Visual Encoder:
    3D-Conv Stem (preserving temporal resolution) -> Pretrained 2D ResNet-18 SE Backbone -> Projection.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # 3D Stem: processes (B, C=3, T=5, H=96, W=96)
        self.stem3d = nn.Sequential(
            nn.Conv3d(
                in_channels,
                64,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
                bias=False
            ),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ) # Output: (B, 64, T=5, 24, 24)

        # 2D ResNet Backbone applied per-frame
        self.layer1 = nn.Sequential(
            ResidualBlock2D(64, 64, stride=1),
            ResidualBlock2D(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock2D(64, 128, stride=2),
            ResidualBlock2D(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock2D(128, 256, stride=2),
            ResidualBlock2D(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock2D(256, 512, stride=2),
            ResidualBlock2D(512, 512, stride=1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Initializes spatial 2D ResNet layers with pretrained ImageNet weights."""
        try:
            import torchvision.models as tv_models
            res = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            for l_name in ["layer1", "layer2", "layer3", "layer4"]:
                src_layer = getattr(res, l_name)
                dst_layer = getattr(self, l_name)
                for b_idx in range(min(len(src_layer), len(dst_layer))):
                    s_b = src_layer[b_idx]
                    d_b = dst_layer[b_idx]
                    d_b.conv1.weight.data.copy_(s_b.conv1.weight.data)
                    d_b.bn1.weight.data.copy_(s_b.bn1.weight.data)
                    d_b.bn1.bias.data.copy_(s_b.bn1.bias.data)
                    d_b.conv2.weight.data.copy_(s_b.conv2.weight.data)
                    d_b.bn2.weight.data.copy_(s_b.bn2.weight.data)
                    d_b.bn2.bias.data.copy_(s_b.bn2.bias.data)
                    if s_b.downsample is not None and len(d_b.shortcut) > 0:
                        d_b.shortcut[0].weight.data.copy_(s_b.downsample[0].weight.data)
                        d_b.shortcut[1].weight.data.copy_(s_b.downsample[1].weight.data)
                        d_b.shortcut[1].bias.data.copy_(s_b.downsample[1].bias.data)
            print("[+] Loaded pretrained ImageNet ResNet-18 weights into Visual Lip Encoder.")
        except Exception as e:
            print(f"[!] Warning: Could not load pretrained visual weights: {e}")

    def forward(self, lip_frames: torch.Tensor) -> torch.Tensor:
        """
        lip_frames: (B, T=5, C=3, H=96, W=96)
        returns: (B, T=5, feature_dim=256) temporal visual embedding sequence
        """
        B, T, C, H, W = lip_frames.shape

        # Permute to (B, C, T, H, W) for Conv3D stem
        x = lip_frames.permute(0, 2, 1, 3, 4)
        x = self.stem3d(x) # (B, 64, T, H', W')

        # Reshape to (B * T, 64, H', W') for 2D ResNet processing
        _, C_stem, T_out, H_stem, W_stem = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T_out, C_stem, H_stem, W_stem)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B * T, 512, H'', W'')

        x = self.global_pool(x).view(B * T_out, -1) # (B * T, 512)
        x = self.proj(x)                            # (B * T, feature_dim)

        # Restore temporal dimension (B, T, feature_dim)
        x = x.view(B, T_out, -1)
        return x
