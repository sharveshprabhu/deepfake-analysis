"""
2D ResNet-18 Audio Phoneme Encoder with Squeeze-and-Excitation (SE) Attention.
Extracts spectral formant transitions and acoustic speech dynamics from Log-Mel Spectrograms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from models.visual_encoder import ResidualBlock2D, SEBlock


class AudioPhonemeEncoder(nn.Module):
    """
    Encodes 2D Log-Mel Spectrogram (B, 1, n_mels=80, T_audio=16) into
    temporal phoneme embedding sequence (B, T_audio=16, feature_dim=256).
    """
    def __init__(self, in_channels: int = 1, n_mels: int = 80, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # Initial stem: downsamples frequency dimension, preserves time dimension
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        ) # Output: (B, 64, n_mels=20, T=16)

        # 2D ResNet layers
        self.layer1 = nn.Sequential(
            ResidualBlock2D(64, 64, stride=1),
            ResidualBlock2D(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock2D(64, 128, stride=(2, 1)), # Freq: 20 -> 10, Time: 16
            ResidualBlock2D(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock2D(128, 256, stride=(2, 1)), # Freq: 10 -> 5, Time: 16
            ResidualBlock2D(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock2D(256, 512, stride=(2, 1)), # Freq: 5 -> 3, Time: 16
            ResidualBlock2D(512, 512, stride=1)
        )

        # Pool over remaining frequency bins while preserving all T_audio frames
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) # (B, 512, 1, T_audio)

        self.proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Initializes ResNet layers with pretrained weights."""
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
            print("[+] Loaded pretrained ResNet-18 weights into Audio Phoneme Encoder.")
        except Exception as e:
            print(f"[!] Warning: Could not load pretrained audio weights: {e}")

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        mel_spec: (B, 1, n_mels=80, T_audio=16)
        returns: (B, T_audio=16, feature_dim=256)
        """
        x = self.stem(mel_spec) # (B, 64, 20, 16)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)      # (B, 512, 3, 16)

        x = self.freq_pool(x).squeeze(2) # (B, 512, T_audio)
        x = x.permute(0, 2, 1)           # (B, T_audio, 512)
        x = self.proj(x)                 # (B, T_audio, feature_dim)
        return x
