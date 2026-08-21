"""
TruthLens DINOv2 Multi-Modal Forensic Architecture (Release Model).
Combines:
1. DINOv2 (ViT-S/14) Self-Supervised Vision Transformer Patch Backbone
2. Learnable Bayar Constrained High-Pass Noise Residual Stream
3. Spatial Multi-Head Cross-Attention Fusion (RGB queries Noise)
4. Multi-Task Heads: Global Manipulation Classifier + Dense Pixel Mask Decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConstrainedConv2d(nn.Module):
    r"""
    Bayar constrained convolutional layer for learning adaptive forensic high-pass filters.
    Enforces the constraint: sum(weights \ {center}) = 1.0 and weight[center] = -1.0.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.05)
        self.center_idx = kernel_size // 2

    def get_constrained_weight(self) -> torch.Tensor:
        w = self.weight.clone()
        w[:, :, self.center_idx, self.center_idx] = 0.0
        sums = w.sum(dim=(2, 3), keepdim=True) + 1e-7
        w = w / sums
        w[:, :, self.center_idx, self.center_idx] = -1.0
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constrained_w = self.get_constrained_weight()
        return F.conv2d(x, constrained_w, bias=None, stride=1, padding=self.padding)


class BayarNoiseEncoder(nn.Module):
    """
    Encodes sensor noise residuals into a spatial feature map matching DINOv2 patch resolution.
    Input: (B, 1, 252, 252) -> Output: (B, 384, 18, 18)
    """

    def __init__(self, out_dim: int = 384):
        super().__init__()
        self.bayar = BayarConstrainedConv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 252 -> 126
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 126 -> 63
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 63 -> 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((18, 18))
        )

    def forward(self, srm_res: torch.Tensor) -> torch.Tensor:
        bayar_feats = self.bayar(srm_res)
        return self.encoder(bayar_feats)


class SpatialCrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention where Spatial RGB tokens query Noise features.
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, rgb_tokens: torch.Tensor, noise_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm1(rgb_tokens)
        k = self.norm1(noise_tokens)
        v = self.norm1(noise_tokens)
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        x = rgb_tokens + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DenseMaskDecoder(nn.Module):
    """
    Decodes 18x18 fused spatial feature tokens into a dense 252x252 pixel tampering mask.
    """

    def __init__(self, in_dim: int = 384):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 192, kernel_size=4, stride=2, padding=1),  # 18 -> 36
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),   # 36 -> 72
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),    # 72 -> 144
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(252, 252), mode="bilinear", align_corners=False), # 144 -> 252
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, kernel_size=1)
        )

    def forward(self, fused_map: torch.Tensor) -> torch.Tensor:
        return self.decoder(fused_map)


class TruthLensDinov2Net(nn.Module):
    """
    Master Forensic Model integrating DINOv2 + Bayar Noise + Cross-Attention + Multi-Task Heads.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        else:
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)

        self.embed_dim = 384
        self.noise_encoder = BayarNoiseEncoder(out_dim=self.embed_dim)
        self.cross_attn = SpatialCrossAttentionFusion(embed_dim=self.embed_dim, num_heads=6)
        self.mask_decoder = DenseMaskDecoder(in_dim=self.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, rgb_img: torch.Tensor, srm_res: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.dinov2.forward_features(rgb_img)
        cls_token = features["x_norm_clstoken"]        # (B, 384)
        patch_tokens = features["x_norm_patchtokens"]  # (B, 324, 384)

        noise_map = self.noise_encoder(srm_res)        # (B, 384, 18, 18)
        B, C, H, W = noise_map.shape
        noise_tokens = noise_map.flatten(2).permute(0, 2, 1)

        fused_tokens = self.cross_attn(patch_tokens, noise_tokens)
        fused_map = fused_tokens.permute(0, 2, 1).reshape(B, C, H, W)

        mask_logits = self.mask_decoder(fused_map)  # (B, 1, 252, 252)

        spatial_pooled = fused_tokens.mean(dim=1)
        global_repr = torch.cat([cls_token, spatial_pooled], dim=1)
        image_logit = self.classifier(global_repr)  # (B, 1)

        return image_logit, mask_logits
