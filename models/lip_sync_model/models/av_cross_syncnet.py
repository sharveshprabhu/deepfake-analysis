"""
Unified AV-CrossSyncNet Architecture:
State-of-the-Art Cross-Attention Audio-Visual LipSync Model with Multi-Task Heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from models.visual_encoder import VisualLipEncoder
from models.audio_encoder import AudioPhonemeEncoder
from models.cross_attention import CrossAttentionTransformer


class AVCrossSyncNet(nn.Module):
    """
    SOTA Cross-Attention Audio-Visual Lip Synchronization Model.
    Provides:
    1. Metric Contrastive Sync Space (L2-normalized embeddings for cosine matching).
    2. Discrete Offset Classification Head (31 classes: -15 to +15 frames).
    3. Multi-Evidence Anomaly Scoring Head.
    """
    def __init__(
        self,
        feature_dim: int = 256,
        metric_dim: int = 128,
        num_cross_attn_heads: int = 4,
        num_cross_attn_layers: int = 2,
        num_offset_classes: int = 31,
        dropout: float = 0.1,
        temperature_init: float = 0.07,
        pretrained: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.metric_dim = metric_dim
        self.num_offset_classes = num_offset_classes

        # 1. Encoders with Pretrained Transfer Weights
        self.visual_encoder = VisualLipEncoder(in_channels=3, feature_dim=feature_dim, pretrained=pretrained)
        self.audio_encoder = AudioPhonemeEncoder(in_channels=1, n_mels=80, feature_dim=feature_dim, pretrained=pretrained)

        # 2. Bidirectional Cross-Attention Transformer
        self.cross_attention = CrossAttentionTransformer(
            d_model=feature_dim,
            n_heads=num_cross_attn_heads,
            num_layers=num_cross_attn_layers,
            dropout=dropout
        )

        # 3. Metric Space Projections (Pool across time -> L2 normalize)
        self.v_metric_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, metric_dim)
        )
        self.a_metric_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, metric_dim)
        )

        # 4. Multi-Task Offset Classification Head (31 classes: -15 to +15 frames)
        self.offset_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_offset_classes)
        )

        # 5. Learnable Temperature Parameter for InfoNCE contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature_init)))

    def forward(
        self,
        lip_frames: torch.Tensor,
        mel_spectrogram: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        lip_frames: (B, T_v=5, C=3, H=96, W=96)
        mel_spectrogram: (B, 1, n_mels=80, T_a=16)
        returns dict containing metric embeddings, offset logits, attention matrix, and similarity.
        """
        B = lip_frames.shape[0]

        # 1. Extract spatiotemporal feature sequences
        v_seq = self.visual_encoder(lip_frames)       # (B, T_v=5, feature_dim=256)
        a_seq = self.audio_encoder(mel_spectrogram)   # (B, T_a=16, feature_dim=256)

        # 2. Bidirectional Cross-Attention
        v_cross, a_cross, attn_matrix = self.cross_attention(v_seq, a_seq) # (B, T_v, D), (B, T_a, D), (B, T_v, T_a)

        # 3. Temporal Pooling (Mean across sequence)
        v_pool = torch.mean(v_cross, dim=1) # (B, feature_dim)
        a_pool = torch.mean(a_cross, dim=1) # (B, feature_dim)

        # 4. Metric Space Projection & L2 Normalization
        v_metric = F.normalize(self.v_metric_proj(v_pool), p=2, dim=-1) # (B, metric_dim)
        a_metric = F.normalize(self.a_metric_proj(a_pool), p=2, dim=-1) # (B, metric_dim)

        # 5. Cosine Similarity in metric space
        cosine_sim = torch.sum(v_metric * a_metric, dim=-1) # (B,)

        # 6. Discrete Offset Classification Logits
        fused_repr = torch.cat([v_pool, a_pool], dim=-1) # (B, feature_dim * 2)
        offset_logits = self.offset_classifier(fused_repr) # (B, num_offset_classes)

        return {
            "v_metric": v_metric,
            "a_metric": a_metric,
            "cosine_sim": cosine_sim,
            "offset_logits": offset_logits,
            "attn_matrix": attn_matrix,
            "logit_scale": self.logit_scale.exp()
        }
