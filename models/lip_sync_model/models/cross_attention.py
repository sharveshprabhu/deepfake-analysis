"""
Bidirectional Multi-Head Cross-Attention Transformer for Audio-Visual Alignment.
Computes frame-to-phoneme cross-modal attention maps and synchronized representations.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class CrossAttentionLayer(nn.Module):
    """Single bidirectional cross-attention block with residual connections and LayerNorm."""
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.v_to_a_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.a_to_v_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_v1 = nn.LayerNorm(d_model)
        self.norm_a1 = nn.LayerNorm(d_model)

        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm_v2 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)

    def forward(
        self,
        v_feat: torch.Tensor,
        a_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        v_feat: (B, T_v, d_model)
        a_feat: (B, T_a, d_model)
        returns: (v_out, a_out, attn_weights)
        """
        # Visual queries attend to Audio keys/values
        v_cross, attn_weights = self.v_to_a_attn(
            query=v_feat,
            key=a_feat,
            value=a_feat,
            need_weights=True,
            average_attn_weights=True
        ) # attn_weights: (B, T_v, T_a)
        v_mid = self.norm_v1(v_feat + v_cross)
        v_out = self.norm_v2(v_mid + self.ffn_v(v_mid))

        # Audio queries attend to Visual keys/values
        a_cross, _ = self.a_to_v_attn(
            query=a_feat,
            key=v_feat,
            value=v_feat,
            need_weights=False
        )
        a_mid = self.norm_a1(a_feat + a_cross)
        a_out = self.norm_a2(a_mid + self.ffn_a(a_mid))

        return v_out, a_out, attn_weights


class CrossAttentionTransformer(nn.Module):
    """
    Multi-Layer Bidirectional Cross-Attention Transformer.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        v_feat: torch.Tensor,
        a_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        v_feat: (B, T_v, d_model)
        a_feat: (B, T_a, d_model)
        returns: (v_aligned, a_aligned, final_attn_matrix)
        """
        v = self.pos_encoder(v_feat)
        a = self.pos_encoder(a_feat)

        last_attn = None
        for layer in self.layers:
            v, a, last_attn = layer(v, a)

        return v, a, last_attn
