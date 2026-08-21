"""
Temporal Deepfake Detection Architectures.
Provides:
1. TemporalTransformerModel: Multi-head temporal self-attention with transition discontinuity analyzer
2. TemporalBiLSTMModel: Bidirectional LSTM with temporal attention pooling
3. Inter-frame anomaly and suspicious timestamp localizer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional
from models.visual_encoder import FrameVisualEncoder

class TemporalTransformerModel(nn.Module):
    """
    Temporal Transformer that models long-range frame-to-frame consistency
    and inter-frame artifact flicker.
    """
    def __init__(
        self,
        visual_encoder: FrameVisualEncoder,
        feature_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        max_seq_len: int = 64
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.feature_dim = feature_dim
        
        # Learnable temporal CLS token for video-level aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
        # Learnable temporal positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, feature_dim) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Inter-frame discontinuity transition scorer
        self.transition_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Video-level classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) # Binary logit: Real vs Fake
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: Video tensor of shape (B, T, C, H, W)
        returns: {
            "logits": (B,),
            "probabilities": (B,),
            "transition_scores": (B, T-1),
            "frame_features": (B, T, D)
        }
        """
        B, T, C, H, W = x.shape
        
        # 1. Extract visual spatial embeddings
        frame_features = self.visual_encoder(x) # (B, T, D)
        
        # 2. Append CLS token and add temporal positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        tokens = torch.cat([cls_tokens, frame_features], dim=1) # (B, T+1, D)
        tokens = tokens + self.pos_embedding[:, :T + 1, :]
        
        # 3. Temporal Transformer Pass
        transformed = self.transformer(tokens) # (B, T+1, D)
        cls_out = transformed[:, 0, :]         # (B, D)
        seq_out = transformed[:, 1:, :]        # (B, T, D)
        
        # 4. Compute Inter-Frame Transition Discontinuity Scores
        # Compare adjacent temporal tokens (t, t+1)
        f_left = seq_out[:, :-1, :] # (B, T-1, D)
        f_right = seq_out[:, 1:, :] # (B, T-1, D)
        diff_pairs = torch.cat([f_left, f_right], dim=-1) # (B, T-1, 2*D)
        transition_scores = self.transition_scorer(diff_pairs).squeeze(-1) # (B, T-1)
        
        # 5. Video-level Logits and Probabilities
        logits = self.classifier(cls_out).squeeze(-1) # (B,)
        probs = torch.sigmoid(logits)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "transition_scores": transition_scores,
            "frame_features": seq_out
        }

class TemporalBiLSTMModel(nn.Module):
    """
    Temporal Bi-LSTM Architecture with attention pooling across frames.
    """
    def __init__(
        self,
        visual_encoder: FrameVisualEncoder,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Temporal Attention Pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Inter-frame transition discontinuity scorer
        self.transition_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Video classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = x.shape
        frame_features = self.visual_encoder(x) # (B, T, D)
        
        lstm_out, _ = self.lstm(frame_features) # (B, T, 2*hidden_dim)
        
        # Attention Pooling
        attn_weights = F.softmax(self.attention_pool(lstm_out), dim=1) # (B, T, 1)
        pooled = torch.sum(attn_weights * lstm_out, dim=1)            # (B, 2*hidden_dim)
        
        # Transition scores
        f_left = lstm_out[:, :-1, :]
        f_right = lstm_out[:, 1:, :]
        diff_pairs = torch.cat([f_left, f_right], dim=-1)
        transition_scores = self.transition_scorer(diff_pairs).squeeze(-1)
        
        logits = self.classifier(pooled).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "transition_scores": transition_scores,
            "frame_features": lstm_out
        }

def build_temporal_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function to build configured temporal model."""
    m_cfg = config.get("model", {})
    backbone_name = m_cfg.get("visual_backbone", m_cfg.get("backbone", "swin_tiny"))
    feature_dim = m_cfg.get("backbone_feature_dim", m_cfg.get("feature_dim", 768))
    freeze_backbone = m_cfg.get("freeze_backbone_epochs", 0) > 0
    arch = m_cfg.get("temporal_architecture", m_cfg.get("name", "transformer"))
    if "transformer" in str(arch).lower():
        arch = "transformer"
    elif "lstm" in str(arch).lower():
        arch = "bilstm"
    
    visual_enc = FrameVisualEncoder(
        backbone_name=backbone_name,
        embedding_dim=feature_dim,
        pretrained=True,
        freeze_backbone=freeze_backbone
    )
    
    if arch == "transformer":
        model = TemporalTransformerModel(
            visual_encoder=visual_enc,
            feature_dim=feature_dim,
            num_layers=m_cfg.get("temporal_num_layers", m_cfg.get("num_layers", 4)),
            num_heads=m_cfg.get("temporal_num_heads", m_cfg.get("num_heads", 8)),
            dropout=m_cfg.get("temporal_dropout", m_cfg.get("dropout", 0.2))
        )
    elif arch in ["bilstm", "gru"]:
        model = TemporalBiLSTMModel(
            visual_encoder=visual_enc,
            feature_dim=feature_dim,
            hidden_dim=m_cfg.get("temporal_hidden_dim", 512) // 2,
            dropout=m_cfg.get("temporal_dropout", m_cfg.get("dropout", 0.2))
        )
    else:
        raise ValueError(f"Unknown temporal architecture: {arch}")
        
    return model

