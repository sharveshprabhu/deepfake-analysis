"""
Audio-Visual Synchronization & Forensic Discrepancy Model.
Analyzes speech phoneme audio representations vs. lip motion viseme features
to detect audio-visual temporal desynchronization and synthetic dubbing artifacts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

class AudioSpectrogramEncoder(nn.Module):
    """Encodes 2D Log-Mel Spectrogram into temporal audio embeddings."""
    def __init__(self, n_mels: int = 80, feature_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((None, 1)) # (B, 128, T_audio, 1)
        )
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        mel_spec: (B, 1, n_mels, T_audio_frames)
        returns: (B, T_audio_frames, feature_dim)
        """
        out = self.conv(mel_spec) # (B, 128, T_out, 1)
        out = out.squeeze(-1).permute(0, 2, 1) # (B, T_out, 128)
        return self.proj(out)

class LipMotionEncoder(nn.Module):
    """Encodes mouth region visual frame sequence into temporal viseme embeddings."""
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)) # (B, 64, T, 1, 1)
        )
        self.proj = nn.Linear(64, feature_dim)

    def forward(self, lip_frames: torch.Tensor) -> torch.Tensor:
        """
        lip_frames: (B, T, C, H, W)
        returns: (B, T, feature_dim)
        """
        # Permute to (B, C, T, H, W) for Conv3D
        x = lip_frames.permute(0, 2, 1, 3, 4)
        out = self.conv3d(x) # (B, 64, T, 1, 1)
        out = out.squeeze(-1).squeeze(-1).permute(0, 2, 1) # (B, T, 64)
        return self.proj(out)

class AVSyncModel(nn.Module):
    """
    Audio-Visual Cross-Modal Temporal Alignment & Sync Scorer.
    """
    def __init__(
        self,
        audio_dim: int = 256,
        lip_dim: int = 256,
        sync_embedding_dim: int = 128
    ):
        super().__init__()
        self.audio_encoder = AudioSpectrogramEncoder(feature_dim=audio_dim)
        self.lip_encoder = LipMotionEncoder(feature_dim=lip_dim)
        
        self.audio_proj = nn.Linear(audio_dim, sync_embedding_dim)
        self.lip_proj = nn.Linear(lip_dim, sync_embedding_dim)
        
        # Sync anomaly classifier
        self.sync_classifier = nn.Sequential(
            nn.Linear(sync_embedding_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        lip_frames: torch.Tensor,
        mel_spectrogram: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        lip_frames: (B, T, C, H, W)
        mel_spectrogram: (B, 1, n_mels, T_audio) or None (silent video)
        """
        B, T, C, H, W = lip_frames.shape
        
        # If no audio track is present
        if mel_spectrogram is None:
            return {
                "has_audio": False,
                "audio_score": None,
                "av_sync_score": 0.0,
                "av_sync_offset_ms": 0.0,
                "explanations": ["Video stream contains no audio track (silent)"]
            }

        audio_emb = self.audio_encoder(mel_spectrogram) # (B, T_a, D_a)
        lip_emb = self.lip_encoder(lip_frames)          # (B, T_v, D_v)
        
        # Project into shared metric space
        a_norm = F.normalize(self.audio_proj(audio_emb), p=2, dim=-1)
        v_norm = F.normalize(self.lip_proj(lip_emb), p=2, dim=-1)
        
        # Temporal pooling / matching
        a_pool = torch.mean(a_norm, dim=1) # (B, D_sync)
        v_pool = torch.mean(v_norm, dim=1) # (B, D_sync)
        
        # Cosine similarity sync score (1.0 = perfectly synced, 0.0 = desynced)
        cosine_sim = torch.sum(a_pool * v_pool, dim=-1) # (B,)
        
        # Anomaly score (higher means higher probability of manipulation / desync)
        combined = torch.cat([a_pool, v_pool], dim=-1)
        anomaly_score = self.sync_classifier(combined).squeeze(-1) # (B,)
        
        # Estimated time offset in ms
        offset_ms = (1.0 - torch.clamp(cosine_sim, 0.0, 1.0)) * 250.0

        return {
            "has_audio": True,
            "cosine_similarity": cosine_sim,
            "av_sync_anomaly_score": anomaly_score,
            "av_sync_offset_ms": offset_ms,
            "explanations": []
        }
