"""
Multi-Evidence Fusion & Decision Engine.
Fuses Visual Temporal Consistency, Audio-Visual Sync, and Object Persistence
into a calibrated, robust verdict with structured explanation strings.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

class MultiEvidenceFusionModel(nn.Module):
    """
    Fuses multi-modal forensic signals using learned weighting and heuristic calibration.
    """
    def __init__(
        self,
        temporal_weight_init: float = 0.65,
        av_sync_weight_init: float = 0.20,
        object_weight_init: float = 0.15
    ):
        super().__init__()
        # Learnable logit weights
        self.weights = nn.Parameter(torch.tensor([
            temporal_weight_init,
            av_sync_weight_init,
            object_weight_init
        ], dtype=torch.float32))

    def forward(
        self,
        temporal_score: float,
        av_sync_anomaly_score: Optional[float] = None,
        object_anomaly_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        temporal_score: in [0, 1]
        av_sync_anomaly_score: in [0, 1] or None
        object_anomaly_score: in [0, 1] or None
        """
        w = torch.softmax(self.weights, dim=0).detach()
        w_temp, w_av, w_obj = float(w[0]), float(w[1]), float(w[2])
        
        active_weights = [w_temp]
        scores = [temporal_score]
        explanations = []

        if temporal_score > 0.60:
            explanations.append(f"High inter-frame temporal feature discontinuity detected ({temporal_score*100:.1f}%)")
        elif temporal_score < 0.30:
            explanations.append(f"Smooth, natural inter-frame transition dynamics observed ({temporal_score*100:.1f}%)")

        if av_sync_anomaly_score is not None:
            active_weights.append(w_av)
            scores.append(av_sync_anomaly_score)
            if av_sync_anomaly_score > 0.65:
                explanations.append(f"Audio-to-visual phoneme/viseme timing desynchronization detected ({av_sync_anomaly_score*100:.1f}%)")
        else:
            explanations.append("Audio track absent or silent; audio-visual sync analysis skipped")

        if object_anomaly_score is not None:
            active_weights.append(w_obj)
            scores.append(object_anomaly_score)
            if object_anomaly_score > 0.60:
                explanations.append(f"Facial landmark trajectory jitter and shape boundary distortion observed ({object_anomaly_score*100:.1f}%)")

        # Normalize active weights
        total_w = sum(active_weights)
        norm_weights = [w / total_w for w in active_weights]
        
        final_score = sum(s * w for s, w in zip(scores, norm_weights))
        final_score = max(0.0, min(1.0, float(final_score)))
        
        label = "FAKE" if final_score >= 0.50 else "REAL"
        confidence = final_score if label == "FAKE" else (1.0 - final_score)
        
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "final_score": round(final_score, 4),
            "temporal_score": round(temporal_score, 4),
            "audio_visual_sync_score": round(av_sync_anomaly_score, 4) if av_sync_anomaly_score is not None else None,
            "object_temporal_score": round(object_anomaly_score, 4) if object_anomaly_score is not None else None,
            "explanations": explanations
        }
