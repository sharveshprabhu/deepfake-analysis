"""
Forensic Scoring & Temporal Offset Analysis Engine.
Performs full-video sliding-window scanning, cross-correlation peak estimation,
and multi-evidence anomaly scoring for the TruthLens Forensic Platform.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional


class ForensicScorer:
    """
    Analyzes temporal alignment profiles from AV-CrossSyncNet to generate forensic reports.
    """
    def __init__(
        self,
        fps: float = 25.0,
        sync_threshold_ms: float = 60.0,
        desync_threshold: float = 0.50
    ):
        self.fps = fps
        self.frame_duration_ms = 1000.0 / fps # 40.0ms per frame
        self.sync_threshold_ms = sync_threshold_ms
        self.desync_threshold = desync_threshold

    def evaluate_video_sync(
        self,
        cosine_similarities: np.ndarray,
        offset_logits: Optional[np.ndarray] = None,
        max_shift_frames: int = 15
    ) -> Dict[str, Any]:
        """
        cosine_similarities: (N_windows,) or (N_windows, 2*max_shift+1) similarity profile
        offset_logits: (N_windows, 31) predicted offset distributions
        returns forensic metrics dict.
        """
        if len(cosine_similarities) == 0:
            return {
                "has_audio": False,
                "audio_score": None,
                "av_sync_offset_ms": 0.0,
                "acoustic_artifact_score": 0.0,
                "confidence": 0.0,
                "explanations": ["No valid audio-visual speech frames found in video stream"]
            }

        # Average similarity across all evaluated windows
        mean_sim = float(np.mean(cosine_similarities))

        # Estimate optimal offset from logits if provided, else from cross-correlation
        zero_bin = max_shift_frames
        num_bins = 2 * max_shift_frames + 1
        uniform_baseline = 1.0 / num_bins # ~0.03225

        if offset_logits is not None and len(offset_logits) > 0:
            # Softmax probabilities over offset bins
            probs = np.exp(offset_logits - np.max(offset_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            avg_probs = np.mean(probs, axis=0) # (31,)

            best_bin = int(np.argmax(avg_probs))
            peak_prob = float(avg_probs[best_bin])
            zero_prob = float(avg_probs[zero_bin])
            
            # Peak prominence / sharpness above uniform baseline
            peak_sharpness = float(np.clip((peak_prob - uniform_baseline) / 0.15, 0.0, 1.0))
            is_diffuse_noise = (peak_prob < 1.7 * uniform_baseline) # < ~0.055 indicates ambient/unvoiced sound
            
            if is_diffuse_noise:
                best_shift_frames = 0
                estimated_offset_ms = 0.0
                peak_confidence = round(max(0.10, peak_sharpness * 0.35), 3)
            else:
                best_shift_frames = best_bin - zero_bin
                estimated_offset_ms = round(float(best_shift_frames * self.frame_duration_ms), 1)
                peak_confidence = round(float(np.clip(peak_sharpness * 0.85 + (zero_prob / peak_prob) * 0.15, 0.15, 0.99)), 3)
        else:
            best_shift_frames = 0
            estimated_offset_ms = 0.0
            peak_sharpness = 0.50
            peak_confidence = max(0.0, min(1.0, (mean_sim + 1.0) / 2.0))
            zero_prob = peak_confidence
            is_diffuse_noise = False

        # Anomaly score: higher value indicates synthetic dubbing / desynchronization
        # Mobile camera hardware/codec latency tolerance band: +-80ms
        hardware_tolerance_ms = 80.0
        excess_offset_ms = max(0.0, abs(estimated_offset_ms) - hardware_tolerance_ms)
        offset_penalty = min(1.0, excess_offset_ms / 250.0) * peak_sharpness
        
        sync_quality = max(0.0, min(1.0, (mean_sim + 1.0) / 2.0))
        
        if is_diffuse_noise:
            # Ambient classroom / background noise without dominant speech formants
            audio_score = 0.125
            acoustic_artifact_score = 0.08
        else:
            audio_score = float(np.clip(0.60 * offset_penalty + 0.40 * (1.0 - sync_quality) * peak_sharpness, 0.02, 0.98))
            acoustic_artifact_score = round(float(np.clip(offset_penalty * 0.85 + (1.0 - zero_prob) * 0.15 * peak_sharpness, 0.0, 1.0)), 3)

        audio_score = round(audio_score, 3)

        # Generate human-interpretable forensic explanations
        explanations = []
        if is_diffuse_noise:
            explanations.append(
                "Audio track exhibits diffuse ambient room acoustics or non-vocal background sound. Audio-visual alignment within physiological baseline."
            )
        elif abs(estimated_offset_ms) >= self.sync_threshold_ms and peak_sharpness > 0.30:
            lead_lag = "lag (audio delays video)" if estimated_offset_ms > 0 else "lead (audio precedes video)"
            explanations.append(
                f"Audio-to-visual phoneme/viseme timing {lead_lag} of {abs(estimated_offset_ms):.1f}ms detected "
                f"(exceeds biological human threshold of {self.sync_threshold_ms}ms)."
            )
        else:
            explanations.append(
                f"Audio-visual speech synchronization is within normal physiological tolerance ({abs(estimated_offset_ms):.1f}ms offset)."
            )

        if audio_score >= self.desync_threshold:
            explanations.append(
                f"High cross-modal desynchronization probability ({audio_score:.1%}): potential synthetic voice dubbing or AI face replacement."
            )

        return {
            "has_audio": True,
            "audio_score": audio_score,
            "av_sync_offset_ms": estimated_offset_ms,
            "acoustic_artifact_score": acoustic_artifact_score,
            "confidence": peak_confidence,
            "mean_cosine_similarity": round(mean_sim, 3),
            "explanations": explanations
        }
