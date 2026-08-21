"""
Suspicious Frame Transition & Timestamp Detector.
Localizes inter-frame discontinuities, trajectory spikes, and flicker anomalies
and converts frame indices to precise video timestamps.
"""
from typing import List, Dict, Any

class TimestampAnomalyDetector:
    """
    Detects and localizes suspicious inter-frame transitions.
    """
    def __init__(self, discontinuity_threshold: float = 0.55):
        self.discontinuity_threshold = discontinuity_threshold

    def detect_suspicious_windows(
        self,
        transition_scores: List[float],
        frame_indices: List[int],
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        transition_scores: list of (T-1) discontinuity scores in [0, 1]
        frame_indices: list of T frame indices
        fps: video frames per second
        """
        suspicious_transitions = []
        suspicious_timestamps = []
        suspicious_frames = []

        for i, score in enumerate(transition_scores):
            f_from = frame_indices[i]
            f_to = frame_indices[i + 1]
            
            if score >= self.discontinuity_threshold:
                t_sec = round(f_from / max(1.0, fps), 2)
                
                # Classify transition anomaly type
                if score > 0.80:
                    a_type = "high_frequency_boundary_flicker"
                    reason = f"Severe blending boundary distortion between frame {f_from} and {f_to}"
                elif score > 0.65:
                    a_type = "inter_frame_landmark_jitter"
                    reason = f"Landmark trajectory jitter observed between frame {f_from} and {f_to}"
                else:
                    a_type = "temporal_texture_discontinuity"
                    reason = f"Temporal feature discontinuity between frame {f_from} and {f_to}"

                suspicious_transitions.append({
                    "from_frame": int(f_from),
                    "to_frame": int(f_to),
                    "discontinuity_score": round(float(score), 4),
                    "type": a_type
                })
                
                suspicious_timestamps.append({
                    "timestamp_seconds": t_sec,
                    "from_frame": int(f_from),
                    "to_frame": int(f_to),
                    "anomaly_score": round(float(score), 4),
                    "reason": reason
                })
                
                if f_from not in suspicious_frames:
                    suspicious_frames.append(int(f_from))
                if f_to not in suspicious_frames:
                    suspicious_frames.append(int(f_to))

        return {
            "suspicious_frame_transitions": suspicious_transitions,
            "suspicious_timestamps": suspicious_timestamps,
            "suspicious_frames": suspicious_frames
        }
