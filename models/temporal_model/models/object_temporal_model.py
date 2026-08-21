"""
Object Temporal Consistency & Trajectory Persistence Model.
Analyzes object bounding box trajectories, aspect ratio stability, IoU transitions,
and sudden disappearances/reappearances to detect spatial-temporal tampering.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

class ObjectTemporalConsistencyModel(nn.Module):
    """
    Evaluates temporal persistence, trajectory smoothness, and boundary stability
    of visual objects and facial landmarks across frames.
    """
    def __init__(
        self,
        max_jump_dist_threshold: float = 0.15,
        min_iou_threshold: float = 0.4,
        aspect_ratio_delta_threshold: float = 0.35
    ):
        super().__init__()
        self.max_jump_dist_threshold = max_jump_dist_threshold
        self.min_iou_threshold = min_iou_threshold
        self.aspect_ratio_delta_threshold = aspect_ratio_delta_threshold

    def calculate_box_iou(self, boxA: np.ndarray, boxB: np.ndarray) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return float(iou)

    def analyze_trajectory(
        self,
        boxes: List[Optional[np.ndarray]],
        frame_indices: List[int]
    ) -> Dict[str, Any]:
        """
        boxes: list of normalized bounding boxes [x1, y1, x2, y2] per frame (or None if missing).
        frame_indices: list of frame indices corresponding to each box.
        """
        T = len(boxes)
        suspicious_transitions = []
        anomaly_scores = []
        
        for t in range(T - 1):
            box_curr = boxes[t]
            box_next = boxes[t + 1]
            idx_curr = frame_indices[t]
            idx_next = frame_indices[t + 1]
            
            # Case 1: Sudden disappearance / reappearance
            if (box_curr is not None and box_next is None) or (box_curr is None and box_next is not None):
                # Check if box was at edge of screen (normal exit)
                box_present = box_curr if box_curr is not None else box_next
                is_at_edge = (
                    box_present[0] < 0.05 or box_present[1] < 0.05 or
                    box_present[2] > 0.95 or box_present[3] > 0.95
                )
                if not is_at_edge:
                    score = 0.85
                    suspicious_transitions.append({
                        "from_frame": idx_curr,
                        "to_frame": idx_next,
                        "discontinuity_score": score,
                        "type": "abnormal_disappearance" if box_next is None else "abnormal_appearance"
                    })
                    anomaly_scores.append(score)
                continue

            if box_curr is None and box_next is None:
                continue

            # Case 2: Trajectory position jump
            center_curr = np.array([(box_curr[0] + box_curr[2]) / 2, (box_curr[1] + box_curr[3]) / 2])
            center_next = np.array([(box_next[0] + box_next[2]) / 2, (box_next[1] + box_next[3]) / 2])
            jump_dist = np.linalg.norm(center_next - center_curr)
            
            # Case 3: IoU overlap drop
            iou = self.calculate_box_iou(box_curr, box_next)
            
            # Case 4: Sudden Aspect Ratio distortion
            w_c, h_c = max(1e-4, box_curr[2] - box_curr[0]), max(1e-4, box_curr[3] - box_curr[1])
            w_n, h_n = max(1e-4, box_next[2] - box_next[0]), max(1e-4, box_next[3] - box_next[1])
            ar_c = w_c / h_c
            ar_n = w_n / h_n
            ar_delta = abs(ar_n - ar_c) / max(ar_c, ar_n)

            if jump_dist > self.max_jump_dist_threshold or iou < self.min_iou_threshold or ar_delta > self.aspect_ratio_delta_threshold:
                disc_score = min(1.0, float(jump_dist * 3.0 + (1.0 - iou) * 0.5 + ar_delta * 0.5))
                suspicious_transitions.append({
                    "from_frame": idx_curr,
                    "to_frame": idx_next,
                    "discontinuity_score": disc_score,
                    "type": "trajectory_jitter" if jump_dist > self.max_jump_dist_threshold else "shape_distortion"
                })
                anomaly_scores.append(disc_score)

        object_score = float(np.mean(anomaly_scores)) if anomaly_scores else 0.05
        
        return {
            "object_temporal_score": object_score,
            "suspicious_transitions": suspicious_transitions
        }
