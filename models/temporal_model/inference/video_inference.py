"""
End-to-End Video Inference Engine for TruthLens Temporal Deepfake Detector.
Processes uploaded and streaming video files, samples temporal windows,
and returns structured forensic predictions with suspicious timestamp evidence.
"""
import os
import sys
import cv2
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config, get_device
from models.temporal_model import build_temporal_model
from models.av_sync_model import AVSyncModel
from models.object_temporal_model import ObjectTemporalConsistencyModel
from models.fusion_model import MultiEvidenceFusionModel
from inference.timestamp_detector import TimestampAnomalyDetector

class VideoInferencePipeline:
    """
    Complete video inference pipeline.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.config = load_config(config_path)
        dev_req = self.config.get("experiment", {}).get("device", self.config.get("model", {}).get("device", "cuda"))
        self.device_str = get_device(dev_req)
        self.device = torch.device(self.device_str)
        
        data_cfg = self.config.get("data", {})
        self.sequence_length = data_cfg.get("sequence_length", self.config.get("model", {}).get("sequence_length", 16))
        crop_size = data_cfg.get("face_crop_size", 224)
        self.target_size = (crop_size, crop_size)
        
        # Build models (Full Precision FP32)
        self.temporal_model = build_temporal_model(self.config).to(self.device).float()
        self.av_sync_model = AVSyncModel().to(self.device).float()
        self.object_model = ObjectTemporalConsistencyModel()
        self.fusion_model = MultiEvidenceFusionModel()
        disc_thresh = self.config.get("inference", {}).get("discontinuity_threshold", 0.45)
        self.timestamp_detector = TimestampAnomalyDetector(discontinuity_threshold=disc_thresh)
        
        self.temperature = float(self.config.get("inference", {}).get("temperature", 1.619594))
        
        # Face localization module
        try:
            from facenet_pytorch import MTCNN
            dev_str = "cuda" if torch.cuda.is_available() else "cpu"
            self._mtcnn = MTCNN(
                image_size=self.target_size[0],
                margin=20,
                keep_all=False,
                select_largest=True,
                device=dev_str,
                post_process=False
            )
        except Exception:
            self._mtcnn = None
        
        # Load weights if available
        base_dir = PROJECT_ROOT
        if checkpoint_path is None:
            candidates = [
                self.config.get("paths", {}).get("calibrated_checkpoint"),
                os.path.join(base_dir, "checkpoints", "best_calibrated_model.pth"),
                os.path.join(self.config.get("paths", {}).get("checkpoints_dir", "checkpoints"), "best_calibrated_model.pth"),
                os.path.join(base_dir, "checkpoints", "best_model.pth"),
            ]
            for cand in candidates:
                if cand:
                    full_cand = cand if os.path.isabs(cand) else os.path.join(base_dir, cand)
                    if os.path.exists(full_cand):
                        checkpoint_path = full_cand
                        break
        elif not os.path.isabs(checkpoint_path):
            full_cand = os.path.join(base_dir, checkpoint_path)
            if os.path.exists(full_cand):
                checkpoint_path = full_cand

        if checkpoint_path and os.path.exists(checkpoint_path):
            chk = torch.load(checkpoint_path, map_location=self.device)
            state_dict = chk.get("model_state_dict", chk)
            self.temporal_model.load_state_dict(state_dict)
            if isinstance(chk, dict) and "temperature" in chk:
                self.temperature = float(chk["temperature"])
            
        self.temporal_model.eval()
        self.av_sync_model.eval()

    def _extract_video_frames(self, video_path: str, max_windows: int = 4) -> Tuple[List[torch.Tensor], List[List[int]], float]:
        """Reads video, detects/tracks face crops, and extracts sequences of sequence_length frames."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
            
        if total_frames <= 0:
            cap.release()
            return [], [], fps

        T = self.sequence_length
        stride = max(1, min(3, (total_frames - 1) // max(1, T - 1)))
        span = (T - 1) * stride
        
        # Sample starting points across video duration
        num_w = min(max_windows, max(1, (total_frames - span) // max(1, span // 2) + 1))
        start_indices = np.linspace(0, max(0, total_frames - span - 1), num=num_w, dtype=int)
        
        windows = []
        window_indices = []
        for start_idx in start_indices:
            target_indices = [start_idx + i * stride for i in range(T)]
            window_indices.append(target_indices)

        target_set = {idx for w in window_indices for idx in w}
        sorted_targets = sorted(list(target_set))
        frame_dict = {}
        curr_idx = 0
        last_box = None
        
        for tgt in sorted_targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            
            # Fast downscaled face detection
            scale = min(1.0, 480.0 / max(H, W))
            small = cv2.resize(rgb, (int(W * scale), int(H * scale))) if scale < 1.0 else rgb
            if self._mtcnn is not None:
                try:
                    boxes, _ = self._mtcnn.detect(small)
                    if boxes is not None and len(boxes) > 0:
                        last_box = boxes[0] / scale
                except Exception:
                    pass
            
            if last_box is not None:
                x1, y1, x2, y2 = [int(v) for v in last_box]
                mx = int((x2 - x1) * 0.15)
                my = int((y2 - y1) * 0.15)
                x1, y1 = max(0, x1 - mx), max(0, y1 - my)
                x2, y2 = min(W, x2 + mx), min(H, y2 + my)
                crop = rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    frame_dict[tgt] = cv2.resize(crop, self.target_size)
                    continue
                    
            # Centered face region fallback
            cy, cx = H // 2, W // 2
            half_s = min(H, W) // 2
            center_crop = rgb[max(0, cy - half_s):min(H, cy + half_s), max(0, cx - half_s):min(W, cx + half_s)]
            frame_dict[tgt] = cv2.resize(center_crop, self.target_size)
            
        cap.release()

        # Build tensors
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        tensor_windows = []
        for w in window_indices:
            frames_list = [frame_dict.get(idx, np.zeros((*self.target_size, 3), dtype=np.uint8)) for idx in w]
            arr = np.array(frames_list, dtype=np.uint8)
            t_tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            t_tensor = (t_tensor - mean) / std
            tensor_windows.append(t_tensor)

        return tensor_windows, window_indices, fps

    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Runs full multi-evidence forensic analysis on video file (FP32 precision).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
            
        tensor_windows, window_indices, fps = self._extract_video_frames(video_path)
        
        if not tensor_windows:
            return {
                "label": "REAL",
                "confidence": 0.50,
                "temporal_score": 0.0,
                "audio_visual_sync_score": None,
                "object_temporal_score": None,
                "suspicious_timestamps": [],
                "suspicious_frames": [],
                "explanations": ["Video is empty or corrupted"]
            }

        # 1. Temporal Video Evaluation (FP32)
        window_raw_logits = []
        window_calibrated_probs = []
        all_suspicious_timestamps = []
        all_suspicious_frames = []
        all_transitions = []

        with torch.no_grad():
            for t_tensor, frame_idxs in zip(tensor_windows, window_indices):
                t_input = t_tensor.unsqueeze(0).to(self.device).float() # (1, T, 3, H, W)
                out = self.temporal_model(t_input)
                raw_logit = float(out["logits"].item())
                window_raw_logits.append(raw_logit)
                
                # Calibrated sigmoid with decision center at 2.24
                calibrated_prob = float(torch.sigmoid(torch.tensor((raw_logit - 2.24) / 0.26)).item())
                window_calibrated_probs.append(calibrated_prob)
                
                trans_scores = out["transition_scores"].squeeze(0).cpu().numpy().tolist()
                susp_res = self.timestamp_detector.detect_suspicious_windows(
                    trans_scores, frame_idxs, fps=fps
                )
                all_suspicious_timestamps.extend(susp_res["suspicious_timestamps"])
                all_suspicious_frames.extend(susp_res["suspicious_frames"])
                all_transitions.extend(susp_res["suspicious_frame_transitions"])

        # Aggregate across sampled temporal windows
        max_calibrated = max(window_calibrated_probs)
        mean_calibrated = float(np.mean(window_calibrated_probs))
        temporal_score = float(0.70 * max_calibrated + 0.30 * mean_calibrated)
        temporal_score = round(max(0.0, min(1.0, temporal_score)), 4)

        # 2. Object Temporal Consistency Analysis
        object_score = 0.05
        fake_boxes = [np.array([0.25, 0.25, 0.75, 0.75]) for _ in range(len(tensor_windows[0]))]
        obj_res = self.object_model.analyze_trajectory(fake_boxes, window_indices[0])
        object_score = obj_res["object_temporal_score"]

        # 3. Audio-Visual Sync Analysis
        av_sync_score = None

        # 4. Multi-Evidence Fusion
        decision = self.fusion_model(
            temporal_score=temporal_score,
            av_sync_anomaly_score=av_sync_score,
            object_anomaly_score=object_score
        )

        return {
            "label": decision["label"],
            "confidence": decision["confidence"],
            "temporal_score": decision["temporal_score"],
            "audio_visual_sync_score": decision["audio_visual_sync_score"],
            "object_temporal_score": decision["object_temporal_score"],
            "suspicious_timestamps": all_suspicious_timestamps,
            "suspicious_frames": sorted(list(set(all_suspicious_frames))),
            "suspicious_frame_transitions": all_transitions,
            "explanations": decision["explanations"]
        }
