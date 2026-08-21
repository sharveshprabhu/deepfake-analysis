"""
TruthLens Visual AI Model Adapter & Master Forensic Pipeline (Release v2).
Conforms strictly to TruthLens Backend Specifications.

Combines:
- DINOv2 Spatial Tampering & Neural Mask Extraction (DeepTamperDetector)
- Noise & SRM Inconsistency (SRMNoiseExtractor)
- Frequency & ELA Anomaly (FrequencyForensicsAnalyzer)
- Illumination Physics & Direction Consistency (IlluminationForensicsAnalyzer)
- Calibrated Fusion & Explanation Generator (ImageForensicsFusionEngine)
"""

import sys
import os
import asyncio
from pathlib import Path
import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from inference.srm_filters import SRMNoiseExtractor
    from inference.frequency_analysis import FrequencyForensicsAnalyzer
    from inference.illumination_forensics import IlluminationForensicsAnalyzer
    from inference.deep_tamper_detector import DeepTamperDetector
    from inference.fusion_and_localization import ImageForensicsFusionEngine
    from utils.config import load_config, get_device
except ImportError:
    from srm_filters import SRMNoiseExtractor
    from frequency_analysis import FrequencyForensicsAnalyzer
    from illumination_forensics import IlluminationForensicsAnalyzer
    from deep_tamper_detector import DeepTamperDetector
    from fusion_and_localization import ImageForensicsFusionEngine
    from utils.config import load_config, get_device


class VisualForensicsPipeline:
    """
    Unified Image Forensic Analysis Pipeline for TruthLens Platform.
    """

    def __init__(self, heatmap_storage_dir: str = None, device: str = None):
        config = load_config()
        
        if heatmap_storage_dir is None:
            cfg_dir = config.get("storage", {}).get("heatmap_output_dir", "storage/heatmaps")
            heatmap_storage_dir = str(PACKAGE_ROOT / cfg_dir) if not Path(cfg_dir).is_absolute() else cfg_dir
        
        self.heatmap_storage_dir = Path(heatmap_storage_dir)
        self.heatmap_storage_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            req_dev = config.get("model", {}).get("device", "cuda")
            self.device = get_device(req_dev)
        else:
            self.device = get_device(device)

        # Initialize sub-modules (Full Precision FP32)
        self.srm_extractor = SRMNoiseExtractor()
        self.freq_analyzer = FrequencyForensicsAnalyzer()
        self.illum_analyzer = IlluminationForensicsAnalyzer()
        self.deep_detector = DeepTamperDetector(device=str(self.device))
        self.deep_detector.model = self.deep_detector.model.float()
        self.fusion_engine = ImageForensicsFusionEngine(heatmap_output_dir=str(self.heatmap_storage_dir))
        
        # Face localization module
        try:
            from facenet_pytorch import MTCNN
            dev_str = "cuda" if torch.cuda.is_available() else "cpu"
            self._mtcnn = MTCNN(
                image_size=224,
                margin=20,
                keep_all=False,
                select_largest=True,
                device=dev_str,
                post_process=False
            )
        except Exception:
            self._mtcnn = None

    def _extract_face_crop(self, image_bgr: np.ndarray) -> tuple[np.ndarray, Optional[list]]:
        """Extracts face crop with margin if present, returning (crop_bgr, box)."""
        H, W = image_bgr.shape[:2]
        if self._mtcnn is not None:
            try:
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                scale = min(1.0, 480.0 / max(H, W))
                small = cv2.resize(rgb, (int(W * scale), int(H * scale))) if scale < 1.0 else rgb
                boxes, _ = self._mtcnn.detect(small)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = [int(v / scale) for v in boxes[0]]
                    mx = int((x2 - x1) * 0.15)
                    my = int((y2 - y1) * 0.15)
                    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
                    x2, y2 = min(W, x2 + mx), min(H, y2 + my)
                    crop = image_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        return crop, [y1, x1, y2, x2]
            except Exception:
                pass
        return image_bgr, None

    def _process_single_frame(self, image_bgr: np.ndarray, evidence_id: str = "TL-2026-AUTO", generate_visuals: bool = True, is_video_frame: bool = False):
        """Processes one BGR frame through the full forensic pipeline."""
        # Check for face ROI
        target_roi, face_box = self._extract_face_crop(image_bgr)

        # 1. DINOv2 Deep Spatial Tampering on Face ROI / Primary Subject
        spatial_prob, cam_map, deep_metrics = self.deep_detector.predict(target_roi)

        # 2. SRM Noise Residual & Sensor Noise Variance (on full frame)
        srm_score, srm_map = self.srm_extractor.compute_noise_variance_inconsistency(image_bgr)

        # 3. Frequency & Error Level Analysis (ELA / DCT)
        freq_score, freq_map, freq_metrics = self.freq_analyzer.analyze(image_bgr)
        if is_video_frame:
            # Calibrate down video codec macroblock noise
            freq_score = float(freq_score * 0.65)

        # 4. Illumination Physics & Angular Consistency
        illum_score, angle_deg, illum_map, illum_metrics = self.illum_analyzer.compute_illumination_inconsistency(image_bgr)

        # 5. Multi-Signal Calibration & Fusion
        visual_score, frequency_score, manip_score = self.fusion_engine.fuse_signals(
            spatial_score=spatial_prob,
            srm_score=srm_score,
            frequency_score=freq_score,
            illum_score=illum_score,
            angle_deg=angle_deg
        )

        heatmap_filename = ""
        regions = []
        if generate_visuals:
            heatmap_filename, blended_anomaly_map = self.fusion_engine.generate_heatmap(
                image_bgr=image_bgr,
                cam_map=cam_map,
                srm_map=srm_map,
                freq_map=freq_map,
                illum_map=illum_map,
                evidence_id=evidence_id
            )
            regions = self.fusion_engine.extract_suspicious_regions(
                anomaly_map=blended_anomaly_map,
                illum_score=illum_score,
                freq_score=freq_score,
                angle_deg=angle_deg,
                manipulation_score=manip_score
            )

        return {
            "visual_score": visual_score,
            "frequency_score": frequency_score,
            "manip_score": manip_score,
            "spatial_prob": spatial_prob,
            "srm_score": srm_score,
            "angle_deg": angle_deg,
            "illum_score": illum_score,
            "heatmap_filename": heatmap_filename,
            "regions": regions,
            "deep_metrics": deep_metrics,
            "freq_metrics": freq_metrics,
            "illum_metrics": illum_metrics,
            "cam_map": cam_map,
            "srm_map": srm_map,
            "freq_map": freq_map,
            "illum_map": illum_map
        }

    def analyze_sync(self, file_path: str, evidence_id: str = "TL-2026-AUTO") -> dict:
        """
        Synchronous execution of full multi-stream forensic pipeline.
        Handles both static images and video media (sampling frames across duration).
        """
        path = Path(file_path)
        if not path.exists():
            return {
                "module": "visual_ai",
                "evidence_id": evidence_id,
                "visual_score": 0.0,
                "frequency_score": 0.0,
                "suspicious_frames": [],
                "regions": [],
                "heatmap_filename": "",
                "explanations": [f"File not found: {file_path}"],
                "status": "ERROR"
            }

        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}
        ext = path.suffix.lower()
        is_video = ext in video_exts

        try:
            if is_video:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    return {
                        "module": "visual_ai",
                        "evidence_id": evidence_id,
                        "visual_score": 0.0,
                        "frequency_score": 0.0,
                        "suspicious_frames": [],
                        "regions": [],
                        "heatmap_filename": "",
                        "explanations": [f"Failed to open video stream: {file_path}"],
                        "status": "ERROR"
                    }

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0:
                    fps = 30.0

                if total_frames <= 0:
                    cap.release()
                    return {
                        "module": "visual_ai",
                        "evidence_id": evidence_id,
                        "visual_score": 0.0,
                        "frequency_score": 0.0,
                        "suspicious_frames": [],
                        "regions": [],
                        "heatmap_filename": "",
                        "explanations": [f"Video contains 0 frames: {file_path}"],
                        "status": "ERROR"
                    }

                # Sample up to 8 frames evenly across video duration
                num_samples = min(8, max(4, total_frames // 15))
                sample_indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int).tolist()

                frame_results = []
                suspicious_frames = []
                best_anomaly_idx = 0
                best_anomaly_score = -1.0
                best_frame_bgr = None
                best_res = None

                for f_idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        continue

                    res = self._process_single_frame(frame_bgr, evidence_id=evidence_id, generate_visuals=False, is_video_frame=True)
                    frame_results.append((f_idx, res))

                    if res["manip_score"] > 0.40:
                        suspicious_frames.append(int(f_idx))

                    if res["manip_score"] > best_anomaly_score:
                        best_anomaly_score = res["manip_score"]
                        best_anomaly_idx = f_idx
                        best_frame_bgr = frame_bgr.copy()
                        best_res = res

                cap.release()

                if not frame_results:
                    return {
                        "module": "visual_ai",
                        "evidence_id": evidence_id,
                        "visual_score": 0.0,
                        "frequency_score": 0.0,
                        "suspicious_frames": [],
                        "regions": [],
                        "heatmap_filename": "",
                        "explanations": ["Could not read any frames from video file."],
                        "status": "ERROR"
                    }

                # Generate heatmap for the highest-anomaly frame
                heatmap_filename, blended_anomaly_map = self.fusion_engine.generate_heatmap(
                    image_bgr=best_frame_bgr,
                    cam_map=best_res["cam_map"],
                    srm_map=best_res["srm_map"],
                    freq_map=best_res["freq_map"],
                    illum_map=best_res["illum_map"],
                    evidence_id=evidence_id
                )
                regions = self.fusion_engine.extract_suspicious_regions(
                    anomaly_map=blended_anomaly_map,
                    illum_score=best_res["illum_score"],
                    freq_score=best_res["freq_metrics"].get("combined_frequency_score", 0.0),
                    angle_deg=best_res["angle_deg"],
                    manipulation_score=best_anomaly_score
                )

                # Aggregate multi-frame scores (robust 75th-percentile + mean to suppress single-frame outlier noise)
                vis_scores = [r["visual_score"] for _, r in frame_results]
                freq_scores = [r["frequency_score"] for _, r in frame_results]
                manip_scores = [r["manip_score"] for _, r in frame_results]

                sorted_vis = sorted(vis_scores)
                p75_vis = sorted_vis[min(int(len(sorted_vis) * 0.75), len(sorted_vis) - 1)]
                agg_visual_score = float(0.40 * p75_vis + 0.60 * (sum(vis_scores) / len(vis_scores)))

                sorted_freq = sorted(freq_scores)
                p75_freq = sorted_freq[min(int(len(sorted_freq) * 0.75), len(sorted_freq) - 1)]
                agg_freq_score = float(0.40 * p75_freq + 0.60 * (sum(freq_scores) / len(freq_scores)))

                sorted_manip = sorted(manip_scores)
                p75_manip = sorted_manip[min(int(len(sorted_manip) * 0.75), len(sorted_manip) - 1)]
                agg_manip_score = float(0.40 * p75_manip + 0.60 * (sum(manip_scores) / len(manip_scores)))

                explanations = self.fusion_engine.generate_explanations(
                    visual_score=agg_visual_score,
                    frequency_score=agg_freq_score,
                    manipulation_score=agg_manip_score,
                    illum_score=best_res["illum_score"],
                    angle_deg=best_res["angle_deg"],
                    regions=regions
                )

                return {
                    "module": "visual_ai",
                    "evidence_id": evidence_id,
                    "visual_score": round(agg_visual_score, 3),
                    "frequency_score": round(agg_freq_score, 3),
                    "suspicious_frames": suspicious_frames,
                    "regions": regions,
                    "heatmap_filename": heatmap_filename,
                    "explanations": explanations,
                    "status": "SUCCESS",
                    "details": {
                        "manipulation_score": round(agg_manip_score, 3),
                        "frames_sampled": len(frame_results),
                        "peak_anomaly_frame": best_anomaly_idx,
                        "illumination_angle_discrepancy_deg": round(best_res["angle_deg"], 2),
                        "srm_noise_inconsistency": round(best_res["srm_score"], 3),
                        "deep_tamper_score": round(best_res["spatial_prob"], 3),
                        "deep_architecture": best_res["deep_metrics"].get("architecture", "TruthLensDinov2Net"),
                        "frequency_metrics": best_res["freq_metrics"],
                        "illumination_metrics": best_res["illum_metrics"]
                    }
                }

            else:
                # Static Image Path
                image_bgr = cv2.imread(str(path))
                if image_bgr is None:
                    return {
                        "module": "visual_ai",
                        "evidence_id": evidence_id,
                        "visual_score": 0.0,
                        "frequency_score": 0.0,
                        "suspicious_frames": [],
                        "regions": [],
                        "heatmap_filename": "",
                        "explanations": [f"Failed to decode image file: {file_path}"],
                        "status": "ERROR"
                    }

                res = self._process_single_frame(image_bgr, evidence_id=evidence_id, generate_visuals=True)

                explanations = self.fusion_engine.generate_explanations(
                    visual_score=res["visual_score"],
                    frequency_score=res["frequency_score"],
                    manipulation_score=res["manip_score"],
                    illum_score=res["illum_score"],
                    angle_deg=res["angle_deg"],
                    regions=res["regions"]
                )

                suspicious_frames = [0] if res["manip_score"] > 0.40 else []

                return {
                    "module": "visual_ai",
                    "evidence_id": evidence_id,
                    "visual_score": round(res["visual_score"], 3),
                    "frequency_score": round(res["frequency_score"], 3),
                    "suspicious_frames": suspicious_frames,
                    "regions": res["regions"],
                    "heatmap_filename": res["heatmap_filename"],
                    "explanations": explanations,
                    "status": "SUCCESS",
                    "details": {
                        "manipulation_score": round(res["manip_score"], 3),
                        "illumination_angle_discrepancy_deg": round(res["angle_deg"], 2),
                        "srm_noise_inconsistency": round(res["srm_score"], 3),
                        "deep_tamper_score": round(res["spatial_prob"], 3),
                        "deep_architecture": res["deep_metrics"].get("architecture", "TruthLensDinov2Net"),
                        "frequency_metrics": res["freq_metrics"],
                        "illumination_metrics": res["illum_metrics"]
                    }
                }

        except Exception as e:
            return {
                "module": "visual_ai",
                "evidence_id": evidence_id,
                "visual_score": 0.0,
                "frequency_score": 0.0,
                "suspicious_frames": [],
                "regions": [],
                "heatmap_filename": "",
                "explanations": [f"Forensic pipeline error: {str(e)}"],
                "status": "ERROR"
            }

    async def analyze_async(self, file_path: str, evidence_id: str) -> dict:
        """
        Asynchronous wrapper around the pipeline for FastAPI backend orchestrator integration.
        """
        return await asyncio.to_thread(self.analyze_sync, file_path, evidence_id)


# Global singleton instance
global_visual_pipeline = VisualForensicsPipeline()


# Standard adapter functions
async def truthlens_visual_adapter(file_path: str, evidence_id: str) -> dict:
    return await global_visual_pipeline.analyze_async(file_path, evidence_id)


async def my_visual_model(file_path: str, evidence_id: str = "TL-2026-AUTO") -> dict:
    return await global_visual_pipeline.analyze_async(file_path, evidence_id)
