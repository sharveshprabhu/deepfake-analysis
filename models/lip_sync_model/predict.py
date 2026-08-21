"""
TruthLens AV-CrossSync Standalone Inference Module.
Analyzes full-video lip synchronization and speech dubbing manipulation.
"""
import os
import sys
import yaml
import subprocess
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

RELEASE_DIR = os.path.dirname(os.path.abspath(__file__))
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

from models.av_cross_syncnet import AVCrossSyncNet
from models.forensic_scorer import ForensicScorer
from data.mouth_extractor import MouthExtractor


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400
) -> np.ndarray:
    """Computes normalized Log-Mel Spectrogram."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0
    )
    log_mel = np.log(np.maximum(mel, 1e-5))
    norm_mel = (log_mel + 4.5) / 4.5
    return norm_mel


class VideoSyncAnalyzer:
    """
    Production Video Audio-Visual Synchronization and Dubbing Forensic Engine.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        if config_path is None:
            config_path = os.path.join(RELEASE_DIR, "config", "sync_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(RELEASE_DIR, "checkpoints", "best_av_cross_syncnet.pt")

        self.model = AVCrossSyncNet(
            feature_dim=self.config["model"]["feature_dim"],
            metric_dim=self.config["model"]["metric_dim"],
            num_cross_attn_heads=self.config["model"]["num_cross_attn_heads"],
            num_cross_attn_layers=self.config["model"]["num_cross_attn_layers"],
            num_offset_classes=self.config["model"]["num_offset_classes"],
            dropout=0.0,
            pretrained=False
        ).to(self.device)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        self.mouth_extractor = MouthExtractor(crop_size=self.config["video"]["crop_size"])
        self.forensic_scorer = ForensicScorer(
            fps=self.config["video"]["fps"],
            sync_threshold_ms=self.config["forensics"]["sync_threshold_ms"],
            desync_threshold=self.config["forensics"]["desync_threshold"]
        )

        self.seq_len = self.config["video"]["sequence_length"] # 5
        self.sr = self.config["audio"]["sample_rate"]          # 16000
        self.fps = self.config["video"]["fps"]                 # 25.0
        self.hop_length = self.config["audio"]["hop_length"]   # 160
        self.win_frames = self.config["audio"]["window_frames"]# 16

    def analyze_video(self, video_path: str, is_lrs3_prealigned: bool = False) -> Dict[str, Any]:
        """
        Runs full temporal scan across video file.
        """
        if not os.path.exists(video_path):
            return {
                "has_audio": False,
                "audio_score": None,
                "av_sync_offset_ms": 0.0,
                "acoustic_artifact_score": 0.0,
                "status": "ERROR",
                "explanations": [f"Video file not found at {video_path}"]
            }

        # 1. Extract audio stream
        try:
            cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-f", "s16le", "-ac", "1", "-ar", str(self.sr), "-"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            if len(out) > 0:
                audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
                has_audio = (len(audio) > 0 and np.max(np.abs(audio)) > 1e-4)
            else:
                has_audio = False
                audio = None
        except Exception:
            has_audio = False
            audio = None

        if not has_audio:
            return {
                "has_audio": False,
                "audio_score": None,
                "av_sync_offset_ms": 0.0,
                "acoustic_artifact_score": 0.0,
                "status": "SUCCESS",
                "explanations": ["Video has no valid audio stream; skipping audio-visual forensics."]
            }

        # 2. Extract video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) < self.seq_len:
            return {
                "has_audio": True,
                "audio_score": 0.5,
                "av_sync_offset_ms": 0.0,
                "acoustic_artifact_score": 0.0,
                "status": "SUCCESS",
                "explanations": ["Video duration is too short for temporal speech analysis"]
            }

        frames = np.array(frames)
        T_total = len(frames)

        # 3. Slide 5-frame window across video
        stride = 2
        window_starts = list(range(0, T_total - self.seq_len + 1, stride))
        max_windows = 25
        if len(window_starts) > max_windows:
            step_w = len(window_starts) // max_windows
            window_starts = window_starts[::step_w][:max_windows]

        batch_lips = []
        batch_mels = []

        for start_f in window_starts:
            v_chunk = frames[start_f : start_f + self.seq_len]
            m_crop = self.mouth_extractor.crop_video_frames(v_chunk, is_lrs3_prealigned=is_lrs3_prealigned)
            m_tensor = torch.from_numpy(m_crop).permute(0, 3, 1, 2).float() / 255.0

            # Corresponding audio slice
            v_center = start_f + (self.seq_len // 2)
            audio_center = int((v_center / self.fps) * self.sr)
            half_len = int((self.win_frames * self.hop_length) / 2)
            a_start = max(0, audio_center - half_len)
            a_end = a_start + (self.win_frames * self.hop_length)

            if a_end > len(audio):
                a_seg = np.pad(audio[a_start:], (0, a_end - len(audio)), mode='constant')
            else:
                a_seg = audio[a_start:a_end]

            mel = compute_mel_spectrogram(
                a_seg,
                sr=self.sr,
                n_mels=self.config["audio"]["n_mels"],
                n_fft=self.config["audio"]["n_fft"],
                hop_length=self.hop_length,
                win_length=self.config["audio"]["win_length"]
            )
            if mel.shape[1] < self.win_frames:
                mel = np.pad(mel, ((0, 0), (0, self.win_frames - mel.shape[1])), mode='constant')
            else:
                mel = mel[:, :self.win_frames]

            mel_tensor = torch.from_numpy(mel).unsqueeze(0).float()

            batch_lips.append(m_tensor)
            batch_mels.append(mel_tensor)

        # Batch tensor execution
        lips_tensor = torch.stack(batch_lips, dim=0).to(self.device)  # (N_win, 5, 3, 96, 96)
        mels_tensor = torch.stack(batch_mels, dim=0).to(self.device)  # (N_win, 1, 80, 16)

        with torch.no_grad():
            outputs = self.model(lips_tensor, mels_tensor)
            offset_logits = outputs["offset_logits"].cpu().numpy()
            v_embed = outputs["v_metric"].cpu()
            a_embed = outputs["a_metric"].cpu()

            cos_sims = torch.sum(v_embed * a_embed, dim=1).numpy()
            attn_diags = torch.diagonal(outputs["attn_matrix"].cpu(), dim1=-2, dim2=-1).mean(dim=-1).numpy()

        # Score across temporal window sequence
        report = self.forensic_scorer.evaluate_video_sync(
            cosine_similarities=cos_sims,
            offset_logits=offset_logits
        )

        return {
            "has_audio": True,
            "audio_score": report["audio_score"],
            "av_sync_offset_ms": report["av_sync_offset_ms"],
            "acoustic_artifact_score": report["acoustic_artifact_score"],
            "confidence": report["confidence"],
            "mean_cosine_similarity": report["mean_cosine_similarity"],
            "explanations": report["explanations"],
            "status": "SUCCESS"
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TruthLens AV-CrossSync Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    analyzer = VideoSyncAnalyzer()
    res = analyzer.analyze_video(args.video)

    print("\n" + "=" * 50)
    print("TruthLens AV-CrossSync Forensics Report:")
    print("=" * 50)
    for k, v in res.items():
        print(f"  {k}: {v}")
