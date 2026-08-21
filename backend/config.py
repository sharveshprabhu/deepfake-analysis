from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
REPORTS_DIR = STORAGE_DIR / "reports"
HEATMAPS_DIR = STORAGE_DIR / "heatmaps"
DEMO_DIR = BASE_DIR / "demo_media"

# Ensure directories exist
for path in [STORAGE_DIR, UPLOADS_DIR, REPORTS_DIR, HEATMAPS_DIR, DEMO_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_PATH = STORAGE_DIR / "evidence.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

# Model & System Metadata
MODEL_VERSION = "TruthLens v1.0"
SYSTEM_NAME = "TruthLens Forensic Engine"

# Decision Thresholds
THRESHOLD_MANIPULATED = 0.50
THRESHOLD_AUTHENTIC = 0.35
THRESHOLD_INCONCLUSIVE_SPREAD = 0.45  # If visual and audio/temporal differ wildly

# Signal Fusion Weights (Calculated directly from trained model validation ROC-AUCs)
# Visual (DINOv2): 0.8120, Temporal (BiLSTM/Transformer): 0.7766, Audio (AV-CrossSyncNet): 0.7686, Frequency/SRM: 0.7100
FUSION_WEIGHTS = {
    "visual": 0.35,       # Person 1: DINOv2 + Illumination + Spatial (Val AUC: 0.8120)
    "temporal": 0.40,     # Person 2A: Video Sequence BiLSTM / Transformer (Val AUC: 0.7766)
    "audio": 0.30,        # Person 2B: AV-CrossSyncNet Lip-Sync (Val AUC: 0.7686)
    "frequency": 0.15     # Person 1 Sub-stream: SRM Noise & ELA/FFT (Val AUC: 0.7100)
}

# Max upload size: 100 MB
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Allowed file extensions
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_EXTS = ALLOWED_IMAGE_EXTS.union(ALLOWED_VIDEO_EXTS)

# AI Hardware & Precision Configuration
# All models running strictly in FP32 (Full Precision)
ENABLE_REAL_AI_MODELS = True
AI_DEVICE = "cuda"  # fallback to 'cpu' handled dynamically
MODEL_PRECISION = {
    "visual": "fp32",     # Full precision (FP32): high-frequency forensic artifacts (SRM, FFT, DINOv2)
    "temporal": "fp32",   # Full precision (FP32): video sequence attention tensors
    "audio_av": "fp32",   # Full precision (FP32): acoustic-visual sync features
    "fusion": "cpu"       # CPU mode: statistical & rule-based fusion
}
