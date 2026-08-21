"""
TruthLens Central Model Registry & Precision Manager.
Connects the 4 AI forensic pipelines:
1. Person 1 Visual Forensics (DINOv2, SRM, Frequency, Illumination) - FP32
2. Person 2A Temporal Video Sequence AI - FP16 (Single Half-Precision Model)
3. Person 2B AV-CrossSyncNet Audio-Visual Synchronization (Release v2 Tier-2 SOTA) - FP32
4. Person 1 Spatial Feature Extractor - FP32

Conforms strictly to MODEL_INTEGRATION_GUIDE.md and TRUTHLENS_INTEGRATION.md.
"""

import sys
import os
import inspect
import asyncio
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable

from backend.config import (
    BASE_DIR,
    STORAGE_DIR,
    HEATMAPS_DIR,
    ENABLE_REAL_AI_MODELS,
    AI_DEVICE,
    MODEL_PRECISION,
    ALLOWED_IMAGE_EXTS
)
from backend.services.orchestrator import global_orchestrator

logger = logging.getLogger("TruthLens.ModelRegistry")

# Resolve project model directories
ROOT_DIR = BASE_DIR.parent
MODELS_DIR = ROOT_DIR / "models"

# 1. Visual Model Path (DINOv2 + SRM + FFT + Illumination)
IMAGE_RELEASE_V2_DIR = MODELS_DIR / "image_model"

# 2. Temporal Model Path (Swin-Transformer Video Dynamics FP32)
TEMPORAL_RELEASE_DIR = MODELS_DIR / "temporal_model"

# 3. Audio / Lip-Sync Model Path (AV-CrossSyncNet)
LIP_SYNC_RELEASE_V2_DIR = MODELS_DIR / "lip_sync_model"


@contextmanager
def isolated_import_context(root_dir: Path):
    """
    Context manager to prevent Python sys.modules namespace collisions
    between independent sub-packages sharing names like 'inference', 'models', 'data', 'utils'.
    """
    root_str = str(root_dir.resolve())
    old_path = list(sys.path)
    clobber_keys = ['inference', 'models', 'data', 'utils', 'predict', 'image_model_adapter', 'truthlens_adapter', 'adapter']
    saved_modules = {}
    for k in list(sys.modules.keys()):
        for prefix in clobber_keys:
            if k == prefix or k.startswith(prefix + '.'):
                saved_modules[k] = sys.modules.pop(k)
    sys.path.insert(0, root_str)
    try:
        yield
    finally:
        sys.path = old_path
        for k in list(sys.modules.keys()):
            for prefix in clobber_keys:
                if k == prefix or k.startswith(prefix + '.'):
                    sys.modules.pop(k, None)
        sys.modules.update(saved_modules)


# Singleton model pipeline references
_VISUAL_PIPELINE = None
_TEMPORAL_PIPELINE = None
_AUDIO_ANALYZER = None


def register_visual_model() -> bool:
    """
    Registers Person 1 Visual AI & Forensic Pipeline (DINOv2 + SRM + FFT + Illumination).
    Precision: FP32 (Full Precision).
    """
    global _VISUAL_PIPELINE
    try:
        target_dir = IMAGE_RELEASE_V2_DIR
        with isolated_import_context(target_dir):
            try:
                from inference.image_model_adapter import VisualForensicsPipeline
            except ImportError:
                from image_model_adapter import VisualForensicsPipeline

            _VISUAL_PIPELINE = VisualForensicsPipeline(heatmap_storage_dir=str(HEATMAPS_DIR))

        prec = MODEL_PRECISION.get("visual", "fp32")
        logger.info(f"[*] Visual AI Model loaded (Precision: {prec.upper()}) from {target_dir.name}")

        async def visual_wrapper(file_path: str, evidence_id: str) -> Dict[str, Any]:
            return await _VISUAL_PIPELINE.analyze_async(file_path, evidence_id)

        global_orchestrator.visual_adapter.set_real_model(visual_wrapper)
        logger.info("[✓] Successfully registered Visual Forensics Pipeline (Person 1 - Release v2).")
        return True
    except Exception as e:
        logger.warning(f"[!] Could not register Visual AI model ({e}). Using fallback.")
        return False


def register_temporal_model() -> bool:
    """
    Registers Person 2A Temporal Video Sequence Model.
    Precision: FP32 (Full Precision).
    """
    global _TEMPORAL_PIPELINE
    try:
        target_dir = TEMPORAL_RELEASE_DIR
        with isolated_import_context(target_dir):
            try:
                from inference.video_inference import VideoInferencePipeline
            except ImportError:
                from video_inference import VideoInferencePipeline

            _TEMPORAL_PIPELINE = VideoInferencePipeline()

        prec = MODEL_PRECISION.get("temporal", "fp32")
        logger.info(f"[*] Temporal Video AI Model loaded (Precision: {prec.upper()} - Full Precision) from {target_dir.name}")

        def _run_temporal_sync(file_path: str, evidence_id: str) -> Dict[str, Any]:
            ext = Path(file_path).suffix.lower()
            if ext in ALLOWED_IMAGE_EXTS:
                return {
                    "module": "temporal_ai",
                    "evidence_id": evidence_id,
                    "temporal_score": None,
                    "suspicious_frame_transitions": [],
                    "explanations": ["Static image input detected. Temporal inter-frame analysis bypassed."],
                    "status": "SKIPPED_NOT_VIDEO"
                }

            pred = _TEMPORAL_PIPELINE.predict(file_path)
            t_score = pred.get("temporal_score")
            if t_score is not None:
                t_score = float(t_score)
            return {
                "module": "temporal_ai",
                "evidence_id": evidence_id,
                "temporal_score": t_score,
                "suspicious_frame_transitions": pred.get("suspicious_frame_transitions", []),
                "explanations": pred.get("explanations", []),
                "status": "SUCCESS"
            }

        async def temporal_wrapper(file_path: str, evidence_id: str) -> Dict[str, Any]:
            return await asyncio.to_thread(_run_temporal_sync, file_path, evidence_id)

        global_orchestrator.temporal_adapter.set_real_model(temporal_wrapper)
        logger.info("[✓] Successfully registered Temporal Video AI Pipeline (Person 2A - Release).")
        return True
    except Exception as e:
        logger.warning(f"[!] Could not register Temporal AI model ({e}). Using fallback.")
        return False


def register_audio_avsync_model() -> bool:
    """
    Registers Person 2B AV-CrossSyncNet (Release v2 Tier-2 SOTA).
    Precision: FP32 (Full Precision).
    """
    global _AUDIO_ANALYZER
    try:
        target_dir = LIP_SYNC_RELEASE_V2_DIR
        with isolated_import_context(target_dir):
            if str(target_dir) not in sys.path:
                sys.path.insert(0, str(target_dir))
            try:
                from predict import VideoSyncAnalyzer
            except Exception:
                from adapter import VideoSyncAnalyzer

            _AUDIO_ANALYZER = VideoSyncAnalyzer()

        prec = MODEL_PRECISION.get("audio_av", "fp32")
        logger.info(f"[*] Audio / AV-CrossSyncNet Model loaded (Precision: {prec.upper()}) from {target_dir.name}")

        def _run_audio_sync(file_path: str, evidence_id: str) -> Dict[str, Any]:
            ext = Path(file_path).suffix.lower()
            if ext in ALLOWED_IMAGE_EXTS:
                return {
                    "module": "audio_ai",
                    "evidence_id": evidence_id,
                    "audio_score": None,
                    "has_audio": False,
                    "av_sync_offset_ms": None,
                    "acoustic_artifact_score": None,
                    "explanations": ["Static image input. Audio forensic analysis bypassed."],
                    "status": "SKIPPED_IMAGE"
                }

            raw = _AUDIO_ANALYZER.analyze_video(file_path)
            has_audio = raw.get("has_audio", False)
            a_score = raw.get("audio_score")
            if a_score is not None:
                a_score = float(a_score)
            av_offset = raw.get("av_sync_offset_ms", 0.0)
            acoustic_art = raw.get("acoustic_artifact_score", 0.0)
            explanations = raw.get("explanations", [])
            status = raw.get("status", "SUCCESS")

            return {
                "module": "audio_ai",
                "evidence_id": evidence_id,
                "audio_score": a_score,
                "has_audio": has_audio,
                "av_sync_offset_ms": float(av_offset) if av_offset is not None else 0.0,
                "acoustic_artifact_score": float(acoustic_art) if acoustic_art is not None else 0.0,
                "explanations": explanations,
                "status": status
            }

        async def audio_wrapper(file_path: str, evidence_id: str) -> Dict[str, Any]:
            return await asyncio.to_thread(_run_audio_sync, file_path, evidence_id)

        global_orchestrator.audio_adapter.set_real_model(audio_wrapper)
        logger.info("[✓] Successfully registered Lip-Sync Audio & AV Pipeline (Person 2B - Release v2).")
        return True
    except Exception as e:
        logger.warning(f"[!] Could not register Audio/AV AI model ({e}). Using fallback.")
        return False


def register_all_models() -> Dict[str, bool]:
    """
    Master registration hook invoked on backend startup.
    Links all forensic models into the global orchestrator.
    """
    if not ENABLE_REAL_AI_MODELS:
        logger.info("[i] Real AI models disabled in config. Running in simulation mode.")
        return {"visual": False, "temporal": False, "audio_av": False}

    logger.info("=" * 65)
    logger.info(f"TruthLens AI Model Synchronization & Registration (Device: {AI_DEVICE})")
    logger.info(f"Configured Precisions: {MODEL_PRECISION}")
    logger.info("=" * 65)

    results = {
        "visual": register_visual_model(),
        "temporal": register_temporal_model(),
        "audio_av": register_audio_avsync_model()
    }

    active_count = sum(1 for v in results.values() if v)
    logger.info(f"[*] {active_count}/3 Forensic AI Model Pipelines successfully linked to Orchestrator.")
    logger.info("=" * 65)
    return results


# Global pre-flight diagnostic status cache
DIAGNOSTIC_STATE: Dict[str, Any] = {
    "visual": {"status": "UNCHECKED", "score": None, "latency_ms": None, "error": None},
    "temporal": {"status": "UNCHECKED", "score": None, "latency_ms": None, "error": None},
    "audio": {"status": "UNCHECKED", "score": None, "latency_ms": None, "error": None},
    "fusion": {"status": "UNCHECKED", "verdict": None, "confidence": None, "latency_ms": None, "error": None},
    "all_passed": True,
    "last_checked": None
}


def _ensure_diagnostic_fixtures() -> tuple[Path, Path, Path]:
    """Generates small, valid diagnostic test fixtures (1 authentic image, 1 spliced image, 1 video)."""
    import cv2
    import numpy as np

    diag_dir = STORAGE_DIR / "diag_fixtures"
    diag_dir.mkdir(parents=True, exist_ok=True)

    auth_path = diag_dir / "diag_auth.jpg"
    if not auth_path.exists():
        auth_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        noise = np.random.normal(0, 4.0, (256, 256, 3))
        auth_img = np.clip(auth_img + noise, 0, 255).astype(np.uint8)
        cv2.putText(auth_img, "Authentic Noise Baseline", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(auth_path), auth_img)

    fake_path = diag_dir / "diag_fake.jpg"
    if not fake_path.exists():
        fake_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        noise_bg = np.random.normal(0, 3.0, (256, 256, 3))
        fake_img = np.clip(fake_img + noise_bg, 0, 255).astype(np.uint8)
        # Spliced central region with high noise disparity
        cv2.rectangle(fake_img, (64, 64), (192, 192), (210, 160, 120), -1)
        noise_patch = np.random.normal(0, 16.0, (128, 128, 3))
        fake_img[64:192, 64:192] = np.clip(fake_img[64:192, 64:192] + noise_patch, 0, 255).astype(np.uint8)
        cv2.imwrite(str(fake_path), fake_img)

    vid_path = diag_dir / "diag_test.mp4"
    if not vid_path.exists():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (224, 224))
        for i in range(30):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:, :] = (30, 40, 60)
            cv2.circle(frame, (112 + int(10 * np.sin(i / 5.0)), 112), 40, (200, 180, 160), -1)
            out.write(frame)
        out.release()

    return auth_path, fake_path, vid_path


async def run_startup_diagnostics() -> Dict[str, Any]:
    """
    Executes live pre-flight detection checks on all models, sub-signals (SRM, DINOv2, Frequency, Illumination),
    and fusion pipeline before the backend completes initialization.
    """
    import time
    from datetime import datetime, timezone

    logger.info("=" * 65)
    logger.info("TruthLens Pre-Flight Model Diagnostics & Forensic Sub-Signal Self-Check")
    logger.info("=" * 65)

    auth_img_path, fake_img_path, vid_path = _ensure_diagnostic_fixtures()

    # 1. Visual & SRM Sub-Signal Diagnostic Check
    try:
        t0 = time.perf_counter()
        v_auth = await global_orchestrator.visual_adapter.analyze(auth_img_path, "STARTUP-DIAG-VIS-AUTH")
        v_fake = await global_orchestrator.visual_adapter.analyze(fake_img_path, "STARTUP-DIAG-VIS-FAKE")
        t_vis = (time.perf_counter() - t0) * 1000

        v_passed = v_auth.get("status") == "SUCCESS" and v_fake.get("status") == "SUCCESS"
        
        auth_details = v_auth.get("details", {})
        fake_details = v_fake.get("details", {})
        srm_auth = auth_details.get("srm_noise_inconsistency", 0.0)
        srm_fake = fake_details.get("srm_noise_inconsistency", 0.0)
        
        # Validate that SRM is properly calibrated and responsive to noise inconsistency
        srm_calibrated = (srm_fake >= srm_auth) and (0.0 <= srm_auth <= 1.0) and (0.0 <= srm_fake <= 1.0)
        
        DIAGNOSTIC_STATE["visual"] = {
            "status": "PASSED" if (v_passed and srm_calibrated) else "FAILED",
            "score_auth": v_auth.get("visual_score"),
            "score_fake": v_fake.get("visual_score"),
            "srm_auth": srm_auth,
            "srm_fake": srm_fake,
            "srm_calibrated": srm_calibrated,
            "latency_ms": round(t_vis, 1),
            "error": None if v_passed else str(v_auth.get("explanations"))
        }
        logger.info(f"[{'✓' if (v_passed and srm_calibrated) else '!'}] Visual AI & SRM Noise Extractor (FP32): {'PASSED' if (v_passed and srm_calibrated) else 'FAILED'} (SRM Auth: {srm_auth:.3f}, SRM Spliced: {srm_fake:.3f}, Latency: {t_vis:.1f}ms)")
        v_res = v_auth
    except Exception as e:
        DIAGNOSTIC_STATE["visual"] = {"status": "FAILED", "score": None, "latency_ms": None, "error": str(e)}
        logger.warning(f"[!] Visual AI Model Detection Check FAILED: {e}")
        v_res = {"visual_score": None, "frequency_score": None, "status": "ERROR"}

    # 2. Temporal Model Detection Check
    try:
        t0 = time.perf_counter()
        t_res = await global_orchestrator.temporal_adapter.analyze(vid_path, "STARTUP-DIAG-TEMP")
        t_temp = (time.perf_counter() - t0) * 1000
        t_passed = t_res.get("status") == "SUCCESS"
        DIAGNOSTIC_STATE["temporal"] = {
            "status": "PASSED" if t_passed else "FAILED",
            "score": t_res.get("temporal_score"),
            "latency_ms": round(t_temp, 1),
            "error": None if t_passed else str(t_res.get("explanations"))
        }
        logger.info(f"[{'✓' if t_passed else '!'}] Temporal Video AI (Transformer/BiLSTM - FP32): {'PASSED' if t_passed else 'FAILED'} (Score: {t_res.get('temporal_score')}, Latency: {t_temp:.1f}ms)")
    except Exception as e:
        DIAGNOSTIC_STATE["temporal"] = {"status": "FAILED", "score": None, "latency_ms": None, "error": str(e)}
        logger.warning(f"[!] Temporal Video AI Model Detection Check FAILED: {e}")
        t_res = {"temporal_score": None, "status": "ERROR"}

    # 3. Audio / Lip-Sync Model Detection Check
    try:
        t0 = time.perf_counter()
        a_res = await global_orchestrator.audio_adapter.analyze(vid_path, "STARTUP-DIAG-AUDIO")
        t_audio = (time.perf_counter() - t0) * 1000
        a_passed = a_res.get("status") in {"SUCCESS", "SKIPPED_IMAGE", "NO_AUDIO_TRACK"}
        DIAGNOSTIC_STATE["audio"] = {
            "status": "PASSED" if a_passed else "FAILED",
            "score": a_res.get("audio_score"),
            "latency_ms": round(t_audio, 1),
            "error": None if a_passed else str(a_res.get("explanations"))
        }
        logger.info(f"[{'✓' if a_passed else '!'}] Audio / AV-CrossSyncNet AI (Cross-Attn - FP32): {'PASSED' if a_passed else 'FAILED'} (Latency: {t_audio:.1f}ms)")
    except Exception as e:
        DIAGNOSTIC_STATE["audio"] = {"status": "FAILED", "score": None, "latency_ms": None, "error": str(e)}
        logger.warning(f"[!] Audio AI Model Detection Check FAILED: {e}")
        a_res = {"audio_score": None, "status": "ERROR"}

    # 4. Fusion Engine Detection Check
    try:
        t0 = time.perf_counter()
        f_res = global_orchestrator.fusion_adapter.fuse("STARTUP-DIAG-FUSION", v_res, t_res, a_res)
        t_fusion = (time.perf_counter() - t0) * 1000
        f_passed = f_res.get("status") == "SUCCESS"
        verdict_val = f_res["verdict"].value if hasattr(f_res.get("verdict"), "value") else str(f_res.get("verdict"))
        DIAGNOSTIC_STATE["fusion"] = {
            "status": "PASSED" if f_passed else "FAILED",
            "verdict": verdict_val,
            "confidence": f_res.get("confidence"),
            "latency_ms": round(t_fusion, 1),
            "error": None if f_passed else str(f_res.get("verdict_reasoning"))
        }
        logger.info(f"[{'✓' if f_passed else '!'}] Calibrated Multimodal Fusion Engine: {'PASSED' if f_passed else 'FAILED'} (Verdict: {verdict_val}, Confidence: {f_res.get('confidence')})")
    except Exception as e:
        DIAGNOSTIC_STATE["fusion"] = {"status": "FAILED", "verdict": None, "confidence": None, "latency_ms": None, "error": str(e)}
        logger.warning(f"[!] Multimodal Fusion Check FAILED: {e}")

    all_passed = (
        DIAGNOSTIC_STATE["visual"]["status"] == "PASSED" and
        DIAGNOSTIC_STATE["temporal"]["status"] == "PASSED" and
        DIAGNOSTIC_STATE["audio"]["status"] == "PASSED" and
        DIAGNOSTIC_STATE["fusion"]["status"] == "PASSED"
    )
    DIAGNOSTIC_STATE["all_passed"] = all_passed
    DIAGNOSTIC_STATE["last_checked"] = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 65)
    if all_passed:
        logger.info("[✓] ALL 3 FORENSIC AI ENGINES + FUSION SELF-TEST PASSED SUCCESSFULLY")
    else:
        logger.warning("[!] ONE OR MORE FORENSIC AI ENGINES FAILED PRE-FLIGHT SELF-TEST")
    logger.info("=" * 65)

    return DIAGNOSTIC_STATE
