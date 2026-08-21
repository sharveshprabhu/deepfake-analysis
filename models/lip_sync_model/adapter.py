"""
TruthLens Backend Integration Adapter for Person 2B (Audio Forensic & AV-Sync Adapter).
Conforms strictly to Section 3 of MODEL_INTEGRATION_GUIDE.md.
"""
import os
import sys
import asyncio
from typing import Dict, Any, List, Optional

RELEASE_DIR = os.path.dirname(os.path.abspath(__file__))
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

from predict import VideoSyncAnalyzer

# Singleton analyzer instance
_ANALYZER_INSTANCE: Optional[VideoSyncAnalyzer] = None


def get_analyzer() -> VideoSyncAnalyzer:
    """Returns singleton instance of VideoSyncAnalyzer."""
    global _ANALYZER_INSTANCE
    if _ANALYZER_INSTANCE is None:
        _ANALYZER_INSTANCE = VideoSyncAnalyzer()
    return _ANALYZER_INSTANCE


def run_sync_analysis(video_path: str, evidence_id: str = "TL-EVID-DEFAULT") -> Dict[str, Any]:
    """
    Synchronous analysis entrypoint.
    Returns compliant Person 2B contract schema.
    """
    analyzer = get_analyzer()
    raw = analyzer.analyze_video(video_path)

    has_audio = raw.get("has_audio", False)
    audio_score = raw.get("audio_score")
    av_sync_offset_ms = raw.get("av_sync_offset_ms", 0.0)
    acoustic_artifact_score = raw.get("acoustic_artifact_score", 0.0)
    explanations = raw.get("explanations", [])
    status = raw.get("status", "SUCCESS")

    return {
        "module": "audio_ai",
        "evidence_id": evidence_id,
        "audio_score": audio_score,
        "has_audio": has_audio,
        "av_sync_offset_ms": float(av_sync_offset_ms),
        "acoustic_artifact_score": float(acoustic_artifact_score),
        "explanations": explanations,
        "status": status
    }


async def truthlens_audio_avsync_adapter(
    evidence_id: str,
    video_path: str,
    extra_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Asynchronous TruthLens Person 2B Adapter function.
    Non-blocking async wrapper executing in worker thread pool.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        run_sync_analysis,
        video_path,
        evidence_id
    )
    return result


def register_with_orchestrator(orchestrator: Any = None) -> Dict[str, Any]:
    """
    Auto-registration hook for TruthLens backend orchestrator.
    """
    module_info = {
        "module_name": "audio_ai",
        "version": "2.0.0",
        "author": "Person 2B (Audio Forensic & AV-Sync Specialist)",
        "model_architecture": "AV-CrossSyncNet (Pretrained 3D-ResNet + Cross-Attention Transformer)",
        "contract_schema": {
            "module": "audio_ai",
            "evidence_id": "str",
            "audio_score": "Optional[float] (0.0=real, 1.0=manipulated)",
            "has_audio": "bool",
            "av_sync_offset_ms": "float (audio-visual offset in ms)",
            "acoustic_artifact_score": "float",
            "explanations": "List[str]",
            "status": "SUCCESS | ERROR"
        },
        "async_handler": truthlens_audio_avsync_adapter,
        "sync_handler": run_sync_analysis
    }
    if orchestrator is not None and hasattr(orchestrator, "register_module"):
        orchestrator.register_module("audio_ai", truthlens_audio_avsync_adapter)
    return module_info
