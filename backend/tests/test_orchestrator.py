import asyncio
from pathlib import Path
from backend.services.orchestrator import global_orchestrator
from backend.schemas.contracts import VerdictEnum
from backend.database.db import SessionLocal


def test_orchestrator_pipeline_fake():
    db = SessionLocal()
    demo_file = Path("demo_media/sample_fake.mp4")
    result = asyncio.run(
        global_orchestrator.run_pipeline(
            file_path=demo_file,
            evidence_id="TL-2026-FAKE-TEST",
            filename="sample_fake.mp4",
            db=db
        )
    )
    db.close()

    assert result["evidence_id"] == "TL-2026-FAKE-TEST"
    assert result["verdict"] == VerdictEnum.MANIPULATED
    assert result["confidence"] > 0.70
    assert len(result["suspicious_frames"]) > 0
    assert result["heatmap_url"] is not None
    assert result["report_url"] is not None


def test_orchestrator_pipeline_real():
    db = SessionLocal()
    demo_file = Path("demo_media/sample_real.jpg")
    result = asyncio.run(
        global_orchestrator.run_pipeline(
            file_path=demo_file,
            evidence_id="TL-2026-REAL-TEST",
            filename="sample_real.jpg",
            db=db
        )
    )
    db.close()

    assert result["evidence_id"] == "TL-2026-REAL-TEST"
    assert result["verdict"] == VerdictEnum.AUTHENTIC
    assert result["confidence"] > 0.70


def test_orchestrator_pipeline_inconclusive():
    db = SessionLocal()
    demo_file = Path("demo_media/sample_inconclusive.mp4")
    result = asyncio.run(
        global_orchestrator.run_pipeline(
            file_path=demo_file,
            evidence_id="TL-2026-INCONCL-TEST",
            filename="sample_inconclusive.mp4",
            db=db
        )
    )
    db.close()

    assert result["evidence_id"] == "TL-2026-INCONCL-TEST"
    assert result["verdict"] == VerdictEnum.INCONCLUSIVE


def test_herm_fusion_single_modality_spoofing():
    """
    Verifies that a high-confidence single-modality manipulation (e.g., cloned audio or frame splice)
    is NOT suppressed by pristine/authentic modalities, unlike naive linear averaging.
    """
    # Case: Cloned audio (0.92 score) with pristine visual (0.10) and temporal (0.12)
    fused = global_orchestrator.fusion_adapter.fuse(
        evidence_id="TL-2026-VOICE-CLONE",
        visual_res={"visual_score": 0.10, "frequency_score": 0.12, "status": "SUCCESS"},
        temporal_res={"temporal_score": 0.12, "status": "SUCCESS"},
        audio_res={"audio_score": 0.92, "has_audio": True, "status": "SUCCESS"}
    )
    assert fused["verdict"] == VerdictEnum.MANIPULATED
    assert fused["fusion_score"] >= 0.70
    assert fused["confidence"] >= 0.70
    assert fused["attack_vector"] == "AUDIO_DEEPFAKE_VOICE_CLONE"


def test_herm_fusion_attack_vector_classification():
    """
    Verifies attack vector classification across distinct manipulation typologies.
    """
    # 1. Full multimodal synthesis
    full_synth = global_orchestrator.fusion_adapter.fuse(
        evidence_id="TL-2026-FULL-SYNTH",
        visual_res={"visual_score": 0.88, "frequency_score": 0.85, "status": "SUCCESS"},
        temporal_res={"temporal_score": 0.89, "status": "SUCCESS"},
        audio_res={"audio_score": 0.82, "has_audio": True, "status": "SUCCESS"}
    )
    assert full_synth["verdict"] == VerdictEnum.MANIPULATED
    assert full_synth["attack_vector"] == "MULTIMODAL_FULL_SYNTHESIS"

    # 2. Verified authentic media
    auth_media = global_orchestrator.fusion_adapter.fuse(
        evidence_id="TL-2026-AUTH",
        visual_res={"visual_score": 0.10, "frequency_score": 0.12, "status": "SUCCESS"},
        temporal_res={"temporal_score": 0.11, "status": "SUCCESS"},
        audio_res={"audio_score": 0.08, "has_audio": True, "status": "SUCCESS"}
    )
    assert auth_media["verdict"] == VerdictEnum.AUTHENTIC
    assert auth_media["attack_vector"] == "VERIFIED_AUTHENTIC_MEDIA"


def test_frequency_srm_diagnostic_worker():
    """
    Verifies that the Frequency & SRM diagnostic worker audits media and confirms invariants.
    """
    from backend.services.frequency_srm_worker import global_frequency_srm_worker
    report = global_frequency_srm_worker.run_comprehensive_audit()
    assert report["total_samples_audited"] > 0
    assert "key_findings" in report
    for sample in report["sample_audits"]:
        summ = sample["summary"]
        assert summ["is_calibrated_score_lesser_than_1"] is True

