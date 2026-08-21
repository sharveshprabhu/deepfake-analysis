"""
TruthLens Master Pre-Hackathon Verification Test Suite
For Person 3 (Backend + Orchestrator + Evidence Guardian)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Ensure stdout supports UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pytest
import asyncio
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.db import init_db, SessionLocal
from backend.database.models import EvidenceRecord
from backend.services.hasher import calculate_sha256_file, calculate_sha256_bytes, verify_file_integrity
from backend.services.id_generator import generate_evidence_id
from backend.services.report_generator import generate_pdf_report
from backend.services.orchestrator import global_orchestrator
from backend.schemas.contracts import VerdictEnum


client = TestClient(app)


def run_all_checks():
    """Runs all 12 core verification checks for Person 3."""
    results = {}

    # 1. Server check
    try:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "HEALTHY"
        results["Server"] = True
    except Exception as e:
        print(f"[FAIL] Server: {e}")
        results["Server"] = False

    # 2. Upload API check
    demo_file = Path("demo_media/sample_fake.jpg")
    if not demo_file.exists():
        pytest.fail("demo_media/sample_fake.jpg not found. Run demo generator first.")

    evidence_id = None
    sha256_val = None
    try:
        with open(demo_file, "rb") as f:
            res = client.post("/upload", files={"file": ("sample_fake.jpg", f, "image/jpeg")})
        assert res.status_code == 201
        data = res.json()
        assert "evidence_id" in data
        assert "sha256" in data
        evidence_id = data["evidence_id"]
        sha256_val = data["sha256"]
        results["Upload API"] = True
    except Exception as e:
        print(f"[FAIL] Upload API: {e}")
        results["Upload API"] = False

    # 3. File storage check
    try:
        saved_files = list(Path("storage/uploads").glob(f"{evidence_id}_*"))
        assert len(saved_files) >= 1
        assert saved_files[0].exists()
        results["File storage"] = True
    except Exception as e:
        print(f"[FAIL] File storage: {e}")
        results["File storage"] = False

    # 4. SHA-256 check
    try:
        computed_hash = calculate_sha256_file(demo_file)
        assert computed_hash == sha256_val
        assert verify_file_integrity(demo_file, computed_hash) is True
        results["SHA-256"] = True
    except Exception as e:
        print(f"[FAIL] SHA-256: {e}")
        results["SHA-256"] = False

    # 5. Database check
    try:
        db = SessionLocal()
        record = db.query(EvidenceRecord).filter(EvidenceRecord.evidence_id == evidence_id).first()
        assert record is not None
        assert record.sha256 == sha256_val
        db.close()
        results["Database"] = True
    except Exception as e:
        print(f"[FAIL] Database: {e}")
        results["Database"] = False

    # 6. Visual API check
    try:
        v_res = asyncio.run(
            global_orchestrator.visual_adapter.analyze(demo_file, evidence_id)
        )
        assert v_res["module"] == "visual_ai"
        assert "visual_score" in v_res
        assert "heatmap_filename" in v_res
        results["Visual API"] = True
    except Exception as e:
        print(f"[FAIL] Visual API: {e}")
        results["Visual API"] = False

    # 7. Temporal API check
    try:
        video_demo = Path("demo_media/sample_fake.mp4")
        t_res = asyncio.run(
            global_orchestrator.temporal_adapter.analyze(video_demo, "TL-2026-TEST")
        )
        assert t_res["module"] == "temporal_ai"
        assert "temporal_score" in t_res
        results["Temporal API"] = True
    except Exception as e:
        print(f"[FAIL] Temporal API: {e}")
        results["Temporal API"] = False

    # 8. Audio API check (including no-audio handling)
    try:
        no_audio_demo = Path("demo_media/sample_no_audio.mp4")
        a_res = asyncio.run(
            global_orchestrator.audio_adapter.analyze(no_audio_demo, "TL-2026-TEST")
        )
        assert a_res["module"] == "audio_ai"
        assert a_res["has_audio"] is False
        assert a_res["audio_score"] is None
        results["Audio API"] = True
    except Exception as e:
        print(f"[FAIL] Audio API: {e}")
        results["Audio API"] = False

    # 9. Fusion check
    try:
        fusion_out = global_orchestrator.fusion_adapter.fuse(
            evidence_id="TL-2026-TEST",
            visual_res={"visual_score": 0.94, "frequency_score": 0.89},
            temporal_res={"temporal_score": 0.87},
            audio_res={"audio_score": 0.76}
        )
        assert fusion_out["verdict"] == VerdictEnum.MANIPULATED
        assert fusion_out["fusion_score"] >= 0.85
        results["Fusion"] = True
    except Exception as e:
        print(f"[FAIL] Fusion: {e}")
        results["Fusion"] = False

    # 10. Evidence record & Tamper detection check
    try:
        res = client.get(f"/evidence/{evidence_id}")
        assert res.status_code == 200
        ev_data = res.json()
        assert ev_data["evidence_id"] == evidence_id
        assert ev_data["is_tampered"] is False
        results["Evidence record"] = True
    except Exception as e:
        print(f"[FAIL] Evidence record: {e}")
        results["Evidence record"] = False

    # 11. PDF report check
    try:
        with open(demo_file, "rb") as f:
            an_res = client.post("/analyze", files={"file": ("sample_fake.jpg", f, "image/jpeg")})
        assert an_res.status_code == 200
        an_data = an_res.json()
        new_ev_id = an_data["evidence_id"]

        rep_res = client.get(f"/report/{new_ev_id}")
        assert rep_res.status_code == 200
        assert rep_res.headers["content-type"] == "application/pdf"
        assert len(rep_res.content) > 1000
        results["PDF report"] = True
    except Exception as e:
        print(f"[FAIL] PDF report: {e}")
        results["PDF report"] = False

    # 12. Error handling check
    try:
        bad_res = client.get("/evidence/TL-NON-EXISTENT")
        assert bad_res.status_code == 404

        unsupported_res = client.post(
            "/upload",
            files={"file": ("malware.exe", b"malicious", "application/x-msdownload")}
        )
        assert unsupported_res.status_code == 400
        results["Error handling"] = True
    except Exception as e:
        print(f"[FAIL] Error handling: {e}")
        results["Error handling"] = False

    return results


def print_banner(results):
    print("\n====================================")
    print("       TRUTHLENS BACKEND TEST       ")
    print("====================================")
    all_passed = True
    for item, status_ok in results.items():
        symbol = "[OK]" if sys.platform == "win32" and sys.stdout.encoding != 'utf-8' else "✓"
        mark = symbol if status_ok else ("FAIL" if sys.platform == "win32" and sys.stdout.encoding != 'utf-8' else "✗")
        print(f"{item:<22} {mark}")
        if not status_ok:
            all_passed = False
    print("====================================")
    if all_passed:
        print("ALL BACKEND SYSTEMS READY\n")
    else:
        print("SOME TESTS FAILED - CHECK OUTPUT\n")
    return all_passed


def test_truthlens_all():
    init_db()
    results = run_all_checks()
    assert all(results.values())


if __name__ == "__main__":
    init_db()
    res = run_all_checks()
    passed = print_banner(res)
    sys.exit(0 if passed else 1)
