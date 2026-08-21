from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.database.db import get_db
from backend.schemas.contracts import HealthCheckResponse
from backend.config import MODEL_VERSION, SYSTEM_NAME
from backend.model_registry import DIAGNOSTIC_STATE
from backend.services.frequency_srm_worker import global_frequency_srm_worker

router = APIRouter(tags=["System Health & Diagnostics"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Returns the live status of the backend API, SQLite database, and AI modules.
    """
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    vis_stat = DIAGNOSTIC_STATE.get("visual", {}).get("status", "READY")
    temp_stat = DIAGNOSTIC_STATE.get("temporal", {}).get("status", "READY")
    aud_stat = DIAGNOSTIC_STATE.get("audio", {}).get("status", "READY")
    fuse_stat = DIAGNOSTIC_STATE.get("fusion", {}).get("status", "READY")

    # If any specific module explicitly FAILED, mark DEGRADED, otherwise HEALTHY
    any_failed = any(
        DIAGNOSTIC_STATE.get(m, {}).get("status") == "FAILED"
        for m in ["visual", "temporal", "audio", "fusion"]
    )
    is_healthy = db_connected and not any_failed

    return HealthCheckResponse(
        status="HEALTHY" if is_healthy else "DEGRADED",
        system=SYSTEM_NAME,
        version=MODEL_VERSION,
        database_connected=db_connected,
        ai_modules={
            "visual_ai": f"OPERATIONAL ({vis_stat})" if vis_stat == "PASSED" else f"STATUS: {vis_stat}",
            "temporal_ai": f"OPERATIONAL ({temp_stat})" if temp_stat == "PASSED" else f"STATUS: {temp_stat}",
            "audio_ai": f"OPERATIONAL ({aud_stat})" if aud_stat == "PASSED" else f"STATUS: {aud_stat}",
            "fusion": f"OPERATIONAL ({fuse_stat})" if fuse_stat == "PASSED" else f"STATUS: {fuse_stat}"
        },
        timestamp=datetime.now(timezone.utc)
    )


@router.get("/diagnostics/frequency-srm", tags=["System Health & Diagnostics"])
async def get_frequency_srm_audit():
    """
    Executes and returns the comprehensive Frequency and SRM noise inconsistency audit,
    explaining why normalized scores are strictly bounded < 1.0 while raw physical dispersion metrics exceed 1.0.
    """
    audit = global_frequency_srm_worker.run_comprehensive_audit()
    return audit

