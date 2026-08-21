from typing import List
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import EvidenceRecord
from backend.services.hasher import verify_file_integrity
from backend.schemas.contracts import EvidenceRecordSchema, VerdictEnum

router = APIRouter(tags=["Evidence Guardian"])


@router.get("/evidence/{evidence_id}", response_model=EvidenceRecordSchema)
async def get_evidence_passport(
    evidence_id: str,
    db: Session = Depends(get_db)
):
    """
    Evidence Guardian Passport:
    Retrieves full audit trail for the media item and performs a live
    cryptographic SHA-256 hash check to verify that the file remains 100% untampered.
    """
    record = db.query(EvidenceRecord).filter(EvidenceRecord.evidence_id == evidence_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Evidence ID '{evidence_id}' not found.")

    # Live cryptographic integrity verification
    file_path = Path(record.file_path)
    is_tampered = False
    if file_path.exists():
        is_tampered = not verify_file_integrity(file_path, record.sha256)
    else:
        is_tampered = True  # File missing on disk

    heatmap_url = f"/static/heatmaps/{record.heatmap_path}" if record.heatmap_path else None
    report_url = f"/report/{record.evidence_id}" if record.report_path else None

    return EvidenceRecordSchema(
        evidence_id=record.evidence_id,
        filename=record.filename,
        file_size_bytes=record.file_size_bytes,
        mime_type=record.mime_type,
        sha256=record.sha256,
        is_tampered=is_tampered,
        uploaded_at=record.uploaded_at,
        verdict=VerdictEnum(record.verdict) if record.verdict else None,
        confidence=record.confidence,
        fusion_score=record.fusion_score,
        visual_score=record.visual_score,
        frequency_score=record.frequency_score,
        temporal_score=record.temporal_score,
        audio_score=record.audio_score,
        suspicious_frames=record.suspicious_frames,
        explanations=record.explanations,
        heatmap_url=heatmap_url,
        report_url=report_url,
        model_version=record.model_version or "TruthLens v1.0"
    )


@router.get("/evidence", response_model=List[EvidenceRecordSchema])
async def list_all_evidence(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Lists all forensic evidence records currently tracked in the database.
    """
    records = db.query(EvidenceRecord).order_by(EvidenceRecord.uploaded_at.desc()).offset(offset).limit(limit).all()
    results = []
    for r in records:
        heatmap_url = f"/static/heatmaps/{r.heatmap_path}" if r.heatmap_path else None
        report_url = f"/report/{r.evidence_id}" if r.report_path else None
        results.append(
            EvidenceRecordSchema(
                evidence_id=r.evidence_id,
                filename=r.filename,
                file_size_bytes=r.file_size_bytes,
                mime_type=r.mime_type,
                sha256=r.sha256,
                is_tampered=False,
                uploaded_at=r.uploaded_at,
                verdict=VerdictEnum(r.verdict) if r.verdict else None,
                confidence=r.confidence,
                fusion_score=r.fusion_score,
                visual_score=r.visual_score,
                frequency_score=r.frequency_score,
                temporal_score=r.temporal_score,
                audio_score=r.audio_score,
                suspicious_frames=r.suspicious_frames,
                explanations=r.explanations,
                heatmap_url=heatmap_url,
                report_url=report_url,
                model_version=r.model_version or "TruthLens v1.0"
            )
        )
    return results
