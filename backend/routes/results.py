from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import EvidenceRecord
from backend.schemas.contracts import FinalAnalysisResponse, VerdictEnum

router = APIRouter(tags=["Forensic Results"])


@router.get("/result/{evidence_id}", response_model=FinalAnalysisResponse)
async def get_analysis_result(
    evidence_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieves the forensic analysis result conforming to the frozen shared JSON contract.
    """
    record = db.query(EvidenceRecord).filter(EvidenceRecord.evidence_id == evidence_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Evidence record '{evidence_id}' not found.")

    if not record.verdict:
        raise HTTPException(
            status_code=400,
            detail=f"Evidence '{evidence_id}' has not been analyzed yet. Run POST /analyze first."
        )

    heatmap_url = f"/static/heatmaps/{record.heatmap_path}" if record.heatmap_path else None
    report_url = f"/report/{record.evidence_id}" if record.report_path else None

    return FinalAnalysisResponse(
        evidence_id=record.evidence_id,
        verdict=VerdictEnum(record.verdict),
        confidence=record.confidence or 0.0,
        fusion_score=record.fusion_score or 0.0,
        visual_score=record.visual_score,
        frequency_score=record.frequency_score,
        temporal_score=record.temporal_score,
        audio_score=record.audio_score,
        suspicious_frames=record.suspicious_frames,
        regions=record.regions,
        explanations=record.explanations,
        sha256=record.sha256,
        heatmap_url=heatmap_url,
        report_url=report_url,
        model_version=record.model_version or "TruthLens v1.0",
        created_at=record.analyzed_at or record.uploaded_at
    )
