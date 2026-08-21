from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import EvidenceRecord
from backend.services.report_generator import generate_pdf_report
from backend.config import REPORTS_DIR

router = APIRouter(tags=["Forensic Reports"])


@router.get("/report/{evidence_id}")
async def download_forensic_report(
    evidence_id: str,
    db: Session = Depends(get_db)
):
    """
    Downloads or views the official PDF Forensic Analysis Report.
    If not yet generated, compiles it on-demand from the stored evidence record.
    """
    record = db.query(EvidenceRecord).filter(EvidenceRecord.evidence_id == evidence_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Evidence record '{evidence_id}' not found.")

    if not record.verdict:
        raise HTTPException(
            status_code=400,
            detail=f"Evidence '{evidence_id}' has not been analyzed yet. Run POST /analyze first."
        )

    report_path = Path(record.report_path) if record.report_path else REPORTS_DIR / f"{evidence_id}_Forensic_Report.pdf"

    if not report_path.exists():
        # Re-generate report on demand
        try:
            report_path = generate_pdf_report(
                evidence_id=record.evidence_id,
                filename=record.filename,
                sha256_hash=record.sha256,
                verdict=record.verdict,
                confidence=record.confidence or 0.5,
                fusion_score=record.fusion_score or 0.5,
                visual_score=record.visual_score,
                frequency_score=record.frequency_score,
                temporal_score=record.temporal_score,
                audio_score=record.audio_score,
                suspicious_frames=record.suspicious_frames,
                explanations=record.explanations,
                analyzed_at=record.analyzed_at
            )
            record.report_path = str(report_path)
            db.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate forensic PDF report: {str(e)}")

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=f"TruthLens_Report_{evidence_id}.pdf"
    )
