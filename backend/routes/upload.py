import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.config import UPLOADS_DIR, ALLOWED_EXTS, MAX_UPLOAD_BYTES
from backend.database.db import get_db
from backend.database.models import EvidenceRecord
from backend.services.hasher import calculate_sha256_file
from backend.services.id_generator import generate_evidence_id
from backend.services.orchestrator import global_orchestrator
from backend.schemas.contracts import MediaUploadResponse, FinalAnalysisResponse

router = APIRouter(tags=["Ingestion & Analysis"])


@router.post("/upload", response_model=MediaUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_media(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Ingests and securely seals media:
    1. Validates format and size.
    2. Saves file to isolated storage.
    3. Calculates cryptographic SHA-256 fingerprint.
    4. Generates unique Evidence ID.
    5. Creates Evidence Guardian record in SQLite.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided in upload.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTS))}"
        )

    evidence_id = generate_evidence_id(db)
    safe_filename = f"{evidence_id}_{Path(file.filename).name}"
    save_path = UPLOADS_DIR / safe_filename

    # Stream file to disk to preserve memory
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {str(e)}")

    file_size = os.path.getsize(save_path)
    if file_size > MAX_UPLOAD_BYTES:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024*1024)} MB).")

    # Cryptographic SHA-256
    sha256_hash = calculate_sha256_file(save_path)
    media_type = "image" if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else "video"
    uploaded_at = datetime.now(timezone.utc)

    # Persist in SQLite
    record = EvidenceRecord(
        evidence_id=evidence_id,
        filename=file.filename,
        file_path=str(save_path),
        file_size_bytes=file_size,
        mime_type=file.content_type or "application/octet-stream",
        media_type=media_type,
        sha256=sha256_hash,
        uploaded_at=uploaded_at
    )
    db.add(record)
    db.commit()

    return MediaUploadResponse(
        evidence_id=evidence_id,
        filename=file.filename,
        file_size_bytes=file_size,
        sha256=sha256_hash,
        media_type=media_type,
        uploaded_at=uploaded_at,
        message="Media successfully registered and sealed in Evidence Guardian."
    )


@router.post("/analyze", response_model=FinalAnalysisResponse)
async def analyze_media(
    file: UploadFile = File(None),
    evidence_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Main TruthLens Analysis Endpoint:
    Accepts direct file upload OR an existing evidence_id,
    runs the full multi-signal AI pipeline (Visual, Temporal, Audio, Fusion),
    generates the forensic report, and returns the frozen JSON contract response.
    """
    if file is not None and getattr(file, "filename", None):
        # Step 1: Upload & Seal
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'.")

        target_evidence_id = generate_evidence_id(db)
        safe_filename = f"{target_evidence_id}_{Path(file.filename).name}"
        save_path = UPLOADS_DIR / safe_filename

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(save_path)
        sha256_hash = calculate_sha256_file(save_path)
        media_type = "image" if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else "video"

        record = EvidenceRecord(
            evidence_id=target_evidence_id,
            filename=file.filename,
            file_path=str(save_path),
            file_size_bytes=file_size,
            mime_type=file.content_type or "application/octet-stream",
            media_type=media_type,
            sha256=sha256_hash,
            uploaded_at=datetime.now(timezone.utc)
        )
        db.add(record)
        db.commit()

        # Step 2: Orchestrate Analysis
        result = await global_orchestrator.run_pipeline(
            file_path=save_path,
            evidence_id=target_evidence_id,
            filename=file.filename,
            db=db
        )
        return result

    elif evidence_id:
        record = db.query(EvidenceRecord).filter(EvidenceRecord.evidence_id == evidence_id).first()
        if not record:
            raise HTTPException(status_code=404, detail=f"Evidence ID '{evidence_id}' not found.")

        result = await global_orchestrator.run_pipeline(
            file_path=record.file_path,
            evidence_id=record.evidence_id,
            filename=record.filename,
            db=db
        )
        return result

    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either a media file upload or an existing evidence_id."
        )
