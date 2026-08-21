import random
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.database.models import EvidenceRecord


def generate_evidence_id(db: Session = None) -> str:
    """
    Generates a unique TruthLens Evidence ID formatted as TL-YYYY-XXXX.
    Example: TL-2026-0001
    """
    year = datetime.now(timezone.utc).year
    
    if db is not None:
        try:
            count = db.query(EvidenceRecord).count()
            candidate = f"TL-{year}-{(count + 1):04d}"
            # Ensure uniqueness
            exists = db.query(EvidenceRecord).filter(EvidenceRecord.evidence_id == candidate).first()
            if not exists:
                return candidate
        except Exception:
            pass

    # Fallback to high-entropy random sequence if DB isn't available
    rand_num = random.randint(1000, 9999)
    return f"TL-{year}-{rand_num}"
