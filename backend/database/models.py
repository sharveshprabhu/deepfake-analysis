import json
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(String(64), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    mime_type = Column(String(128), nullable=False, default="application/octet-stream")
    media_type = Column(String(32), nullable=False, default="video")
    sha256 = Column(String(64), nullable=False, index=True)
    
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    analyzed_at = Column(DateTime, nullable=True)
    
    # Forensic Verdicts & Scores
    verdict = Column(String(32), nullable=True)  # AUTHENTIC, MANIPULATED, INCONCLUSIVE
    confidence = Column(Float, nullable=True)
    fusion_score = Column(Float, nullable=True)
    visual_score = Column(Float, nullable=True)
    frequency_score = Column(Float, nullable=True)
    temporal_score = Column(Float, nullable=True)
    audio_score = Column(Float, nullable=True)
    
    # Serialized Structured Forensic Data
    suspicious_frames_json = Column(Text, nullable=True, default="[]")
    regions_json = Column(Text, nullable=True, default="[]")
    explanations_json = Column(Text, nullable=True, default="[]")
    
    # Artifact file paths
    heatmap_path = Column(String(512), nullable=True)
    report_path = Column(String(512), nullable=True)
    model_version = Column(String(64), default="TruthLens v1.0")

    @property
    def suspicious_frames(self):
        try:
            return json.loads(self.suspicious_frames_json or "[]")
        except Exception:
            return []

    @suspicious_frames.setter
    def suspicious_frames(self, value):
        self.suspicious_frames_json = json.dumps(value or [])

    @property
    def regions(self):
        try:
            return json.loads(self.regions_json or "[]")
        except Exception:
            return []

    @regions.setter
    def regions(self, value):
        self.regions_json = json.dumps(value or [])

    @property
    def explanations(self):
        try:
            return json.loads(self.explanations_json or "[]")
        except Exception:
            return []

    @explanations.setter
    def explanations(self, value):
        self.explanations_json = json.dumps(value or [])
