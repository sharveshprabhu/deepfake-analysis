from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class VerdictEnum(str, Enum):
    AUTHENTIC = "AUTHENTIC"
    MANIPULATED = "MANIPULATED"
    INCONCLUSIVE = "INCONCLUSIVE"


class MediaTypeEnum(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class SuspiciousRegion(BaseModel):
    frame_index: Optional[int] = None
    box: List[int] = Field(default_factory=list, description="[x1, y1, x2, y2] bounding box")
    label: str = "anomaly"
    anomaly_score: float = 0.0


class SuspiciousTransition(BaseModel):
    from_frame: int
    to_frame: int
    discontinuity_score: float
    type: str = "jitter"


# Module-Specific Schemas
class VisualAnalysisResult(BaseModel):
    module: str = "visual_ai"
    evidence_id: str
    visual_score: float = Field(ge=0.0, le=1.0)
    frequency_score: float = Field(ge=0.0, le=1.0)
    suspicious_frames: List[int] = Field(default_factory=list)
    regions: List[SuspiciousRegion] = Field(default_factory=list)
    heatmap_filename: Optional[str] = None
    explanations: List[str] = Field(default_factory=list)
    status: str = "SUCCESS"


class TemporalAnalysisResult(BaseModel):
    module: str = "temporal_ai"
    evidence_id: str
    temporal_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    suspicious_frame_transitions: List[SuspiciousTransition] = Field(default_factory=list)
    explanations: List[str] = Field(default_factory=list)
    status: str = "SUCCESS"


class AudioAnalysisResult(BaseModel):
    module: str = "audio_ai"
    evidence_id: str
    audio_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    has_audio: bool = False
    av_sync_offset_ms: Optional[float] = None
    acoustic_artifact_score: Optional[float] = None
    explanations: List[str] = Field(default_factory=list)
    status: str = "SUCCESS"


class FusionAnalysisResult(BaseModel):
    module: str = "fusion"
    evidence_id: str
    fusion_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: VerdictEnum
    attack_vector: Optional[str] = "UNKNOWN"
    weights_used: Dict[str, float] = Field(default_factory=dict)
    conflict_index: Optional[float] = 0.0
    uncertainty_index: Optional[float] = 0.0
    modality_scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    verdict_reasoning: str = ""
    status: str = "SUCCESS"


# API Request/Response Schemas
class MediaUploadResponse(BaseModel):
    evidence_id: str
    filename: str
    file_size_bytes: int
    sha256: str
    media_type: str
    uploaded_at: datetime
    message: str = "File uploaded and hashed successfully. Ready for forensic analysis."


class FinalAnalysisResponse(BaseModel):
    evidence_id: str
    verdict: VerdictEnum
    confidence: float
    fusion_score: float
    attack_vector: Optional[str] = "UNKNOWN"
    visual_score: Optional[float] = None
    frequency_score: Optional[float] = None
    temporal_score: Optional[float] = None
    audio_score: Optional[float] = None
    conflict_index: Optional[float] = 0.0
    uncertainty_index: Optional[float] = 0.0
    suspicious_frames: List[int] = Field(default_factory=list)
    regions: List[SuspiciousRegion] = Field(default_factory=list)
    explanations: List[str] = Field(default_factory=list)
    sha256: str
    heatmap_url: Optional[str] = None
    report_url: Optional[str] = None
    model_version: str = "TruthLens v1.0"
    created_at: datetime


class EvidenceRecordSchema(BaseModel):
    evidence_id: str
    filename: str
    file_size_bytes: int
    mime_type: str
    sha256: str
    is_tampered: bool = False
    uploaded_at: datetime
    verdict: Optional[VerdictEnum] = None
    confidence: Optional[float] = None
    fusion_score: Optional[float] = None
    visual_score: Optional[float] = None
    frequency_score: Optional[float] = None
    temporal_score: Optional[float] = None
    audio_score: Optional[float] = None
    suspicious_frames: List[int] = Field(default_factory=list)
    explanations: List[str] = Field(default_factory=list)
    heatmap_url: Optional[str] = None
    report_url: Optional[str] = None
    model_version: str = "TruthLens v1.0"


class HealthCheckResponse(BaseModel):
    status: str
    system: str
    version: str
    database_connected: bool
    ai_modules: Dict[str, str]
    timestamp: datetime
