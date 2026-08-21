import asyncio
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Union, Optional
from sqlalchemy.orm import Session

from backend.ai_adapters.visual_adapter import VisualAIAdapter
from backend.ai_adapters.temporal_adapter import TemporalAIAdapter
from backend.ai_adapters.audio_adapter import AudioAIAdapter
from backend.ai_adapters.fusion_adapter import ForensicFusionAdapter
from backend.services.hasher import calculate_sha256_file
from backend.services.report_generator import generate_pdf_report
from backend.database.models import EvidenceRecord
from backend.config import MODEL_VERSION


class Orchestrator:
    """
    Central Nervous System of TruthLens (Person 3 Core).
    Coordinates ingestion, hashing, multi-AI execution, fusion, database persistence,
    and forensic report generation.
    """

    def __init__(self):
        self.visual_adapter = VisualAIAdapter()
        self.temporal_adapter = TemporalAIAdapter()
        self.audio_adapter = AudioAIAdapter()
        self.fusion_adapter = ForensicFusionAdapter()

    async def run_pipeline(
        self,
        file_path: Union[str, Path],
        evidence_id: str,
        filename: str,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Executes full multimodal forensic analysis with error isolation and graceful degradation.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Target media file not found: {file_path}")

        # 1. Compute SHA-256 Fingerprint
        sha256_hash = calculate_sha256_file(path)

        # 2. Execute AI Modules concurrently with Exception Isolation
        visual_task = asyncio.create_task(self.visual_adapter.analyze(path, evidence_id))
        temporal_task = asyncio.create_task(self.temporal_adapter.analyze(path, evidence_id))
        audio_task = asyncio.create_task(self.audio_adapter.analyze(path, evidence_id))

        results = await asyncio.gather(visual_task, temporal_task, audio_task, return_exceptions=True)

        # Safe fallback handling if any module threw an unexpected exception
        visual_res = results[0] if not isinstance(results[0], Exception) else {
            "module": "visual_ai",
            "evidence_id": evidence_id,
            "visual_score": 0.50,
            "frequency_score": 0.50,
            "suspicious_frames": [],
            "regions": [],
            "heatmap_filename": None,
            "explanations": ["Visual module encountered an unexpected error; signal defaulted."],
            "status": "ERROR"
        }

        temporal_res = results[1] if not isinstance(results[1], Exception) else {
            "module": "temporal_ai",
            "evidence_id": evidence_id,
            "temporal_score": None,
            "suspicious_frame_transitions": [],
            "explanations": ["Temporal module bypassed or unavailable."],
            "status": "ERROR"
        }

        audio_res = results[2] if not isinstance(results[2], Exception) else {
            "module": "audio_ai",
            "evidence_id": evidence_id,
            "audio_score": None,
            "has_audio": False,
            "av_sync_offset_ms": None,
            "acoustic_artifact_score": None,
            "explanations": ["Audio module bypassed or unavailable."],
            "status": "ERROR"
        }

        # 3. Fuse Multi-modal Signals
        fusion_res = self.fusion_adapter.fuse(
            evidence_id=evidence_id,
            visual_res=visual_res,
            temporal_res=temporal_res,
            audio_res=audio_res
        )

        # 4. Consolidate Explanations & Suspicious Frames
        all_explanations = []
        all_explanations.extend(visual_res.get("explanations", []))
        all_explanations.extend(temporal_res.get("explanations", []))
        all_explanations.extend(audio_res.get("explanations", []))
        if fusion_res.get("verdict_reasoning"):
            all_explanations.insert(0, fusion_res["verdict_reasoning"])

        suspicious_frames = visual_res.get("suspicious_frames", [])
        regions = visual_res.get("regions", [])
        heatmap_filename = visual_res.get("heatmap_filename")
        heatmap_url = f"/static/heatmaps/{heatmap_filename}" if heatmap_filename else None

        analyzed_at = datetime.now(timezone.utc)

        # 5. Generate Forensic PDF Report
        try:
            report_path = generate_pdf_report(
                evidence_id=evidence_id,
                filename=filename,
                sha256_hash=sha256_hash,
                verdict=fusion_res["verdict"].value if hasattr(fusion_res["verdict"], "value") else str(fusion_res["verdict"]),
                confidence=fusion_res["confidence"],
                fusion_score=fusion_res["fusion_score"],
                visual_score=visual_res.get("visual_score"),
                frequency_score=visual_res.get("frequency_score"),
                temporal_score=temporal_res.get("temporal_score"),
                audio_score=audio_res.get("audio_score"),
                suspicious_frames=suspicious_frames,
                explanations=all_explanations,
                analyzed_at=analyzed_at
            )
            report_url = f"/report/{evidence_id}"
        except Exception as e:
            report_path = None
            report_url = None

        # 6. Database Persistence
        if db is not None:
            try:
                record = db.query(EvidenceRecord).filter(EvidenceRecord.evidence_id == evidence_id).first()
                if not record:
                    record = EvidenceRecord(
                        evidence_id=evidence_id,
                        filename=filename,
                        file_path=str(path),
                        file_size_bytes=os.path.getsize(path),
                        sha256=sha256_hash
                    )
                    db.add(record)

                record.analyzed_at = analyzed_at
                record.verdict = fusion_res["verdict"].value if hasattr(fusion_res["verdict"], "value") else str(fusion_res["verdict"])
                record.confidence = fusion_res["confidence"]
                record.fusion_score = fusion_res["fusion_score"]
                record.visual_score = visual_res.get("visual_score")
                record.frequency_score = visual_res.get("frequency_score")
                record.temporal_score = temporal_res.get("temporal_score")
                record.audio_score = audio_res.get("audio_score")
                record.suspicious_frames = suspicious_frames
                record.regions = regions
                record.explanations = all_explanations
                record.heatmap_path = str(heatmap_filename) if heatmap_filename else None
                record.report_path = str(report_path) if report_path else None
                record.model_version = MODEL_VERSION

                db.commit()
                db.refresh(record)
            except Exception as e:
                db.rollback()

        # 7. Build Standardized Clean Response for Frontend (Person 4)
        return {
            "evidence_id": evidence_id,
            "verdict": fusion_res["verdict"],
            "confidence": fusion_res["confidence"],
            "fusion_score": fusion_res["fusion_score"],
            "attack_vector": fusion_res.get("attack_vector", "UNKNOWN"),
            "visual_score": visual_res.get("visual_score"),
            "frequency_score": visual_res.get("frequency_score"),
            "temporal_score": temporal_res.get("temporal_score"),
            "audio_score": audio_res.get("audio_score"),
            "conflict_index": fusion_res.get("conflict_index", 0.0),
            "uncertainty_index": fusion_res.get("uncertainty_index", 0.0),
            "suspicious_frames": suspicious_frames,
            "regions": regions,
            "explanations": all_explanations,
            "sha256": sha256_hash,
            "heatmap_url": heatmap_url,
            "report_url": report_url,
            "model_version": MODEL_VERSION,
            "created_at": analyzed_at
        }


# Global singleton orchestrator instance
global_orchestrator = Orchestrator()
