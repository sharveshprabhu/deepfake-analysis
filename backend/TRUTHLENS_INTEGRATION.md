# TRUTHLENS INTEGRATION MANUAL
**AI Digital Forensics & Deepfake Authentication Platform**
*Target: 36-Hour Hackathon | Prepared by Person 3 (Backend + Orchestrator + Evidence Guardian)*

---

## 1. Quick Start

### Start the Backend Server
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start FastAPI development server
uvicorn backend.main:app --reload --port 8000
```
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Root Health Status**: [http://localhost:8000/health](http://localhost:8000/health)

### Run Automated Backend Verification
```bash
python tests/test_truthlens.py
```
Expected output:
```text
====================================
       TRUTHLENS BACKEND TEST       
====================================
Server                 ✓
Upload API             ✓
File storage           ✓
SHA-256                ✓
Database               ✓
Visual API             ✓
Temporal API           ✓
Audio API              ✓
Fusion                 ✓
Evidence record        ✓
PDF report             ✓
Error handling         ✓
====================================
ALL BACKEND SYSTEMS READY
```

---

## 2. Team Architecture & Flow

```text
                                  FRONTEND (Person 4)
                                          │
                                 POST /analyze or /upload
                                          │
                                          ▼
                               FASTAPI BACKEND (Person 3)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         EVIDENCE GUARDIAN                                  ORCHESTRATOR
         - SHA-256 Hash Digest                              - Signal Isolation & Timeouts
         - Evidence ID (TL-2026-XXXX)                       - Graceful Degradation (no audio/face)
         - SQLite Persistent DB                             - Modular AI Adapter Pipeline
                  │                                               │
                  │                      ┌────────────────────────┼────────────────────────┐
                  │                      ▼                        ▼                        ▼
                  │                 VISUAL AI               TEMPORAL AI                 AUDIO AI
                  │              (Person 1 / Mock)      (Person 2A / Mock)         (Person 2B / Mock)
                  │                      │                        │                        │
                  │                      └────────────────────────┼────────────────────────┘
                  │                                               ▼
                  │                                        FUSION MODULE
                  │                                     (Person 2B / Mock)
                  │                                               │
                  ▼                                               ▼
           SQLITE DATABASE ◀───────────────────────────── COMBINED RESULT
                  │                                               │
                  ▼                                               ▼
         FORENSIC PDF REPORT                             JSON RESPONSE TO UI
         (ReportLab / FPDF2)                           (Shared Frozen Contract)
```

---

## 3. Frozen API Endpoints

### 1. Ingest & Seal Media (`POST /upload`)
- **URL**: `/upload`
- **Method**: `POST`
- **Payload**: `multipart/form-data` with `file: [binary]`
- **Response**:
```json
{
  "evidence_id": "TL-2026-0001",
  "filename": "suspect_video.mp4",
  "file_size_bytes": 1420840,
  "sha256": "8a91f4b5c7321e8d9047bca899014529fbe92a3489e27cbfd1283995819773c2",
  "media_type": "video",
  "uploaded_at": "2026-08-19T14:50:00Z",
  "message": "Media successfully registered and sealed in Evidence Guardian."
}
```

---

### 2. Full Forensic Analysis (`POST /analyze`)
- **URL**: `/analyze`
- **Method**: `POST`
- **Payload**: `multipart/form-data` with `file: [binary]` OR `evidence_id: "TL-2026-0001"`
- **Response (Shared Frozen Contract)**:
```json
{
  "evidence_id": "TL-2026-0001",
  "verdict": "MANIPULATED",
  "confidence": 0.937,
  "fusion_score": 0.912,
  "visual_score": 0.94,
  "frequency_score": 0.89,
  "temporal_score": 0.87,
  "audio_score": 0.76,
  "suspicious_frames": [14, 15, 16],
  "regions": [
    {
      "frame_index": 14,
      "box": [120, 85, 240, 210],
      "label": "facial_boundary_distortion",
      "anomaly_score": 0.95
    }
  ],
  "explanations": [
    "High-confidence manipulation detected across 4 forensic signals (Score: 91.2%)",
    "Facial boundary blending artifacts detected in frames 14-16",
    "High-frequency discrete cosine transform anomaly in cheek and jaw regions (89%)",
    "Inter-frame landmark trajectory jitter observed between frame 13 and 14",
    "Audio-to-visual phoneme/viseme timing lag of 142.5ms detected"
  ],
  "sha256": "8a91f4b5c7321e8d9047bca899014529fbe92a3489e27cbfd1283995819773c2",
  "heatmap_url": "/static/heatmaps/TL-2026-0001_heatmap.png",
  "report_url": "/report/TL-2026-0001",
  "model_version": "TruthLens v1.0",
  "created_at": "2026-08-19T14:50:00Z"
}
```

---

### 3. Fetch Stored Analysis Result (`GET /result/{evidence_id}`)
- **URL**: `/result/TL-2026-0001`
- **Method**: `GET`
- **Response**: Same as `/analyze` response above.

---

### 4. Evidence Guardian Passport & Tamper Check (`GET /evidence/{evidence_id}`)
- **URL**: `/evidence/TL-2026-0001`
- **Method**: `GET`
- **Response**:
```json
{
  "evidence_id": "TL-2026-0001",
  "filename": "suspect_video.mp4",
  "file_size_bytes": 1420840,
  "mime_type": "video/mp4",
  "sha256": "8a91f4b5c7321e8d9047bca899014529fbe92a3489e27cbfd1283995819773c2",
  "is_tampered": false,
  "uploaded_at": "2026-08-19T14:50:00Z",
  "verdict": "MANIPULATED",
  "confidence": 0.937,
  "fusion_score": 0.912,
  "visual_score": 0.94,
  "frequency_score": 0.89,
  "temporal_score": 0.87,
  "audio_score": 0.76,
  "suspicious_frames": [14, 15, 16],
  "explanations": ["..."],
  "heatmap_url": "/static/heatmaps/TL-2026-0001_heatmap.png",
  "report_url": "/report/TL-2026-0001",
  "model_version": "TruthLens v1.0"
}
```

---

### 5. Download Forensic PDF Report (`GET /report/{evidence_id}`)
- **URL**: `/report/TL-2026-0001`
- **Method**: `GET`
- **Response**: Binary PDF file download (`application/pdf`) with complete evidence certificate, signal tables, and cryptographic seal.

---

## 4. How Teammates Plug In Live Models

Person 3's backend is built with **plug-and-play AI adapters**. You do NOT need to rewrite routes or database code to connect your models!

### Person 1 (Visual AI + Grad-CAM Heatmaps)
In your module or `backend/ai_adapters/visual_adapter.py`:
```python
from backend.services.orchestrator import global_orchestrator

# Define your live PyTorch inference function
async def my_pytorch_visual_model(media_path: str, evidence_id: str) -> dict:
    # 1. Run face detection & PyTorch model
    # 2. Save heatmap to storage/heatmaps/{evidence_id}_heatmap.png
    return {
        "module": "visual_ai",
        "evidence_id": evidence_id,
        "visual_score": 0.94,
        "frequency_score": 0.89,
        "suspicious_frames": [14, 15, 16],
        "regions": [{"frame_index": 14, "box": [120, 85, 240, 210], "label": "boundary_distortion", "anomaly_score": 0.95}],
        "heatmap_filename": f"{evidence_id}_heatmap.png",
        "explanations": ["Facial boundary blending artifacts detected in frames 14-16"],
        "status": "SUCCESS"
    }

# Register with backend orchestrator
global_orchestrator.visual_adapter.set_real_model(my_pytorch_visual_model)
```

### Person 2A (Temporal AI)
```python
from backend.services.orchestrator import global_orchestrator

async def my_temporal_model(video_path: str, evidence_id: str) -> dict:
    return {
        "module": "temporal_ai",
        "evidence_id": evidence_id,
        "temporal_score": 0.87,
        "suspicious_frame_transitions": [{"from_frame": 13, "to_frame": 14, "discontinuity_score": 0.88, "type": "jitter"}],
        "explanations": ["Inter-frame landmark trajectory jitter observed"],
        "status": "SUCCESS"
    }

global_orchestrator.temporal_adapter.set_real_model(my_temporal_model)
```

### Person 2B (Audio AI & Fusion)
```python
from backend.services.orchestrator import global_orchestrator

async def my_audio_model(video_path: str, evidence_id: str) -> dict:
    # Note: If no audio is detected, return audio_score: None, has_audio: False
    return {
        "module": "audio_ai",
        "evidence_id": evidence_id,
        "audio_score": 0.76,
        "has_audio": True,
        "av_sync_offset_ms": 142.5,
        "acoustic_artifact_score": 0.74,
        "explanations": ["AV phoneme-viseme lag detected"],
        "status": "SUCCESS"
    }

global_orchestrator.audio_adapter.set_real_model(my_audio_model)
```

---

## 5. Frontend Integration Notes (Person 4)

- **CORS is enabled (`*`)**: Connect directly from `localhost:3000`, `localhost:5173`, etc.
- **Single Request Flow**: Frontend only needs to send `POST /analyze` with the video/image file; the backend immediately returns the complete JSON response, heatmap URL, and PDF download link.
- **Static Assets**:
  - Heatmap image: `http://localhost:8000/static/heatmaps/{evidence_id}_heatmap.png`
  - PDF Report: `http://localhost:8000/report/{evidence_id}`
- **Edge Cases Handled by Backend**:
  - Images (static JPG/PNG) bypass temporal/audio automatically without error.
  - Silent videos return `audio_score: null` without crashing the UI.
  - Uncertain/disagreeing signals automatically return `verdict: "INCONCLUSIVE"`.
