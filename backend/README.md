# TruthLens 🔍🛡️
**AI Digital Forensics & Deepfake Authentication Platform**

TruthLens is an AI-assisted digital forensics platform that analyzes images and videos for deepfake manipulation. Designed around three connected pillars:
1. **Deepfake X-Ray**: Localizes suspicious regions, facial boundaries, and anomalous frames using Grad-CAM heatmaps.
2. **Forensic Fusion**: Dynamically combines independent visual, frequency (spectral), temporal (frame consistency), and audio-visual synchronization signals into a calibrated verdict (`AUTHENTIC`, `MANIPULATED`, or `INCONCLUSIVE`).
3. **Evidence Guardian**: Preserves digital chain of custody with streaming SHA-256 cryptographic digests, unique Evidence IDs (`TL-2026-XXXX`), persistent forensic ledgers, and automated PDF forensic reports.

---

## 📁 Repository Structure

```text
TruthLens/backend/
├── main.py                  # FastAPI server entry point, CORS & static mounts
├── config.py                # System settings, storage paths & decision thresholds
├── schemas/
│   └── contracts.py         # Frozen Pydantic schemas (P1, P2, P3, P4)
├── database/
│   ├── db.py                # SQLite database connection & session
│   └── models.py            # EvidenceRecord SQLAlchemy model
├── services/
│   ├── hasher.py            # Streaming SHA-256 & tamper verification
│   ├── id_generator.py      # Standardized Evidence ID generator (TL-2026-XXXX)
│   ├── report_generator.py  # ReportLab Forensic PDF Report compiler
│   └── orchestrator.py      # Multi-modal AI orchestrator & error isolation
├── ai_adapters/
│   ├── base.py              # Abstract Base AI Adapter
│   ├── visual_adapter.py    # Person 1: Visual AI & Grad-CAM Heatmaps
│   ├── temporal_adapter.py  # Person 2A: Temporal Consistency & Optical Flow
│   ├── audio_adapter.py     # Person 2B: Audio Forensic & AV-Sync (No-Audio safe)
│   └── fusion_adapter.py    # Person 2B: Calibrated Dynamic Fusion
├── routes/
│   ├── upload.py            # POST /upload, POST /analyze
│   ├── results.py           # GET /result/{evidence_id}
│   ├── evidence.py          # GET /evidence/{evidence_id}, GET /evidence
│   ├── reports.py           # GET /report/{evidence_id}
│   └── health.py            # GET /health
├── contracts/               # Frozen JSON contracts for team alignment
├── demo_media/              # Offline demonstration test pack
├── storage/                 # Uploads, heatmaps, reports, and SQLite database
├── tests/
│   ├── test_truthlens.py    # Master Person 3 pre-hackathon verification suite
│   ├── test_api.py          # API endpoint tests
│   ├── test_hasher.py       # Cryptographic SHA-256 tests
│   └── test_orchestrator.py # Multi-modal pipeline tests
├── TRUTHLENS_INTEGRATION.md # Teammate integration manual (P1, P2, P4)
└── requirements.txt         # Pinned project dependencies
```

---

## 🐧 Running in WSL (Windows Subsystem for Linux)

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Backend Server
Run Uvicorn with `--host 0.0.0.0` so it is accessible from host browser:

```bash
fuser -k 8000/tcp
# From workspace root (/home/sharvesh/inno_2025)
cd /home/<username>/inno_2025
source backend/venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Windows Host Access**: automatically forwards ports to `localhost`. You can open these directly in your browser:
> - **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
> - **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
> - **Health Status Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Testing & Verification

### Run Full Test Suite (`pytest`)
```bash
cd backend
source venv/bin/activate
pytest -v
```

---


## 🔒 Evidence Guardian & Forensic API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | System health and AI module readiness status |
| `/upload` | `POST` | Ingest media, compute SHA-256, create Evidence record |
| `/analyze` | `POST` | Multi-modal AI analysis, fusion, database store, and report generation |
| `/result/{id}` | `GET` | Retrieve structured forensic analysis result JSON |
| `/evidence/{id}` | `GET` | Evidence Passport and live cryptographic tamper check |
| `/evidence` | `GET` | List all tracked digital evidence items |
| `/report/{id}` | `GET` | Download official Forensic Analysis PDF Report |

---

## 📡 Example API Invocations (cURL)

**1. Health Check:**
```bash
curl --noproxy '*' http://127.0.0.1:8000/health
```

**2. Analyze an Image:**
```bash
curl --noproxy '*' -X POST \
  -F "file=@demo_media/sample_fake.jpg" \
  http://127.0.0.1:8000/analyze
```

**3. Verify Evidence Integrity (Tamper Detection):**
```bash
curl --noproxy '*' http://127.0.0.1:8000/evidence/TL-2026-0001
```

**4. Download Forensic PDF Report:**
```bash
curl --noproxy '*' -OJ http://127.0.0.1:8000/report/TL-2026-0001
```
# deepfake-analysis
