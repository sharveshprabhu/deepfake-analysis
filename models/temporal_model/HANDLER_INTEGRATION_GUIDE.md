# 📖 TruthLens AI — Temporal & Video AI Handler Integration Guide

This guide provides instructions for the TruthLens Backend Engineer / Orchestrator on how to import, configure, and invoke the **Person 2A Temporal AI** and **Person 2B Audio/AV-Sync AI** detection modules.

---

## 📂 Deployment Package Layout

```text
truthlens_temporal_release/
├── checkpoints/
│   └── best_calibrated_model.pth    # Calibrated PyTorch weights (Swin-T + Temporal Transformer)
├── config/
│   └── model_config.yaml            # Inference configuration & hyperparameters
├── inference/
│   ├── truthlens_adapter.py         # Official Person 2A & 2B adapter entrypoints
│   ├── video_inference.py           # Multi-window sliding video inference engine
│   └── timestamp_detector.py        # Inter-frame transition & timestamp localizer
├── models/
│   ├── visual_encoder.py            # Swin-T spatial frame feature extractor
│   ├── temporal_model.py            # Temporal Multi-Head Attention transformer
│   ├── av_sync_model.py             # Audio-Visual speech synchrony module
│   ├── object_temporal_model.py     # Trajectory consistency module
│   └── fusion_model.py              # Calibrated decision fusion engine
├── utils/
│   └── config.py                    # YAML configuration parser
├── verify_deployment.py             # 1-Click validation script
└── requirements.txt                 # Minimal inference runtime dependencies
```

---

## ⚡ 1. Direct Python Integration (Recommended)

To use the model directly within the TruthLens backend service or orchestrator:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Import and Call the Async Adapter
```python
import asyncio
from inference.truthlens_adapter import my_temporal_model, my_audio_model

async def analyze_evidence():
    evidence_id = "TL-2026-0001"
    video_path = "path/to/uploaded_evidence.mp4"

    # Person 2A: Temporal Deepfake Analysis
    temporal_result = await my_temporal_model(
        file_path=video_path,
        evidence_id=evidence_id
    )
    print("Person 2A Result:", temporal_result)

    # Person 2B: Audio-Visual Sync Analysis
    audio_result = await my_audio_model(
        file_path=video_path,
        evidence_id=evidence_id
    )
    print("Person 2B Result:", audio_result)

if __name__ == "__main__":
    asyncio.run(analyze_evidence())
```

---

## 📋 2. Strict JSON Contract Specifications

### 2A. Temporal AI Contract (`my_temporal_model`)

```json
{
  "module": "temporal_ai",
  "evidence_id": "TL-2026-0001",
  "temporal_score": 0.87,
  "suspicious_frame_transitions": [
    {
      "from_frame": 13,
      "to_frame": 14,
      "discontinuity_score": 0.88,
      "type": "jitter",
      "timestamp_seconds": 0.43
    }
  ],
  "explanations": [
    "High inter-frame temporal discontinuity observed (87.0%)",
    "Suspicious landmark trajectory jump detected between frame 13 and 14"
  ],
  "status": "SUCCESS"
}
```

#### Fields Description:
* `module` (*str*): Always `"temporal_ai"`.
* `evidence_id` (*str*): Passthrough ID matching the incoming request.
* `temporal_score` (*float | None*): Manipulation probability between `0.0` (authentic) and `1.0` (deepfake). Returns `None` if skipped.
* `suspicious_frame_transitions` (*list[dict]*): Specific frame transitions where high temporal discontinuity or warping occurred.
* `explanations` (*list[str]*): Human-readable forensic insights for the investigative report.
* `status` (*str*): `"SUCCESS"` on completed analysis, `"SKIPPED"` on static images, or `"ERROR"` on failure.

---

### 2B. Audio / AV-Sync AI Contract (`my_audio_model`)

```json
{
  "module": "audio_ai",
  "evidence_id": "TL-2026-0001",
  "audio_score": 0.76,
  "av_sync_offset_ms": 120.0,
  "suspicious_audio_segments": [
    {
      "start_sec": 1.2,
      "end_sec": 2.5,
      "sync_error_score": 0.82
    }
  ],
  "explanations": [
    "Audio-visual desynchronization detected: 120ms offset between lip motion and voice onset"
  ],
  "status": "SUCCESS"
}
```

---

## 🖼️ 3. Edge Case Handling

1. **Static Images (`.jpg`, `.png`, `.webp`, `.bmp`)**:
   * The temporal model **automatically detects static images** and skips processing to prevent invalid temporal calculations.
   * Return Value:
     ```json
     {
       "module": "temporal_ai",
       "evidence_id": "TL-2026-0002",
       "temporal_score": null,
       "suspicious_frame_transitions": [],
       "explanations": [
         "Static image provided; temporal sequence analysis was skipped."
       ],
       "status": "SKIPPED"
     }
     ```
2. **Missing Video or Corrupt Stream**:
   * Returns `status: "ERROR"` with `temporal_score: null` and the error reason in `explanations`.
3. **Non-Blocking Asynchronous Execution**:
   * Heavy deep learning inference runs in a worker thread (`asyncio.to_thread`) so the FastAPI / async orchestrator event loop is never blocked.

---

## 🚀 4. FastAPI Microservice Integration (Optional)

If running the Temporal Model as an independent HTTP microservice:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference.truthlens_adapter import my_temporal_model

app = FastAPI(title="TruthLens Temporal AI Microservice", version="1.0.0")

class AnalyzeRequest(BaseModel):
    file_path: str
    evidence_id: str

@app.post("/analyze/temporal")
async def analyze_temporal_endpoint(req: AnalyzeRequest):
    try:
        result = await my_temporal_model(req.file_path, req.evidence_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 🧪 5. How to Verify Deployment

Run the included verification script from this directory:
```bash
python verify_deployment.py
```
Expected Output:
```text
[PASS] Calibrated Checkpoint found: .../checkpoints/best_calibrated_model.pth (169.4 MB)
[*] Testing Person 2A (Static Image Edge Case)...
    Status: SKIPPED | Temporal Score: None
    [PASS] Static image correctly skipped according to TruthLens contract.
[*] Testing Person 2B Audio / AV-Sync Adapter...
    [PASS] Person 2B adapter structure verified.
[SUCCESS] ALL DEPLOYMENT VERIFICATION CHECKS PASSED!
```
