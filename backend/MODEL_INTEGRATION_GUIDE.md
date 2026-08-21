# 🤖 AI Model Integration Guide for TruthLens Backend

This guide is designed for **AI agents** and **developers** building AI model adapters (Visual, Temporal, Audio, and Fusion) for the **TruthLens AI Digital Forensics Platform**.

By following these instructions, your model will plug directly into the backend orchestrator without modifying core API routes or database models.

---

## 🛠️ Architecture Overview

The backend orchestrator (`backend.services.orchestrator.global_orchestrator`) calls each AI adapter asynchronously when an `/analyze` request is received.

```text
                  ┌──────────────────────────────────────────────┐
                  │          POST /analyze Request               │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │     Backend Orchestrator              │
                     └───────────────────┬───────────────────┘
                                         │
      ┌──────────────────────────┬───────┴──────────────────┬──────────────────────────┐
      ▼                          ▼                          ▼                          ▼
┌─────────────┐            ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│ Visual AI   │            │ Temporal AI │            │  Audio AI   │            │ Fusion AI   │
│ (Person 1)  │            │ (Person 2A) │            │ (Person 2B) │            │ (Person 2B) │
└──────┬──────┘            └──────┬──────┘            └──────┬──────┘            └──────┬──────┘
       │                          │                          │                          │
       └──────────────────────────┴───────────┬──────────────┴──────────────────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │ Final Consolidated Response │
                               └─────────────────────────────┘
```

---

## 📋 Integration Quick Reference

To register a live model with the backend:
1. Define an `async` or `sync` inference function matching your module's contract.
2. Import `global_orchestrator` from `backend.services.orchestrator`.
3. Call `set_real_model(...)` on your corresponding adapter.

---

## 1️⃣ Person 1: Visual AI & Grad-CAM Heatmap Adapter

### Function Signature & Requirements
- **Inputs**:
  - `file_path` (`str` or `Path`): Absolute path to the image or video file.
  - `evidence_id` (`str`): Unique identifier (e.g. `"TL-2026-0001"`).
- **Heatmap File**: If a heatmap is generated, save it to `storage/heatmaps/{evidence_id}_heatmap.png`. Return `"heatmap_filename": f"{evidence_id}_heatmap.png"`.

### Expected Return Dictionary / JSON Schema
```json
{
  "module": "visual_ai",
  "evidence_id": "TL-2026-0001",
  "visual_score": 0.94,
  "frequency_score": 0.89,
  "suspicious_frames": [14, 15, 16],
  "regions": [
    {
      "frame_index": 14,
      "box": [120, 85, 240, 210],
      "label": "facial_boundary_distortion",
      "anomaly_score": 0.95
    }
  ],
  "heatmap_filename": "TL-2026-0001_heatmap.png",
  "explanations": [
    "Facial boundary blending artifacts detected in frames 14-16",
    "High-frequency discrete cosine transform anomaly in cheek and jaw regions (89%)"
  ],
  "status": "SUCCESS"
}
```

### Registration Code
```python
from backend.services.orchestrator import global_orchestrator

async def my_visual_model(file_path: str, evidence_id: str) -> dict:
    # 1. Run inference (e.g. PyTorch / OpenCV)
    # 2. Generate and save heatmap to storage/heatmaps/{evidence_id}_heatmap.png
    return {
        "module": "visual_ai",
        "evidence_id": evidence_id,
        "visual_score": 0.94,
        "frequency_score": 0.89,
        "suspicious_frames": [14, 15, 16],
        "regions": [
            {
                "frame_index": 14,
                "box": [120, 85, 240, 210],
                "label": "facial_boundary_distortion",
                "anomaly_score": 0.95
            }
        ],
        "heatmap_filename": f"{evidence_id}_heatmap.png",
        "explanations": ["Facial boundary blending artifacts detected"],
        "status": "SUCCESS"
    }

# Connect model to backend orchestrator
global_orchestrator.visual_adapter.set_real_model(my_visual_model)
```

---

## 2️⃣ Person 2A: Temporal Consistency AI Adapter

### Function Signature & Requirements
- **Inputs**:
  - `file_path` (`str` or `Path`): Path to media file.
  - `evidence_id` (`str`): Unique evidence identifier.
- **Static Image Handling**: For still images, return `temporal_score: None` or `0.0` with `status: "SKIPPED"`.

### Expected Return Dictionary / JSON Schema
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
      "type": "jitter"
    }
  ],
  "explanations": [
    "Inter-frame landmark trajectory jitter observed between frame 13 and 14"
  ],
  "status": "SUCCESS"
}
```

### Registration Code
```python
from backend.services.orchestrator import global_orchestrator

async def my_temporal_model(file_path: str, evidence_id: str) -> dict:
    return {
        "module": "temporal_ai",
        "evidence_id": evidence_id,
        "temporal_score": 0.87,
        "suspicious_frame_transitions": [
            {
                "from_frame": 13,
                "to_frame": 14,
                "discontinuity_score": 0.88,
                "type": "jitter"
            }
        ],
        "explanations": ["Inter-frame landmark trajectory jitter observed"],
        "status": "SUCCESS"
    }

# Connect model to backend orchestrator
global_orchestrator.temporal_adapter.set_real_model(my_temporal_model)
```

---

## 3️⃣ Person 2B: Audio Forensic & AV-Sync Adapter

### Function Signature & Requirements
- **Inputs**:
  - `file_path` (`str` or `Path`): Path to media file.
  - `evidence_id` (`str`): Unique evidence identifier.
- **Silent Video / No Audio Handling**: If no audio track exists, set `has_audio: False` and `audio_score: None`. Do NOT throw an error!

### Expected Return Dictionary / JSON Schema
```json
{
  "module": "audio_ai",
  "evidence_id": "TL-2026-0001",
  "audio_score": 0.76,
  "has_audio": true,
  "av_sync_offset_ms": 142.5,
  "acoustic_artifact_score": 0.74,
  "explanations": [
    "Audio-to-visual phoneme/viseme timing lag of 142.5ms detected"
  ],
  "status": "SUCCESS"
}
```

### Registration Code
```python
from backend.services.orchestrator import global_orchestrator

async def my_audio_model(file_path: str, evidence_id: str) -> dict:
    return {
        "module": "audio_ai",
        "evidence_id": evidence_id,
        "audio_score": 0.76,
        "has_audio": True,
        "av_sync_offset_ms": 142.5,
        "acoustic_artifact_score": 0.74,
        "explanations": ["Audio-to-visual phoneme/viseme timing lag of 142.5ms detected"],
        "status": "SUCCESS"
    }

# Connect model to backend orchestrator
global_orchestrator.audio_adapter.set_real_model(my_audio_model)
```

---

## 4️⃣ Person 2B: Calibrated Fusion Adapter

### Function Signature & Requirements
- **Inputs**:
  - `visual_result` (`dict`): Result from visual adapter.
  - `temporal_result` (`dict`): Result from temporal adapter.
  - `audio_result` (`dict`): Result from audio adapter.
  - `evidence_id` (`str`): Unique evidence identifier.
- **Verdict Values**: Must be one of `"AUTHENTIC"`, `"MANIPULATED"`, or `"INCONCLUSIVE"`.
- **Score Range**: `fusion_score` and `confidence` must be floats between `0.0` and `1.0`.

### Expected Return Dictionary / JSON Schema
```json
{
  "module": "fusion",
  "evidence_id": "TL-2026-0001",
  "fusion_score": 0.912,
  "confidence": 0.937,
  "verdict": "MANIPULATED",
  "weights_used": {
    "visual": 0.4,
    "frequency": 0.2,
    "temporal": 0.2,
    "audio": 0.2
  },
  "verdict_reasoning": "Weighted signal score 0.912 exceeds manipulation threshold 0.65 with high cross-module agreement.",
  "status": "SUCCESS"
}
```

### Registration Code
```python
from backend.services.orchestrator import global_orchestrator

async def my_fusion_model(visual_res: dict, temporal_res: dict, audio_res: dict, evidence_id: str) -> dict:
    # Perform calibrated weighting / machine learning fusion
    return {
        "module": "fusion",
        "evidence_id": evidence_id,
        "fusion_score": 0.912,
        "confidence": 0.937,
        "verdict": "MANIPULATED",
        "weights_used": {"visual": 0.4, "frequency": 0.2, "temporal": 0.2, "audio": 0.2},
        "verdict_reasoning": "High confidence manipulation across visual and temporal signals.",
        "status": "SUCCESS"
    }

# Connect model to backend orchestrator
global_orchestrator.fusion_adapter.set_real_model(my_fusion_model)
```

---

## 5️⃣ 🌐 Distributed / Remote Model Integration (Running Models on Different Computers)

If model teammates are hosting their AI models on **different computers / GPU servers** (e.g., via FastAPI/Flask microservices), you can easily wrap the HTTP client call into the adapter registration:

```python
import httpx
from backend.services.orchestrator import global_orchestrator

# Example: Connecting a Visual AI model running on a remote GPU machine (http://192.168.1.50:5001)
async def remote_visual_model(file_path: str, evidence_id: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(file_path, "rb") as f:
            response = await client.post(
                "http://192.168.1.50:5001/predict",  # Teammate's computer IP
                files={"file": f},
                data={"evidence_id": evidence_id}
            )
    return response.json()

# Register the HTTP client wrapper with the orchestrator
global_orchestrator.visual_adapter.set_real_model(remote_visual_model)
```

### Why this works seamlessly:
- **Parallel HTTP Calls**: The backend orchestrator fires HTTP requests to all remote machines in parallel using `asyncio.gather`.
- **Fault Tolerance**: If a remote computer crashes or loses network connection, the orchestrator safely isolates the error and falls back without crashing the main server.
- **Identical JSON Schema**: As long as the remote endpoint returns JSON matching the module's contract specified in Sections 1–4, everything works out of the box.

---

## 🧪 Integration Verification Script

To test if your model integrates cleanly with the backend without starting the web server, run this test script in Python from the workspace root directory (`inno_2025`):

```python
# Save as test_my_adapter.py at workspace root (/home/sharvesh/inno_2025)
import asyncio
from backend.services.orchestrator import global_orchestrator

async def test_integration():
    # 1. Register your model
    # global_orchestrator.visual_adapter.set_real_model(my_visual_model)

    # 2. Run orchestrator pipeline on demo media
    result = await global_orchestrator.run_pipeline("backend/demo_media/sample_fake.jpg", "TL-2026-TEST")
    
    print("Verdict:", result.verdict)
    print("Fusion Score:", result.fusion_score)
    print("Confidence:", result.confidence)
    print("Explanations:", result.explanations)

if __name__ == "__main__":
    asyncio.run(test_integration())
```

Execute from workspace root:
```bash
cd /home/sharvesh/inno_2025
PYTHONPATH=. source backend/venv/bin/activate && python test_my_adapter.py
```

---

## 📌 Rules & Best Practices

1. **Error Isolation**: Throwing an uncaught exception inside your model will not crash the backend server; the orchestrator will catch it and fall back gracefully. However, returning proper `status: "SUCCESS"` dicts is preferred.
2. **Pathing**: Save generated heatmap files inside `backend/storage/heatmaps/`.
3. **No Network Locks**: Models run asynchronously within the FastAPI event loop. If using CPU/GPU heavy operations, wrap blocking calls with `asyncio.to_thread(sync_inference_function, ...)`.
