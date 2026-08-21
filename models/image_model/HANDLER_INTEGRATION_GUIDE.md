# 🔌 TruthLens Backend Handler Integration Guide (v2.0)

This guide provides everything needed for the backend team (Person 2) to integrate the Visual AI Forensic Module into the TruthLens FastAPI Orchestrator.

---

## 📌 1. Import & Call Signature

### Async Integration (Recommended for FastAPI Orchestrator)
```python
from truthlens_image_release_v2.inference.truthlens_adapter import my_visual_model

# Inside your orchestrator / service route:
result = await my_visual_model(
    file_path="storage/evidence/uploaded_image.jpg",
    evidence_id="TL-2026-0042"
)
```

### Sync Integration
```python
from truthlens_image_release_v2.inference.image_model_adapter import global_visual_pipeline

result = global_visual_pipeline.analyze_sync(
    file_path="storage/evidence/uploaded_image.jpg",
    evidence_id="TL-2026-0042"
)
```

---

## 📋 2. Output Schema Contract

The adapter strictly returns a dictionary with the following schema:

```json
{
  "module": "visual_ai",
  "evidence_id": "TL-2026-0042",
  "visual_score": 0.580,
  "frequency_score": 0.245,
  "suspicious_frames": [0],
  "regions": [
    {
      "frame_index": 0,
      "box": [126, 127, 387, 387],
      "label": "illumination_vector_mismatch",
      "anomaly_score": 0.883
    }
  ],
  "heatmap_filename": "TL-2026-0042_heatmap.png",
  "explanations": [
    "Moderate forensic anomalies detected with partial signal agreement (Score: 53.0%)",
    "Physical illumination direction discrepancy of 44.0° detected between foreground subject and background lighting vectors."
  ],
  "status": "SUCCESS",
  "details": {
    "manipulation_score": 0.530,
    "illumination_angle_discrepancy_deg": 44.0,
    "srm_noise_inconsistency": 0.210,
    "deep_tamper_score": 0.490,
    "deep_architecture": "TruthLensDinov2Net"
  }
}
```

---

## 🗂️ 3. Heatmap Storage & Access

* Generated heatmaps are automatically written to `storage/heatmaps/{evidence_id}_heatmap.png`.
* To serve heatmaps to frontend clients, mount the directory in FastAPI:
```python
from fastapi.staticfiles import StaticFiles

app.mount("/heatmaps", StaticFiles(directory="truthlens_image_release_v2/storage/heatmaps"), name="heatmaps")
```
