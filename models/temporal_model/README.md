# 🎯 TruthLens Temporal AI — Deployment Package

This standalone package contains the complete, calibrated **Temporal Deepfake Detection Model** and **TruthLens Adapter Connectors** for production and backend integration.

---

## 📚 Technical Documentation Index

1. [**HANDLER_INTEGRATION_GUIDE.md**](./HANDLER_INTEGRATION_GUIDE.md) — Step-by-step backend handler integration, asynchronous adapter usage, FastAPI microservice templates, and JSON contract schemas.
2. [**MODEL_ARCHITECTURE.md**](./MODEL_ARCHITECTURE.md) — Comprehensive neural network architectural specifications, layer-by-layer tensor dimensions, Swin-T temporal transformer, AV-sync module, and temperature calibration.
3. [**TRAINING.md**](./TRAINING.md) — Dataset partitioning, 5-point zero-leakage protocol, progressive unfreezing schedule, 20-epoch anti-memorization trace, and benchmark generalization matrices.

---

## 🚀 Quickstart

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Setup**:
   ```bash
   python verify_deployment.py
   ```

3. **Use in Backend Handler**:
   ```python
   import asyncio
   from inference.truthlens_adapter import my_temporal_model, my_audio_model

   async def main():
       # Person 2A: Temporal Deepfake Analysis
       res_temporal = await my_temporal_model("evidence.mp4", "TL-2026-0001")
       print("Temporal AI:", res_temporal)

       # Person 2B: Audio-Visual Sync Analysis
       res_audio = await my_audio_model("evidence.mp4", "TL-2026-0001")
       print("Audio AI:", res_audio)

   if __name__ == "__main__":
       asyncio.run(main())
   ```
