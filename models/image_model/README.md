# 🚀 TruthLens AI — Visual AI Release Package (v2.0)

Production-ready release package for the **TruthLens Visual AI & Dense Mask Heatmap Forensic Module (Person 1 / Person 2)**.

---

## 📦 What's Included

* **Trained Models (`checkpoints/`)**:
  * `truthlens_dinov2_model.pth`: SOTA Vision Transformer (DINOv2 ViT-S/14 + Bayar Noise + Cross-Attention + Dense Mask Decoder) trained checkpoint.
  * `truthlens_sota_model.pth`: Dual-Stream ResNet18 + SRM fallback checkpoint.
* **Core Multi-Stream Forensic Pipeline (`inference/`)**:
  * `deep_tamper_detector.py`: DINOv2 neural feature backbone with dense pixel mask extraction.
  * `srm_filters.py`: Steganalysis Spatial Rich Model noise residual extractor with Poisson-Gaussian stabilization.
  * `frequency_analysis.py`: 2D DCT / FFT azimuthal spectral energy distribution and Error Level Analysis (ELA).
  * `illumination_forensics.py`: 3D physical lighting vector estimation & angular discrepancy ($\Delta \theta$).
  * `fusion_and_localization.py`: Multi-signal fusion, neural mask heatmap overlay & bounding box extraction.
  * `truthlens_adapter.py` / `image_model_adapter.py`: Async & sync adapters conforming to the TruthLens backend contract.
* **Model Architectures (`models/`)**:
  * `dino_forensic_model.py`: Pure PyTorch DINOv2 multi-task architecture (zero training dependencies).
  * `dual_stream_net.py`: Pure PyTorch DualStreamNet architecture.
* **Configuration & Utilities (`config/`, `utils/`)**:
  * YAML hyperparameters, inference thresholds, and GPU/CPU automatic fallback.
* **Storage & Test Samples (`storage/heatmaps/`, `test_samples/`)**:
  * Pre-configured heatmap destination directory and sample evidence for verification.
* **Verification & Documentation (`verify_deployment.py`, `HANDLER_INTEGRATION_GUIDE.md`, `MODEL_ARCHITECTURE_AND_TRAINING.md`)**:
  * 1-Click automated validation script and detailed guides.

---

## ⚡ Quick Start

```bash
# 1. Install minimal inference dependencies
pip install -r requirements.txt

# 2. Run 1-Click Deployment Verification
python verify_deployment.py
```

---

## 📖 Documentation & Guides

* **For the Backend Handler / Orchestrator**: See [HANDLER_INTEGRATION_GUIDE.md](HANDLER_INTEGRATION_GUIDE.md) for how to import, call, and register the async adapter with the TruthLens backend.
* **For Model Architecture, Datasets & Training Protocol**: See [MODEL_ARCHITECTURE_AND_TRAINING.md](MODEL_ARCHITECTURE_AND_TRAINING.md) for architecture diagrams, loss formulas, and dataset details.
