# 🛡️ Person 1 (Spatial AI) — Whole-Frame Spatial Feature Extractor

Welcome to the **Person 1 (Spatial AI)** module for the Innohack Deepfake Detection System.

---

## 📌 Role Boundary & Objective

As **Person 1 (Spatial AI)**, this module is strictly responsible for building a **highly generalizable WHOLE-FRAME SPATIAL FEATURE EXTRACTOR**.

### What Person 1 Does:
* Analyzes the **complete image/frame** (no face-only cropping, no face detection prerequisite).
* Learns multi-scale spatial representations capturing:
  1. Texture & fine-grained manipulation artifacts
  2. Edges & blending boundaries
  3. Local and global visual inconsistencies
  4. Scene geometry, lighting patterns, and shadows
  5. Compression artifacts and background/object relationships
* Produces rich **2048-dimensional spatial embeddings** consumed directly by **Person 2**.

### What Person 1 Does NOT Do:
* Final image Real/Fake classification (Owned by Person 2)
* Temporal/video classification (Owned by Person 2)
* Audio-visual / lip synchronization (Owned by Person 2)
* Final score aggregation (Owned by Person 3)
* Frontend / UI (Owned by Person 4)

---

## 🏗️ Multi-Scale Spatial Architecture

```
                    INPUT IMAGE/FRAME (Whole Frame)
                               │
                         PREPROCESSING
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
          GLOBAL BRANCH                 LOCAL BRANCH
      (Scene structure, lighting,    (Fine artifacts, texture,
       geometry, shadows, objects)    blending edges, patches)
                │                             │
                └──────────────┬──────────────┘
                               │
                     FEATURE FUSION (Gated)
                               │
                               ▼
                        SPATIAL ENCODER
                 (2048-dim Spatial Embedding)
```

---

## 🔒 Zero-Leakage Dataset Strategy

This module enforces **strict source-aware splitting**:
* All manipulated versions derived from the same source video remain in the **exact same split**.
* **Train Split**: Source-isolated FF++ (~70%) + Source-isolated Celeb-DF (~70%)
* **Val Split**: Source-isolated FF++ (~15%) + Source-isolated Celeb-DF (~15%)
* **In-Domain Test Split**: Source-isolated FF++ (~15%) + Celeb-DF Official Test Set
* **Cross-Dataset Benchmark**: DeeperForensics-1.0 (with real-world perturbations)

---

## 📂 Project Structure

```
person1_spatial/
├── data/
│   ├── dataset_loader.py       # Whole-frame loader with balanced sampling
│   ├── dataset_audit.py        # Dataset property & corruption auditor
│   ├── splitter.py             # Zero-leakage source-aware splitter
│   ├── leakage_checker.py      # Strict hash & source overlap validator
│   ├── frame_sampler.py        # Whole-frame temporal sampler
│   ├── audit_report.json       # Generated dataset audit summary
│   └── splits.json             # Source-isolated dataset split manifest
├── models/
│   ├── backbone.py             # Vision backbones (EfficientNet, ConvNeXt, ResNet)
│   ├── global_branch.py        # Scene-wide structure & lighting branch
│   ├── local_branch.py         # Patch-level fine artifact branch
│   ├── feature_fusion.py       # Gated multi-scale feature fusion
│   └── spatial_encoder.py      # Primary SpatialEncoder module
├── training/
│   ├── train.py                # PyTorch GPU training with AMP
│   ├── losses.py               # Focal + Metric representation loss
│   ├── validate.py             # Validation engine
│   └── ablation.py             # Controlled ablation study runner
├── evaluation/
│   ├── metrics.py              # ROC-AUC, PR-AUC, F1, EER
│   ├── cross_dataset.py        # Cross-dataset generalization benchmark
│   └── embedding_analysis.py   # Downstream linear probe evaluation
├── inference/
│   └── extract_features.py     # Clean extraction API for Person 2
├── configs/
│   └── training.yaml           # Experiment hyperparameters
├── checkpoints/                # Saved weights
└── README.md
```

---

## 🤝 Integration Contract for Person 2

Person 2 consumes P1's spatial representation for **Image Classification** and **Video Temporal Analysis**:

```python
from person1_spatial.inference.extract_features import FeatureExtractor

# 1. Initialize extractor
extractor = FeatureExtractor(checkpoint_path="person1_spatial/checkpoints/best_auc.pth")

# 2. Extract spatial representation from single image / frame:
features = extractor.extract_from_frame("path/to/frame.jpg")
# features["spatial_embedding"]: shape (2048,) L2-normalized vector
# features["global_features"]   : shape (1024,) scene structure vector
# features["local_features"]    : shape (1024,) artifact vector
# features["attention_map"]     : shape (H, W) spatial saliency map

# 3. Extract temporal sequence of embeddings from video:
seq_features = extractor.extract_from_video("path/to/video.mp4", sample_fps=1)
# seq_features["sequence_embeddings"]: shape (T, 2048) -> Feed into Temporal LSTM / Transformer!
```

---

## 🚀 Execution Guide

### 1. Run Dataset Audit & Leakage Verification
```bash
python person1_spatial/data/dataset_audit.py
python person1_spatial/data/splitter.py
```

### 2. Train Spatial Representation
```bash
python person1_spatial/training/train.py --config person1_spatial/configs/training.yaml
```

### 3. Run Controlled Ablation Study
```bash
python person1_spatial/training/ablation.py
```

### 4. Run Cross-Dataset Benchmark (DeeperForensics-1.0)
```bash
python person1_spatial/evaluation/cross_dataset.py
```

### 5. Downstream Linear Probe Test
```bash
python person1_spatial/evaluation/embedding_analysis.py
```
