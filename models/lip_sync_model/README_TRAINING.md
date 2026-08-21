# 🎓 TruthLens AV-CrossSyncNet: Complete Training & Reproduction Guide (Person 2B)

This guide documents the complete training pipeline, dataset preprocessing, model architecture, hyperparameters, and benchmark reproduction for the **AV-CrossSyncNet (Audio-Visual Cross-Attention Lip Synchronization and Dubbing Deepfake Detection System)**.

---

## 🏛️ 1. Architecture Overview

AV-CrossSyncNet is a multi-modal neural network combining:
1. **3D Spatiotemporal Stem + Pretrained 2D ResNet-18** (`models/visual_encoder.py`):
   - Input: $5\text{ frames} \times 96 \times 96 \times 3\text{ RGB}$ ($0.2\text{s}$ visual speech window at $25\text{fps}$).
   - Pretrained ImageNet weights provide rich facial contour, edge, and mouth geometry representations.
   - Output: $(B, 5, 256)$ visual viseme embedding sequence.
2. **2D ResNet-18 Audio Phoneme Encoder** (`models/audio_encoder.py`):
   - Input: $1\text{ channel} \times 80\text{ Log-Mel filterbanks} \times 16\text{ acoustic frames}$ ($0.2\text{s}$ at $16\text{kHz}$, $10\text{ms}$ hop).
   - Output: $(B, 16, 256)$ speech phoneme embedding sequence.
3. **Bidirectional Multi-Head Cross-Attention Transformer** (`models/cross_attention.py`):
   - 4 attention heads, 2 transformer layers.
   - Computes cross-modal attention alignment matrix $\mathbf{M} \in \mathbb{R}^{B \times 5 \times 16}$ mapping phonemes to mouth opening/closing visemes.
4. **Multi-Task Heads**:
   - **Shared Metric Space**: 128-D L2-normalized projections with learnable temperature $\tau$ for InfoNCE contrastive alignment.
   - **Discrete Offset Classifier**: 31-class classifier predicting frame shifts in $[-15, +15]$ frames ($-600\text{ms}$ to $+600\text{ms}$).

---

## 📦 2. Dataset Structure & Preprocessing

The primary training corpus is **LRS3-TED** containing **31,982 synchronized video clips** across **4,004 speakers/talks**.

### Step 1: Pre-Extract Features to High-Speed Binary Cache
To eliminate CPU decoding bottlenecks and ensure 90–100% GPU saturation:

```bash
python data/preprocess.py --workers 8 --force
```

This generates:
* `data/processed_cache/train_features.pt` (28,786 clips, ~3.79 GB uint8 tensors)
* `data/processed_cache/val_features.pt` (3,031 clips, ~399 MB uint8 tensors)

---

## 🚀 3. Training Execution

### Standard Full Training (15 Epochs on GPU)

```bash
python run_training.py --epochs 15
```

### Key Training Hyperparameters:
* **Batch Size**: 64 (449 batches per epoch)
* **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$)
* **Learning Rate Schedule**: Cosine Annealing with Warmup ($10^{-4} \rightarrow 10^{-6}$)
* **Precision**: FP16 Automatic Mixed Precision (AMP) via `torch.cuda.amp.autocast()`
* **Loss Function**: Multi-Task Composite Loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + 0.5 \cdot \mathcal{L}_{\text{Offset-CE}} + 0.1 \cdot \mathcal{L}_{\text{Diag-Align}}$$
* **Curriculum Hard-Negative Mining**: 60% of negative shifts sampled from subtle micro-desyncs ($\pm 1$ to $\pm 4$ frames / $\pm 40\text{ms}$ to $\pm 160\text{ms}$).

---

## 📈 4. Training Convergence & Benchmark Results

Trained on an **NVIDIA GeForce RTX 5060 Laptop GPU** in **17.52 minutes** (~69.5s per epoch):

| Epoch | Train Loss | Val Loss | InfoNCE Loss | Val ROC-AUC | Sync Accuracy | PR-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | $1.8440$ | $1.8280$ | $0.5677$ | $0.5283$ | $50.25\%$ | $0.5312$ |
| **05** | $1.5949$ | $1.6974$ | $0.3915$ | $0.5869$ | $55.92\%$ | $0.5744$ |
| **08** | $1.4968$ | $1.6047$ | $0.3169$ | $0.7247$ | $66.41\%$ | $0.7088$ |
| **10** | $1.4529$ | $1.5295$ | $0.3032$ | $0.7901$ | $72.22\%$ | $0.7715$ |
| **12** | $1.4086$ | $1.5102$ | $0.2844$ | $0.8140$ | $73.14\%$ | $0.7962$ |
| **15** | **$1.3828$** | **$1.4863$** | **$0.2690$** | **$0.8341$** | **$75.72\%$** | **$0.8158$** |

* Best model checkpoint automatically saved to: `checkpoints/best_av_cross_syncnet.pt`.

---

## 🧪 5. Evaluation & Out-of-Domain Deepfake Benchmarking

### 1. Benchmark on LRS3 Validation Set:
```bash
python run_training.py --eval-only
```

### 2. Zero-Shot Out-of-Domain Deepfake Benchmark (DFDC & Celeb-DF):
```bash
python training/benchmark_deepfakes.py
```
Evaluates cross-modal speech desync on unseen 1080p deepfakes and writes results to `reports/zero_shot_deepfake_benchmark.json`.

---

## 💻 6. Video Inference & Backend Adapter

### Standalone CLI Prediction:
```bash
python predict.py --video /path/to/test_video.mp4
```

### TruthLens Person 2B Adapter (`audio_ai`):
```python
from adapter import truthlens_audio_avsync_adapter

# Async Non-Blocking Orchestrator Hook:
result = await truthlens_audio_avsync_adapter(
    evidence_id="TL-2026-0042",
    video_path="/path/to/video.mp4"
)
```
