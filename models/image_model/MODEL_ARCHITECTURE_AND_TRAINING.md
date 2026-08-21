# 🔬 TruthLens DINOv2 Forensic Model Architecture & Training Documentation

**Version**: 2.0 (DINOv2 Release)  
**Primary Checkpoint**: `checkpoints/truthlens_dinov2_model.pth`  
**Validation Accuracy**: **79.0%** (Validation Mask IoU: **32.2%**)  
**Inference Latency**: **~0.13s – 0.45s** on NVIDIA GPU  

---

## 🏛️ 1. Architecture

```text
                                         Input Image (252 × 252)
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
      ┌──────────────────────┐                                              ┌──────────────────────┐
      │  DINOv2 (ViT-S/14)   │                                              │ Bayar Constrained +  │
      │  Patch Tokenizer     │                                              │  Noise Residual CNN  │
      │   (384 channels)     │                                              │    (64 channels)     │
      └──────────┬───────────┘                                              └──────────┬───────────┘
                 │                                                                     │
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │   Spatial Cross-Attention   │
                                     │  (RGB Queries Noise Tokens) │
                                     └──────────────┬──────────────┘
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
  ┌──────────────────────────────┐                                      ┌──────────────────────────────┐
  │   Global Classifier Head     │                                      │     Dense Mask Decoder       │
  │   (Image Logit: Real/Fake)   │                                      │  (252×252 Segmentation Mask) │
  └──────────────┬───────────────┘                                      └──────────────┬───────────────┘
                 │                                                                     │
                 │ ──> Focal BCE Loss                                                  │ ──> Focal + Dice Loss
                 │                                                                     │     (CASIA Gt Masks)
                 └──────────────────────────────┬──────────────────────────────────────┘
                                                ▼
                                 ┌─────────────────────────────┐
                                 │   TruthLens Fusion Engine   │
                                 │   (Heatmaps & BBoxes)       │
                                 └─────────────────────────────┘
```

1. **DINOv2 Vision Transformer**: Extracts $14 \times 14$ patch representations (384-dim tokens) with high zero-shot sensitivity to synthetic blending boundaries.
2. **Bayar Constrained High-Pass Stream**: Automatically learns adaptive noise residual kernels with normalized zero-sum constraints.
3. **Spatial Cross-Attention**: Connects RGB color features directly to sensor noise patterns.
4. **Dense Mask Decoder**: Predicts full $252 \times 252$ single-channel pixel tampering masks $\hat{M}$.

---

## 📊 2. Dataset & Multi-Task Training

* **CASIA v2.0**: 7,492 Authentic, 5,123 Tampered paired with **5,123 Ground-Truth binary masks (`Gt`)**.
* **MIT Multi-Illumination**: 1,015 scenes with 25 lighting directions per scene (dynamic synthetic cross-illumination splices).
* **DFDC (Deepfake Detection Challenge)**: Facial deepfake crops with centered region masks.

### Multi-Task Loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}(\hat{y}, y) + 2.0 \cdot \mathcal{L}_{\text{MaskFocal}}(\hat{M}, M) + 1.0 \cdot \mathcal{L}_{\text{Dice}}(\hat{M}, M)$$
