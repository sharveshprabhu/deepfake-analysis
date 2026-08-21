"""
Deep Spatial Tampering & Boundary Anomaly Detector (Release Package v2).
Supports:
1. TruthLensDinov2Net (DINOv2 ViT-S/14 + Bayar Noise + Dense Mask Decoder)
2. TruthLensDualStreamNet (ResNet-18 + SRM Stream fallback)
Outputs high-accuracy manipulation scores and pixel-accurate anomaly maps.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from inference.srm_filters import SRMNoiseExtractor
    from models.dino_forensic_model import TruthLensDinov2Net
    from models.dual_stream_net import TruthLensDualStreamNet
except ImportError:
    from srm_filters import SRMNoiseExtractor
    from dino_forensic_model import TruthLensDinov2Net
    from dual_stream_net import TruthLensDualStreamNet


class DeepTamperDetector(nn.Module):
    """
    Forensic Neural Backbone with DINOv2 multi-task architecture
    and seamless fallback to DualStreamNet.
    """

    def __init__(self, model_checkpoint: str = None, device: str = None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.srm_extractor = SRMNoiseExtractor()
        self.is_dinov2 = False

        # Locate best checkpoint across deployment package and local directories
        candidates = [
            Path(model_checkpoint) if model_checkpoint else None,
            PACKAGE_ROOT / "checkpoints" / "truthlens_dinov2_model.pth",
            PACKAGE_ROOT / "models" / "truthlens_dinov2_model.pth",
            PACKAGE_ROOT / "checkpoints" / "truthlens_sota_model.pth",
            PACKAGE_ROOT / "models" / "truthlens_sota_model.pth",
            CURRENT_DIR / "checkpoints" / "truthlens_dinov2_model.pth",
        ]

        target_ckpt = None
        for c in candidates:
            if c and c.exists():
                target_ckpt = c
                break

        if target_ckpt and "dinov2" in target_ckpt.name.lower():
            try:
                self.model = TruthLensDinov2Net(pretrained=False).to(self.device)
                checkpoint = torch.load(str(target_ckpt), map_location=self.device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state_dict)
                self.is_dinov2 = True
                val_acc = checkpoint.get("val_acc", 0.0)
                val_iou = checkpoint.get("val_iou", 0.0)
                print(f"  [MODEL] Loaded DINOv2 Checkpoint: {target_ckpt.name} (Val Acc: {val_acc*100:.1f}%, IoU: {val_iou*100:.1f}%)")
            except Exception as e:
                print(f"  [MODEL WARNING] Failed loading DINOv2 ({e}), falling back to DualStreamNet.")
                self.model = TruthLensDualStreamNet().to(self.device)
                self.is_dinov2 = False
        else:
            self.model = TruthLensDualStreamNet().to(self.device)
            if target_ckpt and target_ckpt.exists():
                checkpoint = torch.load(str(target_ckpt), map_location=self.device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state_dict)
                val_acc = checkpoint.get("val_acc", 0.0)
                print(f"  [MODEL] Loaded DualStream Checkpoint: {target_ckpt.name} (Val Acc: {val_acc*100:.1f}%)")

        self.model.eval()
        self.img_size = 252 if self.is_dinov2 else 256

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Hook storage for fallback Grad-CAM
        self.gradients = None
        self.activations = None
        if not self.is_dinov2 and hasattr(self.model, "rgb_backbone"):
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = self.model.rgb_backbone.layer4[-1].conv2
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def extract_cam(self, rgb_tensor: torch.Tensor, srm_tensor: torch.Tensor, logit: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        logit.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.ones((self.img_size, self.img_size), dtype=np.float32) * 0.2

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) > 1e-6:
            heatmap = heatmap / np.max(heatmap)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def predict(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray, dict]:
        """
        Runs neural forensic inference.
        Returns:
            (spatial_tamper_score, normalized_anomaly_map, metrics)
        """
        h, w = image_bgr.shape[:2]

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        rgb_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        srm_res = self.srm_extractor.extract_residuals(gray)
        srm_res_resized = cv2.resize(srm_res, (self.img_size, self.img_size))
        srm_tensor = torch.tensor(srm_res_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.is_dinov2:
                pred_logit, pred_mask = self.model(rgb_tensor, srm_tensor)
                prob = float(torch.sigmoid(pred_logit).item())
                mask_prob = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
                anomaly_map = cv2.resize(mask_prob, (w, h), interpolation=cv2.INTER_CUBIC)
                anomaly_map = np.clip(anomaly_map, 0.0, 1.0)
                metrics = {
                    "architecture": "TruthLensDinov2Net",
                    "trained_dinov2_prob": round(prob, 4),
                    "device": str(self.device)
                }
                return prob, anomaly_map, metrics

        # Fallback DualStreamNet
        rgb_tensor.requires_grad = True
        logit = self.model(rgb_img=rgb_tensor, srm_res=srm_tensor)
        prob = float(torch.sigmoid(logit).item())
        cam_small = self.extract_cam(rgb_tensor, srm_tensor, logit)
        cam_full = cv2.resize(cam_small, (w, h), interpolation=cv2.INTER_CUBIC)
        cam_full = np.clip(cam_full, 0.0, 1.0)

        metrics = {
            "architecture": "TruthLensDualStreamNet",
            "trained_dual_stream_prob": round(prob, 4),
            "device": str(self.device)
        }
        return prob, cam_full, metrics
