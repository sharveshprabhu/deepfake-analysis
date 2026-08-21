"""
Whole-Frame Temporal Frame Sampler for Person 1 (Spatial AI)
Extracts and preprocesses COMPLETE whole frames (no face-only cropping).
Preserves scene structure, lighting, background, objects, edges, and fine manipulation artifacts.
"""

import os
import cv2
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
import torchvision.transforms as T
from PIL import Image


import io

class RobustSpatialAugmentations:
    """
    Comprehensive Domain-Destruction Augmentation Pipeline:
    Simulates real-world social media compression, transmission artifacts, and sensor noise.
    """
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.color_jitter = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.06)
        self.gaussian_blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 2.2))
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. Random Horizontal Flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Simulated JPEG Compression Artifacts (Quality 30-85)
        if random.random() < 0.6:
            quality = random.randint(30, 85)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)

        # 3. Social Media Resize & Downsampling Simulation
        if random.random() < 0.5:
            scale_factor = random.choice([0.45, 0.55, 0.70])
            down_w = max(32, int(self.image_size * scale_factor))
            down_h = max(32, int(self.image_size * scale_factor))
            img = img.resize((down_w, down_h), resample=Image.BILINEAR)

        # Resize to canonical network dimension
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        # 4. Photometric Color Jitter
        if random.random() < 0.7:
            img = self.color_jitter(img)

        # 5. Gaussian Blur
        if random.random() < 0.4:
            img = self.gaussian_blur(img)

        # Convert to tensor
        tensor_img = T.functional.to_tensor(img)

        # 6. Additive Sensor Noise (ISO Noise)
        if random.random() < 0.35:
            noise_std = random.uniform(0.01, 0.04)
            noise = torch.randn_like(tensor_img) * noise_std
            tensor_img = torch.clamp(tensor_img + noise, 0.0, 1.0)

        # 7. Channel Normalization
        return self.normalize(tensor_img)


class WholeFrameSampler:
    """
    Samples and preprocesses full whole frames from video files with robustness augmentations.
    """
    def __init__(
        self,
        image_size: int = 224,
        frames_per_video: int = 4,
        is_training: bool = True
    ):
        self.image_size = image_size
        self.frames_per_video = frames_per_video
        self.is_training = is_training
        
        if is_training:
            self.robust_aug = RobustSpatialAugmentations(image_size=image_size)
            self.transform = None
        else:
            self.robust_aug = None
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def sample_video_frames(
        self,
        video_path: str,
        num_frames: Optional[int] = None
    ) -> List[Tuple[torch.Tensor, int, float]]:
        """
        Samples full whole frames from video.
        Returns: List of (tensor_image, frame_idx, timestamp_sec)
        """
        k_frames = num_frames or self.frames_per_video
        
        if not os.path.exists(video_path):
            return self._get_blank_frames(k_frames)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._get_blank_frames(k_frames)
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        if total_frames <= 0:
            cap.release()
            return self._get_blank_frames(k_frames)
            
        if self.is_training:
            # Random temporal sampling
            if total_frames <= k_frames:
                frame_indices = sorted(list(range(total_frames)))
            else:
                frame_indices = sorted(random.sample(range(total_frames), k_frames))
        else:
            # Deterministic uniform sampling
            if total_frames <= k_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, k_frames, dtype=int).tolist()
                
        results = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply whole-frame transforms
            if self.is_training:
                pil_img = Image.fromarray(frame_rgb)
                tensor_frame = self.robust_aug(pil_img)
            else:
                tensor_frame = self.transform(frame_rgb)
                
            timestamp = round(idx / fps, 3)
            results.append((tensor_frame, idx, timestamp))
            
        cap.release()
        
        if len(results) == 0:
            return self._get_blank_frames(k_frames)
            
        return results

    def preprocess_single_image(self, image_input) -> torch.Tensor:
        """
        Preprocesses a single image or numpy array into a normalized whole-frame tensor.
        """
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Cannot open image path: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input)
        else:
            img_rgb = image_input
            
        if self.is_training:
            pil_img = Image.fromarray(img_rgb) if not isinstance(image_input, Image.Image) else image_input
            return self.robust_aug(pil_img)
        else:
            return self.transform(img_rgb)

    def _get_blank_frames(self, count: int) -> List[Tuple[torch.Tensor, int, float]]:
        blank_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        return [(blank_tensor, 0, 0.0) for _ in range(count)]
