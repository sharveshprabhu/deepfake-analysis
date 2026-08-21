"""
Audio-Visual Data Augmentations for AV-CrossSyncNet Training.
Implements spatial jitter, random horizontal flipping, color perturbations,
Gaussian blur, audio scaling, and mild noise addition.
"""
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Tuple, Optional


class VideoAugmentation:
    """Applies consistent spatiotemporal augmentations across a sequence of video frames."""
    def __init__(
        self,
        crop_size: int = 96,
        is_training: bool = True,
        hflip_p: float = 0.5,
        color_jitter_p: float = 0.3,
        blur_p: float = 0.2
    ):
        self.crop_size = crop_size
        self.is_training = is_training
        self.hflip_p = hflip_p
        self.color_jitter_p = color_jitter_p
        self.blur_p = blur_p

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (T, C, H, W) float tensor normalized to [0, 1]
        returns: (T, C, crop_size, crop_size) augmented float tensor
        """
        T, C, H, W = frames.shape

        if not self.is_training:
            # Deterministic center crop and resize
            if H != self.crop_size or W != self.crop_size:
                frames = TF.resize(frames, [self.crop_size, self.crop_size], antialias=True)
            return frames

        # Random horizontal flip applied consistently across all T frames
        if random.random() < self.hflip_p:
            frames = TF.hflip(frames)

        # Random mild brightness / contrast jitter applied across sequence
        if random.random() < self.color_jitter_p:
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)
            frames = TF.adjust_brightness(frames, brightness_factor)
            frames = TF.adjust_contrast(frames, contrast_factor)

        # Random mild Gaussian blur (simulates out-of-focus or low-bitrate compression)
        if random.random() < self.blur_p:
            kernel_size = 3
            sigma = random.uniform(0.1, 1.5)
            frames = TF.gaussian_blur(frames, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        # Resize to target crop size
        if H != self.crop_size or W != self.crop_size:
            frames = TF.resize(frames, [self.crop_size, self.crop_size], antialias=True)

        return torch.clamp(frames, 0.0, 1.0)


class AudioAugmentation:
    """Applies acoustic augmentations to audio waveform / mel spectrogram."""
    def __init__(
        self,
        is_training: bool = True,
        noise_p: float = 0.2,
        gain_p: float = 0.3
    ):
        self.is_training = is_training
        self.noise_p = noise_p
        self.gain_p = gain_p

    def __call__(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        mel_spec: (n_mels, T_audio) float tensor
        returns: (n_mels, T_audio) augmented float tensor
        """
        if not self.is_training:
            return mel_spec

        # Random gain / volume perturbation
        if random.random() < self.gain_p:
            gain = random.uniform(0.90, 1.10)
            mel_spec = mel_spec * gain

        # Random mild additive Gaussian noise (simulates microphone background hiss)
        if random.random() < self.noise_p:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(mel_spec) * noise_std
            mel_spec = mel_spec + noise

        return mel_spec
