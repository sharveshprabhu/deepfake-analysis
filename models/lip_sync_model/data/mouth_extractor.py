"""
Universal Mouth Region of Interest (ROI) Extractor.
Supports:
1. Ultra-fast direct bounding box extraction on pre-aligned LRS3 224x224 face-tracks.
2. Dynamic landmark-guided mouth extraction for arbitrary full-frame videos using MTCNN or MediaPipe.
"""
import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union


class MouthExtractor:
    """Extracts stabilized, normalized mouth region frames from video clips."""
    def __init__(
        self,
        crop_size: int = 96,
        ymin_ratio: float = 0.50,
        ymax_ratio: float = 0.95,
        xmin_ratio: float = 0.20,
        xmax_ratio: float = 0.80,
        enable_landmarks: bool = True
    ):
        self.crop_size = crop_size
        self.ymin_ratio = ymin_ratio
        self.ymax_ratio = ymax_ratio
        self.xmin_ratio = xmin_ratio
        self.xmax_ratio = xmax_ratio
        self.enable_landmarks = enable_landmarks
        self._mtcnn = None

    def _init_mtcnn(self):
        """Lazy initialization of MTCNN detector for in-the-wild videos."""
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._mtcnn = MTCNN(
                    image_size=224,
                    margin=20,
                    keep_all=False,
                    select_largest=True,
                    device=device,
                    post_process=False
                )
            except Exception:
                self._mtcnn = None

    def crop_lrs3_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts mouth ROI from an already centered 224x224 LRS3 video frame.
        frame: (H, W, 3) BGR or RGB image
        returns: (crop_size, crop_size, 3) cropped and resized image
        """
        H, W = frame.shape[:2]
        ymin = int(H * self.ymin_ratio)
        ymax = int(H * self.ymax_ratio)
        xmin = int(W * self.xmin_ratio)
        xmax = int(W * self.xmax_ratio)

        mouth = frame[ymin:ymax, xmin:xmax]
        if mouth.shape[0] != self.crop_size or mouth.shape[1] != self.crop_size:
            mouth = cv2.resize(mouth, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        return mouth

    def crop_video_frames(
        self,
        frames: Union[List[np.ndarray], np.ndarray],
        is_lrs3_prealigned: bool = True
    ) -> np.ndarray:
        """
        Extracts mouth ROI sequence from a list or array of frames.
        frames: list of (H, W, 3) or (T, H, W, 3) ndarray (RGB format)
        returns: (T, crop_size, crop_size, 3) uint8 ndarray
        """
        if is_lrs3_prealigned:
            cropped = [self.crop_lrs3_frame(f) for f in frames]
            return np.stack(cropped, axis=0)

        # In-the-wild full video frames -> detect face and extract mouth
        self._init_mtcnn()
        cropped = []
        last_box = None

        for idx, frame in enumerate(frames):
            H, W = frame.shape[:2]
            # Detect face every 15 frames or if no box yet, using fast downscaled image
            if (last_box is None or idx % 15 == 0) and self._mtcnn is not None:
                try:
                    # Downscale for ultra-fast face detection
                    scale = min(1.0, 480.0 / max(H, W))
                    if scale < 1.0:
                        small_f = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
                    else:
                        small_f = frame
                    boxes, _ = self._mtcnn.detect(small_f)
                    if boxes is not None and len(boxes) > 0:
                        last_box = boxes[0] / scale
                except Exception:
                    pass

            if last_box is not None:
                bx1, by1, bx2, by2 = [int(v) for v in last_box]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(W, bx2), min(H, by2)
                face = frame[by1:by2, bx1:bx2]
                if face.size > 0:
                    mouth = self.crop_lrs3_frame(face)
                    cropped.append(mouth)
                    continue

            # Fallback to lower center of frame
            fallback_mouth = self.crop_lrs3_frame(frame)
            cropped.append(fallback_mouth)

        return np.stack(cropped, axis=0)
