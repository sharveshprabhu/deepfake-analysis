"""
Unified Dataset Loader for Person 1 (Spatial AI)
Loads whole-frame video datasets using source-aware splits with balanced sampling.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Any
import numpy as np

try:
    from .frame_sampler import WholeFrameSampler
    from .splitter import create_source_aware_splits
except ImportError:
    from frame_sampler import WholeFrameSampler
    from splitter import create_source_aware_splits


class SpatialVideoDataset(Dataset):
    """
    PyTorch Dataset serving whole frames from source-isolated video splits.
    """
    def __init__(
        self,
        split_name: str = "train",
        splits_path: str = "d:/Innohack/person1_spatial/data/splits.json",
        image_size: int = 224,
        frames_per_video: int = 4,
        max_samples: Optional[int] = None
    ):
        self.split_name = split_name
        self.image_size = image_size
        self.frames_per_video = frames_per_video
        self.is_training = (split_name == "train")
        
        if not os.path.exists(splits_path):
            print(f"[DatasetLoader] Splits file {splits_path} not found. Generating now...")
            create_source_aware_splits(save_path=splits_path)
            
        with open(splits_path, 'r') as f:
            all_splits = json.load(f)
            
        self.video_items = all_splits.get(split_name, [])
        if max_samples and max_samples < len(self.video_items):
            self.video_items = self.video_items[:max_samples]
            
        self.sampler = WholeFrameSampler(
            image_size=image_size,
            frames_per_video=frames_per_video,
            is_training=self.is_training
        )
        
        # Build flattened frame index: (video_idx, frame_slot_idx)
        self.frame_index = []
        for v_idx in range(len(self.video_items)):
            for f_slot in range(frames_per_video):
                self.frame_index.append((v_idx, f_slot))

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        v_idx, f_slot = self.frame_index[idx]
        v_item = self.video_items[v_idx]
        
        # Sample frames from video
        frames = self.sampler.sample_video_frames(
            video_path=v_item["path"],
            num_frames=self.frames_per_video
        )
        
        if f_slot < len(frames):
            tensor_img, frame_no, timestamp = frames[f_slot]
        else:
            tensor_img, frame_no, timestamp = frames[0]
            
        return {
            "image": tensor_img,
            "label": torch.tensor(v_item["label"], dtype=torch.float32),
            "source_id": v_item.get("source_id", ""),
            "dataset": v_item.get("dataset", ""),
            "video_path": v_item.get("path", ""),
            "frame_idx": frame_no,
            "timestamp": timestamp
        }

    def get_sample_weights(self) -> torch.Tensor:
        """
        Computes balanced inverse-frequency sampling weights across real and fake classes.
        """
        labels = [self.video_items[v_idx]["label"] for v_idx, _ in self.frame_index]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[lbl] for lbl in labels]
        return torch.tensor(sample_weights, dtype=torch.float32)


def get_dataloader(
    split_name: str = "train",
    splits_path: str = "d:/Innohack/person1_spatial/data/splits.json",
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 224,
    frames_per_video: int = 4,
    balanced_sampling: bool = True,
    max_samples: Optional[int] = None
) -> DataLoader:
    """
    Constructs a DataLoader for the given split with optional balanced sampling.
    """
    dataset = SpatialVideoDataset(
        split_name=split_name,
        splits_path=splits_path,
        image_size=image_size,
        frames_per_video=frames_per_video,
        max_samples=max_samples
    )
    
    if split_name == "train" and balanced_sampling:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
