"""
Feature Extraction Interface for Person 1 (Spatial AI)
Directly consumed by Person 2 (Image Classification + Video Temporal Analysis).
Extracts whole-frame multi-scale spatial representations.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Union, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.spatial_encoder import SpatialEncoder
from data.frame_sampler import WholeFrameSampler


class FeatureExtractor:
    """
    High-performance Spatial Feature Extractor for Person 1.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = "d:/Innohack/person1_spatial/checkpoints/best_val_auc.pth",
        backbone_name: str = "efficientnet_b4",
        embedding_dim: int = 2048,
        device: Optional[str] = None,
        image_size: int = 224
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.image_size = image_size
        self.sampler = WholeFrameSampler(image_size=image_size, is_training=False)
        
        self.encoder = SpatialEncoder(
            backbone_name=backbone_name,
            pretrained=False,
            embedding_dim=embedding_dim,
            mode="full",
            include_aux_head=True
        ).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            if "encoder_state_dict" in state:
                self.encoder.load_state_dict(state["encoder_state_dict"])
            else:
                self.encoder.load_state_dict(state)
            print(f"[P1 Extractor] Loaded checkpoint: {checkpoint_path}")
            
        self.encoder.eval()

    def extract_from_frame(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Extracts multi-scale spatial representation from a single whole frame / image.
        Returns:
            global_features: (1024,) float array
            local_features: (1024,) float array
            spatial_embedding: (2048,) normalized float array
            attention_map: (H, W) spatial saliency map
        """
        tensor_img = self.sampler.preprocess_single_image(image_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.encoder(tensor_img)
            
            global_feat = out["global_features"].squeeze(0).cpu().numpy()
            local_feat = out["local_features"].squeeze(0).cpu().numpy()
            spatial_emb = out["spatial_embedding"].squeeze(0).cpu().numpy()
            att_map = out["attention_map"].squeeze().cpu().numpy()
            
        return {
            "global_features": global_feat,
            "local_features": local_feat,
            "spatial_embedding": spatial_emb,
            "attention_map": att_map,
            "embedding_dim": len(spatial_emb)
        }

    def extract_from_video(
        self,
        video_path: str,
        sample_fps: int = 1,
        max_frames: int = 30
    ) -> Dict[str, Any]:
        """
        Extracts temporal sequence embeddings from video for Person 2's Temporal Model.
        Returns:
            sequence_embeddings: (T, 2048) numpy array
            frame_indices: list of frame numbers
            timestamps: list of timestamps in seconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        step = max(1, int(fps / sample_fps))
        frame_indices = list(range(0, total_frames, step))[:max_frames]
        
        sequence_embeddings = []
        timestamps = []
        valid_indices = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue
                
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.extract_from_frame(frame_rgb)
            
            sequence_embeddings.append(res["spatial_embedding"])
            timestamps.append(round(idx / fps, 3))
            valid_indices.append(idx)
            
        cap.release()
        
        return {
            "video_path": video_path,
            "total_frames_extracted": len(sequence_embeddings),
            "frame_indices": valid_indices,
            "timestamps_sec": timestamps,
            "sequence_embeddings": np.array(sequence_embeddings) # Shape: (T, 2048) -> Ready for P2!
        }

    # Aliases for Person 2 convenience
    extract_frame_features = extract_from_frame
    extract_video_sequence = extract_from_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Spatial Features (Person 1 -> Person 2)")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="person1_spatial/checkpoints/best_auc.pth")
    args = parser.parse_args()
    
    extractor = FeatureExtractor(checkpoint_path=args.checkpoint)
    if args.image:
        res = extractor.extract_from_frame(args.image)
        print(f"Extracted Frame Spatial Embedding Shape: {res['spatial_embedding'].shape}")
    elif args.video:
        res = extractor.extract_from_video(args.video, sample_fps=1)
        print(f"Extracted Video Sequence Embeddings Shape: {res['sequence_embeddings'].shape}")
    else:
        print("Please provide --image or --video to extract features.")
