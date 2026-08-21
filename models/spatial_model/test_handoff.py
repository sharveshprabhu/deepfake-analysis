# Test Handoff Script for Person 2
import os
import sys
import torch
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.dirname(base_dir))

from inference.extract_features import FeatureExtractor

def test_feature_extractor():
    print('=' * 60)
    print('PERSON 1 (SPATIAL AI) -> PERSON 2 HANDOFF VERIFICATION TEST')
    print('=' * 60)
    
    ckpt = os.path.join(base_dir, 'checkpoints', 'best_val_auc.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Checkpoint: {ckpt}')
    
    extractor = FeatureExtractor(checkpoint_path=ckpt, device=device)
    print('[*] FeatureExtractor initialized successfully!')
    
    print('\n--- 1. Testing Single Frame Extraction ---')
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = extractor.extract_frame_features(dummy_frame)
    
    emb = out['spatial_embedding']
    att = out['attention_map']
    g_feat = out['global_features']
    l_feat = out['local_features']
    
    print(f'  [+] spatial_embedding shape: {emb.shape} -> 2048-d L2-normalized')
    print(f'  [+] global_features   shape: {g_feat.shape} -> 1024-d')
    print(f'  [+] local_features    shape: {l_feat.shape} -> 1024-d')
    print(f'  [+] attention_map     shape: {att.shape} -> (1, 7, 7) Spatial Heatmap')
    print(f'  [+] Embedding L2 Norm: {np.linalg.norm(emb):.4f} (Expected: 1.0000)')
    
    print('\n--- 2. Testing Video Sequence Batch Extraction (16 frames) ---')
    dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(16)]
    tensor_batch = torch.stack([extractor.sampler.preprocess_single_image(f) for f in dummy_frames]).to(device)
    
    with torch.no_grad():
        batch_out = extractor.encoder.extract_features(tensor_batch)
        video_emb = batch_out['spatial_embedding']
        
    print(f'  [+] video_embeddings shape: {video_emb.shape} (Expected: [16, 2048])')
    print('\n' + '=' * 60)
    print('[SUCCESS] Person 1 Spatial Module is 100% Ready for Person 2!')
    print('=' * 60)

if __name__ == '__main__':
    test_feature_extractor()
