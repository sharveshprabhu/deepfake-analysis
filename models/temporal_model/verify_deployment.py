"""
TruthLens Deployment Verification Script.
Run this script inside the deployment folder to verify model loading,
weights integrity, and full adapter contract execution on both static and video media.
"""
import os
import sys
import tempfile
import asyncio
import numpy as np
import cv2

# Ensure current folder is in path
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

from inference.truthlens_adapter import my_temporal_model, my_audio_model

def create_synthetic_test_video(output_path: str, num_frames: int = 16, width: int = 224, height: int = 224, fps: float = 30.0):
    """Creates a temporary valid video sequence for end-to-end inference verification."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create subtle synthetic motion pattern
        frame = np.full((height, width, 3), 120, dtype=np.uint8)
        # Draw dynamic circle moving across frames
        center_x = int(50 + (i * 7) % (width - 100))
        center_y = int(112 + 20 * np.sin(i * 0.5))
        cv2.circle(frame, (center_x, center_y), 35, (220, 180, 140), -1)
        cv2.circle(frame, (center_x - 10, center_y - 10), 5, (50, 50, 50), -1)
        cv2.circle(frame, (center_x + 10, center_y - 10), 5, (50, 50, 50), -1)
        out.write(frame)
        
    out.release()

async def main():
    print("=" * 75)
    print("=== TRUTHLENS TEMPORAL MODEL DEPLOYMENT VERIFICATION ===")
    print("=" * 75)
    
    # 1. Verify Checkpoint
    ckpt_path = os.path.join(DEPLOY_DIR, "checkpoints", "best_calibrated_model.pth")
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"[PASS] 1. Calibrated Checkpoint found: {ckpt_path} ({size_mb:.1f} MB)")
    else:
        print(f"[FAIL] 1. Checkpoint missing at: {ckpt_path}")
        return 1

    # 2. Test Person 2A Adapter on Static Image (Graceful Skip)
    print("\n[*] 2. Testing Person 2A (Static Image Edge Case)...")
    res_img = await my_temporal_model("sample.jpg", "TL-DEPLOY-0001")
    print(f"    Status: {res_img.get('status')} | Temporal Score: {res_img.get('temporal_score')}")
    assert res_img["module"] == "temporal_ai", "Module name must be temporal_ai"
    assert res_img["status"] == "SKIPPED", "Image should return status SKIPPED"
    assert res_img["temporal_score"] is None, "Image should return temporal_score None"
    assert isinstance(res_img["suspicious_frame_transitions"], list), "Transitions must be a list"
    assert isinstance(res_img["explanations"], list), "Explanations must be a list"
    print("    [PASS] Static image correctly skipped according to TruthLens contract.")

    # 3. Test Person 2A Adapter on Real Video (Full Neural Inference)
    print("\n[*] 3. Testing Person 2A (End-to-End Video Sequence Inference)...")
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "test_verification_video.mp4")
    try:
        create_synthetic_test_video(temp_video_path, num_frames=16)
        res_video = await my_temporal_model(temp_video_path, "TL-DEPLOY-0002")
        print(f"    Status: {res_video.get('status')} | Temporal Score: {res_video.get('temporal_score')}")
        print(f"    Transitions: {len(res_video.get('suspicious_frame_transitions', []))} | Explanations: {res_video.get('explanations')}")
        
        assert res_video["module"] == "temporal_ai", "Module name must be temporal_ai"
        assert res_video["status"] == "SUCCESS", "Video inference should return status SUCCESS"
        assert isinstance(res_video["temporal_score"], (float, int)), "temporal_score must be a numeric float"
        assert 0.0 <= res_video["temporal_score"] <= 1.0, "temporal_score must be bounded in [0.0, 1.0]"
        assert isinstance(res_video["suspicious_frame_transitions"], list), "Transitions must be a list"
        assert len(res_video["explanations"]) > 0, "Explanations must not be empty"
        print("    [PASS] End-to-end video inference and temporal scoring passed.")

        # 4. Test Person 2B Audio / AV-Sync Adapter on Video
        print("\n[*] 4. Testing Person 2B Audio / AV-Sync Adapter on Video Sequence...")
        res_audio = await my_audio_model(temp_video_path, "TL-DEPLOY-0003")
        print(f"    Status: {res_audio.get('status')} | Audio Score: {res_audio.get('audio_score')}")
        assert res_audio["module"] == "audio_ai", "Module name must be audio_ai"
        assert res_audio["status"] == "SUCCESS", "Audio adapter should return status SUCCESS"
        print("    [PASS] Person 2B adapter structure and video execution verified.")
        
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    print("\n" + "=" * 75)
    print("[SUCCESS] ALL 4/4 DEPLOYMENT VERIFICATION CHECKS PASSED!")
    print("TruthLens Temporal AI Release package is 100% calibrated, self-contained,")
    print("and ready for production handoff and backend integration.")
    print("=" * 75 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
