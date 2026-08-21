import sys
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from backend.model_registry import register_all_models
from backend.services.orchestrator import global_orchestrator

async def run_test():
    print("=" * 65)
    print("TruthLens AI Forensics Platform - Inference Verification Test")
    print("=" * 65)
    
    reg_res = register_all_models()
    print("Model Registration:", reg_res)
    
    real_sample = ROOT_DIR / "demo_samples" / "real_video_sample.mp4"
    fake_sample = ROOT_DIR / "demo_samples" / "deepfake_video_sample.mp4"
    
    if real_sample.exists():
        print(f"\n[1/2] Analyzing Authentic Video Sample: {real_sample.name}")
        res_real = await global_orchestrator.run_pipeline(real_sample, "TEST-REAL", real_sample.name)
        v_real = res_real['verdict'].value if hasattr(res_real['verdict'], 'value') else str(res_real['verdict'])
        print(f"  -> Verdict:          {v_real}")
        print(f"  -> Confidence:       {res_real['confidence']*100:.1f}%")
        print(f"  -> Fusion Score:     {res_real['fusion_score']}")
        print(f"  -> Temporal Score:   {res_real['temporal_score']}")
        print(f"  -> Visual Score:     {res_real['visual_score']}")
        print(f"  -> Status:           {'PASS (AUTHENTIC)' if v_real == 'AUTHENTIC' else 'FAIL'}")

    if fake_sample.exists():
        print(f"\n[2/2] Analyzing Deepfake Video Sample: {fake_sample.name}")
        res_fake = await global_orchestrator.run_pipeline(fake_sample, "TEST-FAKE", fake_sample.name)
        v_fake = res_fake['verdict'].value if hasattr(res_fake['verdict'], 'value') else str(res_fake['verdict'])
        print(f"  -> Verdict:          {v_fake}")
        print(f"  -> Confidence:       {res_fake['confidence']*100:.1f}%")
        print(f"  -> Fusion Score:     {res_fake['fusion_score']}")
        print(f"  -> Temporal Score:   {res_fake['temporal_score']}")
        print(f"  -> Visual Score:     {res_fake['visual_score']}")
        print(f"  -> Status:           {'PASS (MANIPULATED)' if v_fake == 'MANIPULATED' else 'FAIL'}")

    print("\n" + "=" * 65)
    print("Inference Verification Completed Successfully.")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_test())
