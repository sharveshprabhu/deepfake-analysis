"""
TruthLens Visual AI Deployment Verification Script (v2.0).
Run this script inside the deployment folder to verify model loading,
weights integrity, DINOv2 neural inference, multi-stream forensic pipeline execution,
and contract schema compliance.
"""

import os
import sys
import time
import asyncio
from pathlib import Path

# Ensure deployment package root is in sys.path
DEPLOY_DIR = Path(__file__).resolve().parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from inference.truthlens_adapter import my_visual_model, global_visual_pipeline


async def run_verification():
    print("=" * 75)
    print("=== TRUTHLENS VISUAL AI MODEL DEPLOYMENT VERIFICATION (v2.0) ===")
    print("=" * 75)

    # 1. Verify Checkpoints
    dinov2_ckpt = DEPLOY_DIR / "checkpoints" / "truthlens_dinov2_model.pth"
    if dinov2_ckpt.exists():
        size_mb = dinov2_ckpt.stat().st_size / (1024 * 1024)
        print(f"[PASS] Primary DINOv2 Trained Checkpoint found: {dinov2_ckpt.name} ({size_mb:.1f} MB)")
    else:
        print(f"[FAIL] Primary Checkpoint missing at: {dinov2_ckpt}")
        return 1

    # 2. Verify Sample Test Asset
    sample_img = DEPLOY_DIR / "test_samples" / "sample_evidence.jpg"
    if not sample_img.exists():
        print(f"[FAIL] Sample test image missing at: {sample_img}")
        return 1
    print(f"[PASS] Sample test evidence found: {sample_img.name}")

    # 3. Test Async Adapter Inference
    evidence_id = "TL-DEPLOY-VERIFY-001"
    print(f"\n[*] Running Live Multi-Stream Forensic Analysis on {sample_img.name}...")
    t0 = time.time()
    result = await my_visual_model(str(sample_img), evidence_id)
    dt = time.time() - t0

    print(f"    [INFO] Execution Time: {dt:.2f}s")
    print(f"    [INFO] Status: {result.get('status')}")
    print(f"    [INFO] Visual Score: {result.get('visual_score')}")
    print(f"    [INFO] Frequency Score: {result.get('frequency_score')}")
    print(f"    [INFO] Manipulation Score: {result.get('details', {}).get('manipulation_score')}")
    print(f"    [INFO] Architecture: {result.get('details', {}).get('deep_architecture')}")
    print(f"    [INFO] Detected Regions: {len(result.get('regions', []))}")
    for r in result.get("regions", []):
        print(f"           -> Box: {r['box']} | Label: {r['label']} | Score: {r['anomaly_score']}")
    print(f"    [INFO] Heatmap: {result.get('heatmap_filename')}")
    print(f"    [INFO] Explanations: {result.get('explanations')}")

    # 4. Validate Strict JSON Contract Specifications
    print("\n[*] Validating Frozen TruthLens JSON Contract Schema...")
    assert result.get("module") == "visual_ai", f"Invalid module: {result.get('module')}"
    assert result.get("evidence_id") == evidence_id, f"Mismatched evidence_id: {result.get('evidence_id')}"
    assert result.get("status") == "SUCCESS", f"Status must be SUCCESS, got {result.get('status')}"
    assert isinstance(result.get("visual_score"), float), "visual_score must be float"
    assert isinstance(result.get("frequency_score"), float), "frequency_score must be float"
    assert isinstance(result.get("suspicious_frames"), list), "suspicious_frames must be list"
    assert isinstance(result.get("regions"), list), "regions must be list"
    assert isinstance(result.get("explanations"), list) and len(result.get("explanations")) > 0, "explanations must be non-empty list"
    assert result.get("heatmap_filename") != "", "heatmap_filename must not be empty"
    print("    [PASS] Strict JSON schema structure successfully validated.")

    # 5. Verify Heatmap Storage Output
    heatmap_path = DEPLOY_DIR / "storage" / "heatmaps" / result["heatmap_filename"]
    if heatmap_path.exists():
        h_size = heatmap_path.stat().st_size / 1024
        print(f"    [PASS] Anomaly Heatmap generated & saved: {heatmap_path.name} ({h_size:.1f} KB)")
    else:
        print(f"    [FAIL] Heatmap file was not found at: {heatmap_path}")
        return 1

    # 6. Test Error Handling Edge Cases
    print("\n[*] Testing Error Handling on Non-Existent File...")
    err_res = await my_visual_model("non_existent_file.jpg", "TL-DEPLOY-ERR-001")
    assert err_res.get("status") == "ERROR", "Missing file should return status ERROR"
    assert err_res.get("visual_score") == 0.0, "Missing file should return visual_score 0.0"
    print("    [PASS] Non-existent file gracefully handled with ERROR status.")

    print("\n" + "=" * 75)
    print("[SUCCESS] ALL DEPLOYMENT VERIFICATION CHECKS PASSED!")
    print("This folder is completely ready for TruthLens Handler Integration.")
    print("=" * 75 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_verification()))
