"""
TruthLens AI Forensics Platform - Master Inference Launcher
Runs the complete multimodal deepfake detection engine and Web UI.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  TRUTHLENS: MULTIMODAL DEEPFAKE FORENSIC PLATFORM")
    print("  Full Precision (FP32) Production Inference Server")
    print("=" * 70)
    print(f"  Repository Root: {ROOT_DIR}")
    print(f"  Backend Engine:  {BACKEND_DIR}")
    print(f"  Web Dashboard:   http://localhost:8000/ui")
    print(f"  Swagger API:     http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run("backend.main:app", app_dir=str(ROOT_DIR), host="0.0.0.0", port=8000, reload=False)
