import sys
from pathlib import Path

# Ensure parent directory (data) is in sys.path so 'backend' package resolves cleanly
_CURRENT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _CURRENT_DIR.parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from backend.config import HEATMAPS_DIR, REPORTS_DIR, SYSTEM_NAME, MODEL_VERSION
from backend.database.db import init_db
from backend.model_registry import register_all_models, run_startup_diagnostics
from backend.routes import (
    upload_router,
    results_router,
    evidence_router,
    reports_router,
    health_router
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TruthLens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events with pre-flight model diagnostics."""
    logger.info("Initializing TruthLens Database & Forensic Engine...")
    init_db()
    logger.info("Linking & Registering AI Forensic Models...")
    try:
        register_all_models()
        logger.info("Running AI Forensic Pre-Flight Detection Self-Check...")
        await run_startup_diagnostics()
    except Exception as e:
        logger.warning(f"Model registration or diagnostics encountered an issue: {e}")
    logger.info("TruthLens Evidence Guardian Online.")
    yield
    logger.info("TruthLens Shutting Down Cleanly.")


app = FastAPI(
    title=f"{SYSTEM_NAME} - AI Digital Forensics",
    description=(
        "Unified Multi-Signal Deepfake Detection, Evidence Guardian, "
        "and Cryptographic Verification Engine."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Frontend (Person 4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for Heatmaps and Forensic Reports
app.mount("/static/heatmaps", StaticFiles(directory=str(HEATMAPS_DIR)), name="heatmaps")
app.mount("/static/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# Register Routers
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(results_router)
app.include_router(evidence_router)
app.include_router(reports_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "system": SYSTEM_NAME,
        "version": MODEL_VERSION,
        "status": "OPERATIONAL",
        "description": "AI Digital Forensics & Deepfake Authentication Platform",
        "endpoints": {
            "web_ui": "/ui",
            "swagger_docs": "/docs",
            "health": "/health",
            "frequency_srm_diagnostic": "GET /diagnostics/frequency-srm",
            "upload": "POST /upload",
            "analyze": "POST /analyze",
            "result": "GET /result/{evidence_id}",
            "evidence": "GET /evidence/{evidence_id}",
            "report": "GET /report/{evidence_id}"
        }
    }


@app.get("/ui", tags=["UI"])
@app.get("/app", tags=["UI"])
async def serve_ui():
    """Serves the TruthLens single-page frontend application."""
    index_path = _DATA_DIR / "frontend" / "index.html"
    if not index_path.exists():
        index_path = _DATA_DIR / "index.html"
    return FileResponse(str(index_path), media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", app_dir=str(_CURRENT_DIR), host="0.0.0.0", port=8000, reload=True)
