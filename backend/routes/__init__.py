from backend.routes.upload import router as upload_router
from backend.routes.results import router as results_router
from backend.routes.evidence import router as evidence_router
from backend.routes.reports import router as reports_router
from backend.routes.health import router as health_router

__all__ = [
    "upload_router",
    "results_router",
    "evidence_router",
    "reports_router",
    "health_router"
]
