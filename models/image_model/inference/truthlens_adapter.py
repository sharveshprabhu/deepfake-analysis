"""
TruthLens Official Visual AI Adapter Entrypoint.
Conforms directly to MODEL_INTEGRATION_GUIDE.md and backend orchestrator specifications.
"""

from .image_model_adapter import (
    my_visual_model,
    truthlens_visual_adapter,
    global_visual_pipeline,
    VisualForensicsPipeline
)

__all__ = [
    "my_visual_model",
    "truthlens_visual_adapter",
    "global_visual_pipeline",
    "VisualForensicsPipeline"
]
