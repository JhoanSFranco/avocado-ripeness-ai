from __future__ import annotations

from fastapi import APIRouter, Depends

from avocado_ai import __version__
from avocado_ai.api.container import Container
from avocado_ai.api.dependencies import get_analysis_service, get_container
from avocado_ai.api.schemas import HealthOut, ModelInfoOut, all_stages
from avocado_ai.services.analysis import AnalysisService

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthOut)
def health(container: Container = Depends(get_container)) -> HealthOut:
    """Liveness probe. Reports `degraded` when the model is not loaded."""
    loaded = container.analysis is not None
    return HealthOut(
        status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__
    )


@router.get("/model", response_model=ModelInfoOut)
def model_info(service: AnalysisService = Depends(get_analysis_service)) -> ModelInfoOut:
    """Stages, thresholds and the evaluation metrics stored in the checkpoint."""
    predictor = service.predictor
    return ModelInfoOut(
        target=predictor.metadata.target_name,
        stages=all_stages(),
        mc_samples=predictor.mc_samples,
        confidence_thresholds={
            "high": predictor.thresholds.high,
            "medium": predictor.thresholds.medium,
        },
        metrics=predictor.metadata.metrics,
        device=str(predictor.device),
    )
