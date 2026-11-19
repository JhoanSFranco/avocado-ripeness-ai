"""FastAPI dependency providers."""

from fastapi import HTTPException, Request, status

from avocado_ai.api.container import Container
from avocado_ai.services.analysis import AnalysisService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_analysis_service(request: Request) -> AnalysisService:
    service = get_container(request).analysis
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not loaded. Check MODEL_PATH and the API logs.",
        )
    return service
