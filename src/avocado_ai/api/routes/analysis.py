from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from avocado_ai.api.container import Container
from avocado_ai.api.dependencies import get_analysis_service, get_container
from avocado_ai.api.schemas import SESSION_ID_PATTERN, AnalysisResponse
from avocado_ai.services.analysis import AnalysisService, InvalidImageError

router = APIRouter(tags=["analysis"])


# A plain `def` endpoint: FastAPI runs it in a worker thread, so the
# CPU-bound PyTorch inference does not block the event loop.
@router.post("/analyze", response_model=AnalysisResponse)
def analyze(
    file: Annotated[UploadFile, File(description="Photo of a single avocado.")],
    session_id: Annotated[
        str | None,
        Form(
            pattern=SESSION_ID_PATTERN,
            description="Photos sent with the same session id are treated as the same "
            "avocado and their evidence is combined. Omit it to start a new session.",
        ),
    ] = None,
    service: AnalysisService = Depends(get_analysis_service),
    container: Container = Depends(get_container),
) -> AnalysisResponse:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "The file must be an image.")

    max_bytes = container.settings.max_upload_bytes
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            413,  # Content Too Large
            f"Images must be smaller than {container.settings.max_upload_mb:g} MB.",
        )

    try:
        result = service.analyze(data, session_id or uuid.uuid4().hex)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return AnalysisResponse.from_domain(result)
