from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response, status

from avocado_ai.api.container import Container
from avocado_ai.api.dependencies import get_container
from avocado_ai.api.schemas import SESSION_ID_PATTERN

router = APIRouter(tags=["sessions"])


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def clear_session(
    session_id: Annotated[str, Path(pattern=SESSION_ID_PATTERN)],
    container: Container = Depends(get_container),
) -> Response:
    """Forget the photos of a session (idempotent)."""
    container.sessions.clear(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
