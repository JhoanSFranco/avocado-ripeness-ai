from __future__ import annotations

from fastapi import APIRouter, Depends, status

from avocado_ai.api.container import Container
from avocado_ai.api.dependencies import get_container
from avocado_ai.api.schemas import FeedbackIn, FeedbackOut
from avocado_ai.services.feedback_store import FeedbackRecord

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
def submit_feedback(body: FeedbackIn, container: Container = Depends(get_container)) -> FeedbackOut:
    """Record whether a prediction was correct (and the right stage if not)."""
    correct_stage = body.predicted_stage if body.is_correct else body.correct_stage
    total = container.feedback.add(
        FeedbackRecord(
            session_id=body.session_id,
            image_id=body.image_id,
            predicted_stage=body.predicted_stage,
            is_correct=body.is_correct,
            correct_stage=correct_stage,
        )
    )
    return FeedbackOut(total_feedback=total)
