"""Request and response models (the public API contract)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from avocado_ai.domain import ConfidenceLevel, RipenessStage
from avocado_ai.insights.base import Insight
from avocado_ai.ml.prediction import Prediction
from avocado_ai.services.analysis import AnalysisResult

# Session ids come from the client; keep them short and URL-safe.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class StageProbabilityOut(BaseModel):
    stage: int = Field(ge=1, le=5)
    label: str
    probability: float


class PredictionOut(BaseModel):
    stage: int = Field(ge=1, le=5, description="Ripening index, 1 (unripe) to 5 (overripe).")
    label: str = Field(description="Short user-facing stage name.")
    dataset_name: str = Field(description="Stage name used by the dataset.")
    confidence: float = Field(description="Top-1 probability.")
    confidence_level: ConfidenceLevel
    reliable: bool = Field(description="True when confidence >= the medium threshold.")
    class_uncertainty: float = Field(description="Mean MC-Dropout variance of the probabilities.")
    days_until_overripe: float
    days_uncertainty: float = Field(description="MC-Dropout standard deviation, in days.")
    probabilities: list[float] = Field(description="Distribution over stages 1..5.")
    top_candidates: list[StageProbabilityOut]
    message: str
    images_used: int
    warning: str | None = None

    @classmethod
    def from_domain(cls, p: Prediction) -> PredictionOut:
        return cls(
            stage=int(p.stage),
            label=p.stage.info.label,
            dataset_name=p.stage.info.dataset_name,
            confidence=round(p.confidence, 4),
            confidence_level=p.confidence_level,
            reliable=p.reliable,
            class_uncertainty=round(p.class_uncertainty, 6),
            days_until_overripe=round(p.days_until_overripe, 2),
            days_uncertainty=round(p.days_uncertainty, 2),
            probabilities=[round(x, 4) for x in p.probabilities],
            top_candidates=[
                StageProbabilityOut(
                    stage=int(c.stage),
                    label=c.stage.info.label,
                    probability=round(c.probability, 4),
                )
                for c in p.top_k(3)
            ],
            message=p.message,
            images_used=p.images_used,
            warning=p.warning,
        )


class InsightOut(BaseModel):
    title: str
    summary: str
    recommendation: str
    shelf_life: str
    alert: str | None
    actions: list[str]
    source: Literal["llm", "local"]

    @classmethod
    def from_domain(cls, insight: Insight) -> InsightOut:
        return cls(**insight.__dict__)


class AnalysisResponse(BaseModel):
    session_id: str
    image_id: str
    prediction: PredictionOut = Field(description="Result fused over every photo in the session.")
    image_prediction: PredictionOut = Field(description="Result for this photo alone.")
    insight: InsightOut | None = None

    @classmethod
    def from_domain(cls, result: AnalysisResult) -> AnalysisResponse:
        return cls(
            session_id=result.session_id,
            image_id=result.image_id,
            prediction=PredictionOut.from_domain(result.prediction),
            image_prediction=PredictionOut.from_domain(result.image_prediction),
            insight=InsightOut.from_domain(result.insight) if result.insight else None,
        )


class StageOut(BaseModel):
    stage: int
    label: str
    dataset_name: str


class ModelInfoOut(BaseModel):
    target: str
    stages: list[StageOut]
    mc_samples: int
    confidence_thresholds: dict[str, float]
    metrics: dict[str, float | None]
    device: str


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str


class FeedbackIn(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    image_id: str = Field(min_length=1, max_length=64)
    predicted_stage: int = Field(ge=1, le=5)
    is_correct: bool
    correct_stage: int | None = Field(default=None, ge=1, le=5)


class FeedbackOut(BaseModel):
    total_feedback: int


def all_stages() -> list[StageOut]:
    return [
        StageOut(stage=int(s), label=s.info.label, dataset_name=s.info.dataset_name)
        for s in RipenessStage
    ]
