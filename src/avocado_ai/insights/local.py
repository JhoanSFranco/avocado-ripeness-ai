"""Deterministic, offline insights built from stage-specific rules."""

from __future__ import annotations

from avocado_ai.domain import ConfidenceLevel, ConfidenceThresholds, RipenessStage
from avocado_ai.insights.base import Insight
from avocado_ai.ml.prediction import Prediction, format_duration


class LocalInsightGenerator:
    def __init__(self, thresholds: ConfidenceThresholds) -> None:
        self.thresholds = thresholds

    def generate(self, prediction: Prediction) -> Insight:
        info = prediction.stage.info
        percent = f"{prediction.confidence * 100:.1f}%"
        level = self.thresholds.level(prediction.confidence)

        if level is ConfidenceLevel.HIGH:
            summary = (
                f"The model identifies the avocado as '{info.label}' with high confidence "
                f"({percent}). {info.description}"
            )
        elif level is ConfidenceLevel.MEDIUM:
            summary = (
                f"The model identifies the avocado as '{info.label}' with medium confidence "
                f"({percent}). Consider taking another photo or checking it by hand."
            )
        else:
            summary = (
                f"The prediction has low confidence ({percent}). Take a clearer photo "
                "before making a decision."
            )

        if prediction.stage is RipenessStage.OVERRIPE:
            shelf_life = "The fruit appears to be past its best point."
        else:
            shelf_life = (
                f"Roughly {format_duration(prediction.days_until_overripe)} until it is overripe "
                f"(± {prediction.days_uncertainty:.1f} days)."
            )

        return Insight(
            title=f"Avocado analysis: {info.label}",
            summary=summary,
            recommendation=info.recommendation,
            shelf_life=shelf_life,
            alert=info.alert,
            actions=list(info.actions),
            source="local",
        )
