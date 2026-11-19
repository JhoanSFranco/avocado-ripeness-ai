"""Prediction value object and the user-facing message rules shared by
single-image inference and multi-image fusion."""

from __future__ import annotations

from dataclasses import dataclass

from avocado_ai.domain import ConfidenceLevel, ConfidenceThresholds, RipenessStage


@dataclass(frozen=True)
class StageProbability:
    stage: RipenessStage
    probability: float


@dataclass(frozen=True)
class Prediction:
    stage: RipenessStage
    confidence: float  # Top-1 probability.
    confidence_level: ConfidenceLevel
    reliable: bool  # Confidence high enough to act on (>= medium threshold).
    class_uncertainty: float  # Mean MC-Dropout variance of the class probabilities.
    days_until_overripe: float  # Estimated remaining days (>= 0).
    days_uncertainty: float  # MC-Dropout standard deviation, in days.
    probabilities: tuple[float, ...]  # Full class distribution (index = stage - 1).
    message: str
    images_used: int = 1
    warning: str | None = None

    def top_k(self, k: int = 3) -> list[StageProbability]:
        ranked = sorted(enumerate(self.probabilities), key=lambda item: item[1], reverse=True)
        return [StageProbability(RipenessStage.from_index(i), p) for i, p in ranked[:k]]


def format_duration(days: float) -> str:
    """Render remaining time; under one day it is shown in hours."""
    if days < 1:
        hours = max(1, round(days * 24))
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{days:.1f} days"


def build_message(
    stage: RipenessStage,
    confidence: float,
    days: float,
    thresholds: ConfidenceThresholds,
    images_used: int = 1,
) -> str:
    """Short, human-readable summary of a prediction."""
    prefix = f"Combined result ({images_used} images): " if images_used > 1 else ""
    label = stage.info.label
    remaining = format_duration(days)
    level = thresholds.level(confidence)

    if level is ConfidenceLevel.LOW:
        return prefix + "Low confidence. Please upload another, clearer photo."
    if stage is RipenessStage.OVERRIPE:
        return prefix + "The avocado appears to be past its shelf life."
    if level is ConfidenceLevel.MEDIUM:
        return (
            prefix
            + f"Medium confidence: {label}. About {remaining} left. Another photo is recommended."
        )
    if days < 0.5:
        return prefix + f"High confidence: {label}. Eat it within a few hours."
    if days < 2:
        return prefix + f"High confidence: {label}. About {remaining} left. Eat it soon."
    return prefix + f"High confidence: {label}. About {remaining} left."
