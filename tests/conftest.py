from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from avocado_ai.domain import ConfidenceThresholds, RipenessStage
from avocado_ai.ml.model import ModelMetadata
from avocado_ai.ml.prediction import Prediction, build_message

THRESHOLDS = ConfidenceThresholds(high=0.70, medium=0.50)


def make_prediction(probabilities, days: float = 5.0, days_std: float = 0.5) -> Prediction:
    """Build a Prediction from a 5-value probability list."""
    probs = tuple(float(p) for p in probabilities)
    index = max(range(len(probs)), key=probs.__getitem__)
    stage = RipenessStage.from_index(index)
    confidence = probs[index]
    return Prediction(
        stage=stage,
        confidence=confidence,
        confidence_level=THRESHOLDS.level(confidence),
        reliable=THRESHOLDS.is_reliable(confidence),
        class_uncertainty=0.001,
        days_until_overripe=0.0 if stage is RipenessStage.OVERRIPE else days,
        days_uncertainty=days_std,
        probabilities=probs,
        message=build_message(stage, confidence, days, THRESHOLDS),
    )


class FakePredictor:
    """Deterministic stand-in for RipenessPredictor (no model weights needed)."""

    def __init__(self, probabilities=(0.05, 0.05, 0.80, 0.05, 0.05)) -> None:
        self.probabilities = probabilities
        self.thresholds = THRESHOLDS
        self.metadata = ModelMetadata(
            target_mean=6.2, target_std=6.4, metrics={"test_accuracy": 0.79}
        )
        self.device = torch.device("cpu")
        self.mc_samples = 8
        self.calls = 0

    def predict(self, image: Image.Image) -> Prediction:
        self.calls += 1
        return make_prediction(self.probabilities)


@pytest.fixture
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(60, 120, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()
