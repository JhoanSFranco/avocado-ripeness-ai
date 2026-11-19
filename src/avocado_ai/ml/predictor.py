"""Single-image inference: preprocessing + MC Dropout + post-processing."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from PIL import Image

from avocado_ai.config import Settings
from avocado_ai.domain import ConfidenceThresholds, RipenessStage
from avocado_ai.ml.model import AvocadoRipenessNet, ModelMetadata, load_model, resolve_device
from avocado_ai.ml.prediction import Prediction, build_message
from avocado_ai.ml.transforms import eval_transform


class Predictor(Protocol):
    """Interface used by the service layer (lets tests plug in a fake)."""

    thresholds: ConfidenceThresholds
    metadata: ModelMetadata
    device: torch.device
    mc_samples: int

    def predict(self, image: Image.Image) -> Prediction: ...


class RipenessPredictor:
    def __init__(
        self,
        model: AvocadoRipenessNet,
        metadata: ModelMetadata,
        device: torch.device,
        mc_samples: int = 8,
        thresholds: ConfidenceThresholds | None = None,
    ) -> None:
        self.model = model.eval()
        self.metadata = metadata
        self.device = device
        self.mc_samples = mc_samples
        self.thresholds = thresholds or ConfidenceThresholds()
        self._transform = eval_transform()

    @classmethod
    def from_settings(cls, settings: Settings) -> RipenessPredictor:
        device = resolve_device(settings.device)
        model, metadata = load_model(settings.model_path, device)
        return cls(
            model=model,
            metadata=metadata,
            device=device,
            mc_samples=settings.mc_samples,
            thresholds=ConfidenceThresholds(settings.confidence_high, settings.confidence_medium),
        )

    def predict(self, image: Image.Image) -> Prediction:
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        output = self.model.predict_with_uncertainty(tensor, num_samples=self.mc_samples)

        probabilities = output.probs_mean[0].cpu().tolist()
        top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        stage = RipenessStage.from_index(top_index)
        confidence = probabilities[top_index]

        # Regression: standardized -> days; variance -> standard deviation in days.
        days = self.metadata.denormalize_days(output.regression_mean[0].item())
        days_std = math.sqrt(max(output.regression_var[0].item(), 0.0)) * self.metadata.target_std

        # Physical constraints: no negative time left, nothing left once overripe.
        days = 0.0 if stage is RipenessStage.OVERRIPE else max(days, 0.0)

        return Prediction(
            stage=stage,
            confidence=confidence,
            confidence_level=self.thresholds.level(confidence),
            reliable=self.thresholds.is_reliable(confidence),
            class_uncertainty=output.probs_var[0].mean().item(),
            days_until_overripe=days,
            days_uncertainty=days_std,
            probabilities=tuple(probabilities),
            message=build_message(stage, confidence, days, self.thresholds),
        )
