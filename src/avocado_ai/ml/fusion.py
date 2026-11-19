"""Fuse the predictions of several photos of the *same* avocado.

Classification
    Each photo gives a distribution P(stage | photo_i). Assuming the photos are
    conditionally independent given the true stage and a uniform prior, the
    joint posterior is proportional to the product of the individual
    distributions. It is computed as a sum of log-probabilities for numerical
    stability. Stages are never averaged as numbers: "ripe (3)" and
    "overripe (5)" do not average to "very ripe (4)".

Regression
    The remaining-days estimates are averaged, weighting each photo by the
    probability it assigned to the final fused stage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from avocado_ai.domain import ConfidenceThresholds, RipenessStage
from avocado_ai.ml.prediction import Prediction, build_message

_EPS = 1e-8
# If individual photos disagree by this many stages or more, flag it.
HETEROGENEITY_SPREAD = 3


def fuse_predictions(
    predictions: Sequence[Prediction], thresholds: ConfidenceThresholds
) -> Prediction:
    if not predictions:
        raise ValueError("At least one prediction is required")
    if len(predictions) == 1:
        return predictions[0]

    n = len(predictions)
    probs = np.array([p.probabilities for p in predictions], dtype=float)  # (N, C)

    # --- 1. Class posterior: normalized product of distributions --------------
    log_posterior = np.log(np.clip(probs, _EPS, 1.0)).sum(axis=0)
    fused = np.exp(log_posterior - log_posterior.max())
    fused /= fused.sum()

    top_index = int(fused.argmax())
    stage = RipenessStage.from_index(top_index)
    confidence = float(fused[top_index])

    # --- 2. Regression: weight each photo by its support for the final stage --
    weights = probs[:, top_index]
    if weights.sum() < _EPS:
        weights = np.ones(n)
    weights = weights / weights.sum()

    days = float(np.dot(weights, [p.days_until_overripe for p in predictions]))
    days_uncertainty = float(np.dot(weights, [p.days_uncertainty for p in predictions]))
    if stage is RipenessStage.OVERRIPE:
        days = 0.0

    # --- 3. Consistency check --------------------------------------------------
    stages = [int(p.stage) for p in predictions]
    warning = None
    if max(stages) - min(stages) >= HETEROGENEITY_SPREAD:
        warning = (
            "The photos show very different ripeness levels. The avocado may be ripening "
            "unevenly, or lighting/angle differs between photos."
        )

    return Prediction(
        stage=stage,
        confidence=confidence,
        confidence_level=thresholds.level(confidence),
        reliable=thresholds.is_reliable(confidence),
        class_uncertainty=float(np.mean([p.class_uncertainty for p in predictions])),
        days_until_overripe=days,
        days_uncertainty=days_uncertainty,
        probabilities=tuple(float(x) for x in fused),
        message=build_message(stage, confidence, days, thresholds, images_used=n),
        images_used=n,
        warning=warning,
    )
