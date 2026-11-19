"""Analysis use case: image bytes -> prediction -> session fusion -> insight."""

from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from avocado_ai.insights.base import Insight, InsightGenerator
from avocado_ai.ml.fusion import fuse_predictions
from avocado_ai.ml.prediction import Prediction
from avocado_ai.ml.predictor import Predictor
from avocado_ai.services.session_store import SessionStore

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    """The uploaded bytes are not a readable image."""


@dataclass(frozen=True)
class AnalysisResult:
    session_id: str
    image_id: str
    image_prediction: Prediction  # This photo alone.
    prediction: Prediction  # Fused over every photo in the session.
    insight: Insight | None  # Only produced for reliable predictions.


class AnalysisService:
    def __init__(
        self,
        predictor: Predictor,
        sessions: SessionStore,
        insights: InsightGenerator,
    ) -> None:
        self.predictor = predictor
        self.sessions = sessions
        self.insights = insights

    def analyze(self, image_bytes: bytes, session_id: str) -> AnalysisResult:
        image = decode_image(image_bytes)
        image_prediction = self.predictor.predict(image)

        session_predictions = self.sessions.append(session_id, image_prediction)
        prediction = fuse_predictions(session_predictions, self.predictor.thresholds)

        insight = self.insights.generate(prediction) if prediction.reliable else None

        logger.info(
            "session=%s images=%d stage=%s confidence=%.3f",
            session_id,
            prediction.images_used,
            prediction.stage.name,
            prediction.confidence,
        )
        return AnalysisResult(
            session_id=session_id,
            image_id=uuid.uuid4().hex,
            image_prediction=image_prediction,
            prediction=prediction,
            insight=insight,
        )


def decode_image(data: bytes) -> Image.Image:
    """Decode and validate an uploaded image (Pillow's decompression-bomb
    guard is active by default)."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The file could not be read as an image.") from exc
