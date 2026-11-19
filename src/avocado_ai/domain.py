"""Domain vocabulary: ripeness stages, confidence levels and stage-specific advice.

The five stages follow the "Ripening Index Classification" of the
Hass Avocado Ripening Photographic Dataset (Mendeley Data, 3xd9n945v8).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class RipenessStage(IntEnum):
    """Ripening index (1-5). The model's class index is `stage - 1`."""

    UNDERRIPE = 1
    BREAKING = 2
    RIPE_FIRST_STAGE = 3
    RIPE_SECOND_STAGE = 4
    OVERRIPE = 5

    @classmethod
    def from_index(cls, index: int) -> RipenessStage:
        """Convert a zero-based model output index into a stage."""
        return cls(index + 1)

    @property
    def index(self) -> int:
        """Zero-based index used by the classifier head."""
        return self.value - 1

    @property
    def info(self) -> StageInfo:
        return STAGES[self]


NUM_STAGES = len(RipenessStage)


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Top-1 probability thresholds used to grade a prediction."""

    high: float = 0.70
    medium: float = 0.50

    def level(self, confidence: float) -> ConfidenceLevel:
        if confidence >= self.high:
            return ConfidenceLevel.HIGH
        if confidence >= self.medium:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def is_reliable(self, confidence: float) -> bool:
        """A prediction is reliable enough to act on from MEDIUM confidence upwards."""
        return confidence >= self.medium


@dataclass(frozen=True)
class StageInfo:
    label: str  # Short, user-facing name.
    dataset_name: str  # Official name in the dataset.
    description: str
    recommendation: str
    alert: str | None
    actions: tuple[str, ...]


STAGES: dict[RipenessStage, StageInfo] = {
    RipenessStage.UNDERRIPE: StageInfo(
        label="Unripe",
        dataset_name="Underripe",
        description="The avocado looks unripe and is not ready to eat yet.",
        recommendation="Let it ripen at room temperature before eating it.",
        alert=None,
        actions=(
            "Keep it at room temperature to help it ripen.",
            "Check its firmness again in 1 to 2 days.",
            "Avoid refrigerating it before it is ripe.",
        ),
    ),
    RipenessStage.BREAKING: StageInfo(
        label="Breaking",
        dataset_name="Breaking",
        description="The avocado has started ripening but is not at its best yet.",
        recommendation="Check it again soon to see whether it is ready.",
        alert=None,
        actions=(
            "Keep it at room temperature to help it ripen.",
            "Check its firmness again in 1 to 2 days.",
            "Avoid refrigerating it before it is ripe.",
        ),
    ),
    RipenessStage.RIPE_FIRST_STAGE: StageInfo(
        label="Ripe",
        dataset_name="Ripe (first stage)",
        description="The avocado looks ripe and ready to eat.",
        recommendation="This is a good time to eat it.",
        alert=None,
        actions=(
            "Eat it soon.",
            "Refrigerate it to slow down further ripening.",
            "Avoid bumping or squeezing the fruit.",
        ),
    ),
    RipenessStage.RIPE_SECOND_STAGE: StageInfo(
        label="Very ripe",
        dataset_name="Ripe (second stage)",
        description="The avocado looks close to the end of its shelf life.",
        recommendation="Eat it as soon as possible to avoid further spoilage.",
        alert="Close to the end of its shelf life.",
        actions=(
            "Eat it as soon as possible.",
            "Check for soft spots or internal damage.",
            "Do not keep it at room temperature for several more days.",
        ),
    ),
    RipenessStage.OVERRIPE: StageInfo(
        label="Overripe",
        dataset_name="Overripe",
        description="The avocado looks past its best point for eating.",
        recommendation="Check its smell, texture and flesh carefully before eating it.",
        alert="Possible advanced spoilage.",
        actions=(
            "Cut it open and check the color, smell and texture of the flesh.",
            "Discard it if it smells bad or shows obvious damage.",
            "Do not rely on the prediction alone if the fruit looks doubtful.",
        ),
    ),
}
