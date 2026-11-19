from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from avocado_ai.ml.prediction import Prediction


@dataclass(frozen=True)
class Insight:
    title: str
    summary: str
    recommendation: str
    shelf_life: str
    alert: str | None
    actions: list[str] = field(default_factory=list)
    source: Literal["llm", "local"] = "local"


class InsightGenerator(Protocol):
    def generate(self, prediction: Prediction) -> Insight: ...
