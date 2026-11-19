"""Turn a structured prediction into human-readable advice."""

from avocado_ai.insights.base import Insight, InsightGenerator
from avocado_ai.insights.factory import build_insight_generator
from avocado_ai.insights.groq import GroqInsightGenerator
from avocado_ai.insights.local import LocalInsightGenerator

__all__ = [
    "GroqInsightGenerator",
    "Insight",
    "InsightGenerator",
    "LocalInsightGenerator",
    "build_insight_generator",
]
