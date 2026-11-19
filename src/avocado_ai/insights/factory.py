from __future__ import annotations

import logging

from avocado_ai.config import Settings
from avocado_ai.domain import ConfidenceThresholds
from avocado_ai.insights.base import InsightGenerator
from avocado_ai.insights.groq import GroqInsightGenerator
from avocado_ai.insights.local import LocalInsightGenerator

logger = logging.getLogger(__name__)


def build_insight_generator(settings: Settings) -> InsightGenerator:
    """Use the LLM when an API key is configured, otherwise the local rules."""
    thresholds = ConfidenceThresholds(settings.confidence_high, settings.confidence_medium)
    local = LocalInsightGenerator(thresholds)

    api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    if not api_key:
        logger.info("GROQ_API_KEY not set: using local rule-based insights")
        return local

    logger.info("Using Groq model %s for insights", settings.groq_model)
    return GroqInsightGenerator(
        api_key=api_key,
        fallback=local,
        model=settings.groq_model,
        timeout_seconds=settings.groq_timeout_seconds,
    )
