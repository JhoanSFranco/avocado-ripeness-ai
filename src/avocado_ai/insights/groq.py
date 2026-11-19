"""LLM-generated insights through Groq's OpenAI-compatible chat API.

Any failure (network, HTTP error, malformed JSON) falls back to the local
generator, so the endpoint never fails because of the LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from avocado_ai.insights.base import Insight, InsightGenerator
from avocado_ai.ml.prediction import Prediction

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an expert in Hass avocado ripening and shelf life. You explain the output "
    "of a computer-vision model to a consumer in plain English. Be concise and practical, "
    "and never contradict the model's predicted stage."
)

USER_PROMPT_TEMPLATE = """A computer-vision model analyzed a photo of a Hass avocado:
- Predicted stage: {stage} ({dataset_name}), stage {stage_number} of 5
- Confidence (top-1 probability): {confidence:.3f}
- Classification uncertainty (MC Dropout variance): {class_uncertainty:.6f}
- Estimated days until overripe: {days:.2f} (standard deviation {days_std:.2f} days)
- Photos combined: {images_used}

Stage scale: 1 unripe, 2 breaking, 3 ripe, 4 very ripe, 5 overripe.

Reply ONLY with a JSON object with exactly these keys:
{{
  "title": string,
  "summary": string,
  "recommendation": string,
  "shelf_life": string,
  "alert": string or null,
  "actions": [string, string, string]
}}"""


class GroqInsightGenerator:
    def __init__(
        self,
        api_key: str,
        fallback: InsightGenerator,
        model: str = "llama-3.3-70b-versatile",
        timeout_seconds: float = 20.0,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._fallback = fallback
        self._model = model
        self._timeout = timeout_seconds
        self._http = session or requests.Session()

    def generate(self, prediction: Prediction) -> Insight:
        try:
            content = self._complete(self._build_prompt(prediction))
            return self._parse(content)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("LLM insight failed, using local fallback: %s", exc)
            return self._fallback.generate(prediction)

    @staticmethod
    def _build_prompt(prediction: Prediction) -> str:
        info = prediction.stage.info
        return USER_PROMPT_TEMPLATE.format(
            stage=info.label,
            dataset_name=info.dataset_name,
            stage_number=int(prediction.stage),
            confidence=prediction.confidence,
            class_uncertainty=prediction.class_uncertainty,
            days=prediction.days_until_overripe,
            days_std=prediction.days_uncertainty,
            images_used=prediction.images_used,
        )

    def _complete(self, prompt: str) -> str:
        response = self._http.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 500,
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(content: str) -> Insight:
        data: dict[str, Any] = json.loads(_strip_code_fence(content))
        actions = data.get("actions") or []
        if not isinstance(actions, list):
            raise ValueError("'actions' must be a list")
        return Insight(
            title=str(data["title"]),
            summary=str(data["summary"]),
            recommendation=str(data["recommendation"]),
            shelf_life=str(data["shelf_life"]),
            alert=str(data["alert"]) if data.get("alert") else None,
            actions=[str(a) for a in actions][:5],
            source="llm",
        )


def _strip_code_fence(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
