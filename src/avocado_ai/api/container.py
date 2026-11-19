from __future__ import annotations

from dataclasses import dataclass

from avocado_ai.config import Settings
from avocado_ai.services.analysis import AnalysisService
from avocado_ai.services.feedback_store import FeedbackStore
from avocado_ai.services.session_store import SessionStore


@dataclass
class Container:
    """Long-lived application services, stored on `app.state.container`."""

    settings: Settings
    feedback: FeedbackStore
    sessions: SessionStore
    analysis: AnalysisService | None = None  # None when the model failed to load.
