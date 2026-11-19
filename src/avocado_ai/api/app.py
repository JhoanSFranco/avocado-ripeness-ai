"""FastAPI application factory.

Run with:
    uvicorn avocado_ai.api.app:create_app --factory --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from avocado_ai import __version__
from avocado_ai.api.container import Container
from avocado_ai.api.routes import analysis, feedback, health, sessions
from avocado_ai.config import Settings, get_settings
from avocado_ai.insights import build_insight_generator
from avocado_ai.ml.predictor import Predictor, RipenessPredictor
from avocado_ai.services.analysis import AnalysisService
from avocado_ai.services.feedback_store import FeedbackStore
from avocado_ai.services.session_store import SessionStore

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Configuration; defaults to environment-based settings.
        predictor: Pre-built predictor (used by tests). When omitted, the model
            checkpoint is loaded at startup from `settings.model_path`.
    """
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        sessions = SessionStore(
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_sessions,
            max_images=settings.max_images_per_session,
        )
        container = Container(
            settings=settings,
            feedback=FeedbackStore(settings.feedback_path),
            sessions=sessions,
        )

        loaded = predictor
        if loaded is None:
            try:
                loaded = RipenessPredictor.from_settings(settings)
            except Exception:
                # Keep the API up (health endpoint reports it) instead of crash-looping.
                logger.exception("Model could not be loaded from %s", settings.model_path)

        if loaded is not None:
            container.analysis = AnalysisService(
                predictor=loaded,
                sessions=sessions,
                insights=build_insight_generator(settings),
            )

        app.state.container = container
        yield

    app = FastAPI(
        title="Avocado Ripeness API",
        description=(
            "Classifies the ripeness stage of Hass avocados from photos and estimates "
            "the remaining days until overripe, with Monte Carlo Dropout uncertainty."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    for router in (health.router, analysis.router, sessions.router, feedback.router):
        app.include_router(router, prefix=API_PREFIX)

    return app
