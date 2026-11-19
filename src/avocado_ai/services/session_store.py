"""In-memory store of per-session predictions (one session = one avocado).

Bounded on every axis so it cannot grow without limit:
- sessions expire after `ttl_seconds` without activity;
- at most `max_sessions` are kept (least recently used are evicted);
- each session keeps only its latest `max_images` predictions.

It lives in the API process, so it is lost on restart and is not shared
between replicas. See docs/ROADMAP.md for the persistent alternative.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from avocado_ai.ml.prediction import Prediction


@dataclass
class _Session:
    predictions: list[Prediction] = field(default_factory=list)
    last_seen: float = 0.0


class SessionStore:
    def __init__(
        self,
        ttl_seconds: float = 3600,
        max_sessions: int = 1000,
        max_images: int = 10,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._max_images = max_images
        self._clock = clock
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.Lock()

    def append(self, session_id: str, prediction: Prediction) -> list[Prediction]:
        """Add a prediction and return a snapshot of the session's predictions."""
        with self._lock:
            now = self._clock()
            self._evict_expired(now)

            session = self._sessions.pop(session_id, None) or _Session()
            session.predictions.append(prediction)
            del session.predictions[: -self._max_images]
            session.last_seen = now
            self._sessions[session_id] = session  # (re)insert as most recent

            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

            return list(session.predictions)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired(self._clock())
            return len(self._sessions)

    def _evict_expired(self, now: float) -> None:
        # OrderedDict is ordered by last activity, so expired sessions are at the front.
        while self._sessions:
            oldest_id, oldest = next(iter(self._sessions.items()))
            if now - oldest.last_seen <= self._ttl:
                break
            del self._sessions[oldest_id]
