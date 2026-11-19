"""Append-only CSV store for user feedback on predictions.

Feedback rows can later be joined with the stored images to build a
relabeled dataset for retraining (see docs/ROADMAP.md).
"""

from __future__ import annotations

import csv
import threading
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class FeedbackRecord:
    session_id: str
    image_id: str
    predicted_stage: int
    is_correct: bool
    correct_stage: int | None = None
    timestamp: str = ""


_FIELDNAMES = [f.name for f in fields(FeedbackRecord)]


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, record: FeedbackRecord) -> int:
        """Persist a record and return the total number of records."""
        row = asdict(record)
        row["timestamp"] = record.timestamp or datetime.now(timezone.utc).isoformat()

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
            return self._count_unlocked()

    def count(self) -> int:
        with self._lock:
            return self._count_unlocked()

    def _count_unlocked(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)  # minus header
