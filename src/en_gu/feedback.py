"""User feedback on translations, kept in a small SQLite file.

Privacy: a thumbs up/down stores no text at all, only the request id, rating and model version. Text (the English
source, the model output and the user's correction) is stored only when the user explicitly submits a correction,
and rows older than the retention period are deleted. These corrections are the raw material for the retraining
loop (reviewed, validated and deduplicated before they get anywhere near training data).
"""

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    request_id TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    model_version TEXT,
    source_text TEXT,
    translation TEXT,
    corrected_translation TEXT
);
CREATE INDEX IF NOT EXISTS feedback_created_at ON feedback (created_at);
CREATE INDEX IF NOT EXISTS feedback_model_version ON feedback (model_version);
"""


class FeedbackStore:
    def __init__(self, path, retention_days=None):
        self.path = Path(path)
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, request_id, rating, model_version=None, source_text=None, translation=None, corrected=None):
        now = datetime.now(UTC)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO feedback (created_at, request_id, rating, model_version, source_text, translation,"
                " corrected_translation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now.isoformat(), request_id, rating, model_version, source_text, translation, corrected),
            )
            if self.retention_days:
                cutoff = (now - timedelta(days=self.retention_days)).isoformat()
                conn.execute("DELETE FROM feedback WHERE created_at < ?", (cutoff,))
            return cur.lastrowid

    def summary(self):
        """Counts per model version: {version: {"up": n, "down": n, "corrections": n}}."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_version, rating, COUNT(*), COUNT(corrected_translation) FROM feedback"
                " GROUP BY model_version, rating"
            ).fetchall()
        out = {}
        for version, rating, count, corrections in rows:
            entry = out.setdefault(version, {"up": 0, "down": 0, "corrections": 0})
            entry[rating] += count
            entry["corrections"] += corrections
        return out

    def rows(self):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM feedback ORDER BY id")]
