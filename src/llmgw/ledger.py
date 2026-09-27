"""Usage ledger and response cache, in one SQLite file.

The ledger is what makes budgets and chargeback possible: every request is recorded with its
team, model, tokens, cost, latency and outcome. Prompt and answer text are NOT stored, so the
ledger can be shared with finance without exposing content.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    ts REAL NOT NULL,
    month TEXT NOT NULL,
    team TEXT NOT NULL,
    request_id TEXT NOT NULL,
    model TEXT NOT NULL,
    route_reason TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    cache_hit INTEGER NOT NULL,
    pii_masked INTEGER NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_team_month ON usage (team, month);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    created REAL NOT NULL,
    value TEXT NOT NULL
);
"""


def month_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


@dataclass
class UsageEntry:
    ts: float
    team: str
    request_id: str
    model: str
    route_reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    cache_hit: bool
    pii_masked: int
    status: str  # "ok", "blocked_budget", "blocked_policy", "provider_error"


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        """Open, commit on success, and always close (open handles lock files on Windows)."""
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def record(self, entry: UsageEntry) -> None:
        row = asdict(entry)
        row["month"] = month_of(entry.ts)
        row["cache_hit"] = int(entry.cache_hit)
        columns = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        with self._lock, self._connect() as db:
            db.execute(f"INSERT INTO usage ({columns}) VALUES ({marks})", row)

    def spent(self, team: str, month: str) -> float:
        with self._connect() as db:
            (total,) = db.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM usage WHERE team = ? AND month = ?",
                (team, month),
            ).fetchone()
        return float(total)

    def monthly_report(self, month: str) -> list[dict]:
        """One row per team and model: the basis for internal chargeback."""
        query = """
            SELECT team, model, COUNT(*) AS requests,
                   SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                   ROUND(SUM(cost_usd), 4) AS cost_usd, SUM(cache_hit) AS cache_hits,
                   SUM(pii_masked) AS pii_values_masked,
                   SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS refused_or_failed
            FROM usage WHERE month = ? GROUP BY team, model ORDER BY team, model
        """
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(query, (month,))]

    # ------------------------------------------------------------------ cache

    def cache_get(self, key: str, ttl_seconds: int) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT created, value FROM cache WHERE key = ?", (key,)).fetchone()
        if row and time.time() - row[0] <= ttl_seconds:
            return json.loads(row[1])
        return None

    def cache_put(self, key: str, value: dict) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO cache (key, created, value) VALUES (?, ?, ?)",
                (key, time.time(), json.dumps(value)),
            )
