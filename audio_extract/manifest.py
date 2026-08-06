"""SQLite manifest — the authoritative per-track record store (docs/v2 §"records").

Holds the Candidate / Passage / Metric rows plus a job-state row. The queue and
this manifest are authoritative; a Pi/DeepSeek session id is advisory metadata.
Every write is idempotent (``INSERT OR REPLACE`` on natural keys).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

# docs/v2 §6.5 queue states.
QUEUE_STATES = (
    "INGESTED",
    "MINING_PASSAGES",
    "SCREENING",
    "MEASURING",
    "PLANNING_REFINEMENT",
    "REFINING",
    "AWAITING_HUMAN",       # audit mode only; autonomous production never enters it
    "FINALIST_SELECTED",    # v2.1 §19.6: a conductor/selector decision is NOT completion
    "RENDERING_FINALISTS",
    "RENDERING_DELIVERY",
    "FINAL_QC",
    "COMPLETE",
    "FAILED",
    "CANCELLED",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate (
    recipe_id           TEXT PRIMARY KEY,
    operation           TEXT NOT NULL,
    parents             TEXT NOT NULL,          -- json array of recipe_ids
    artifact_pcm_sha256 TEXT,
    sample_rate_hz      INTEGER,
    channels            TEXT,                   -- json array
    frames              INTEGER,
    sample_format       TEXT,
    status              TEXT NOT NULL,
    created_at          TEXT
);
CREATE TABLE IF NOT EXISTS passage (
    passage_id   TEXT PRIMARY KEY,
    start_sample INTEGER NOT NULL,
    end_sample   INTEGER NOT NULL,
    tags         TEXT NOT NULL,                 -- json array
    features     TEXT NOT NULL,                 -- json object
    detectors    TEXT NOT NULL                  -- json object
);
CREATE TABLE IF NOT EXISTS metric (
    recipe_id  TEXT NOT NULL,
    passage_id TEXT NOT NULL,
    metric     TEXT NOT NULL,
    value      REAL,
    unit       TEXT,
    details    TEXT,                            -- json object
    PRIMARY KEY (recipe_id, passage_id, metric)
);
CREATE TABLE IF NOT EXISTS job (
    track_id   TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    pi_session TEXT,
    updated_at TEXT
);
"""


class Manifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    # --- candidates ------------------------------------------------------
    def upsert_candidate(self, rec: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO candidate
               (recipe_id, operation, parents, artifact_pcm_sha256, sample_rate_hz,
                channels, frames, sample_format, status, created_at)
               VALUES (:recipe_id,:operation,:parents,:artifact_pcm_sha256,:sample_rate_hz,
                       :channels,:frames,:sample_format,:status,:created_at)""",
            {
                "recipe_id": rec["recipe_id"],
                "operation": rec["operation"],
                "parents": json.dumps(rec.get("parents", [])),
                "artifact_pcm_sha256": rec.get("artifact_pcm_sha256"),
                "sample_rate_hz": rec.get("sample_rate_hz"),
                "channels": json.dumps(rec.get("channels", [])),
                "frames": rec.get("frames"),
                "sample_format": rec.get("sample_format"),
                "status": rec.get("status", "complete"),
                "created_at": rec.get("created_at"),
            },
        )
        self._conn.commit()

    def get_candidate(self, recipe_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM candidate WHERE recipe_id=?", (recipe_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["parents"] = json.loads(d["parents"])
        d["channels"] = json.loads(d["channels"])
        return d

    def list_candidates(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT recipe_id FROM candidate ORDER BY created_at").fetchall()
        return [r["recipe_id"] for r in rows]

    # --- passages --------------------------------------------------------
    def upsert_passages(self, passages: Iterable[dict[str, Any]]) -> None:
        for p in passages:
            self._conn.execute(
                """INSERT OR REPLACE INTO passage
                   (passage_id, start_sample, end_sample, tags, features, detectors)
                   VALUES (:passage_id,:start_sample,:end_sample,:tags,:features,:detectors)""",
                {
                    "passage_id": p["passage_id"],
                    "start_sample": p["start_sample"],
                    "end_sample": p["end_sample"],
                    "tags": json.dumps(p.get("tags", [])),
                    "features": json.dumps(p.get("features", {})),
                    "detectors": json.dumps(p.get("detectors", {})),
                },
            )
        self._conn.commit()

    # --- metrics ---------------------------------------------------------
    def upsert_metric(self, m: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO metric
               (recipe_id, passage_id, metric, value, unit, details)
               VALUES (:recipe_id,:passage_id,:metric,:value,:unit,:details)""",
            {
                "recipe_id": m["recipe_id"],
                "passage_id": m["passage_id"],
                "metric": m["metric"],
                "value": m.get("value"),
                "unit": m.get("unit"),
                "details": json.dumps(m.get("details", {})),
            },
        )
        self._conn.commit()

    # --- job state -------------------------------------------------------
    def set_state(self, track_id: str, state: str, pi_session: str | None = None,
                  updated_at: str | None = None) -> None:
        if state not in QUEUE_STATES:
            raise ValueError(f"unknown queue state: {state!r}")
        self._conn.execute(
            """INSERT OR REPLACE INTO job (track_id, state, pi_session, updated_at)
               VALUES (?,?,?,?)""",
            (track_id, state, pi_session, updated_at),
        )
        self._conn.commit()

    def get_state(self, track_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM job WHERE track_id=?", (track_id,)).fetchone()
        return dict(row) if row else None
