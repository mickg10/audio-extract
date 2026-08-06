"""Data model v2 (v2.1 WP2 / §7): append-only facts, explicit roles + cohorts,
challenge/judge/selector records, restart-safe budgets.

Key semantics differences from v1 (`manifest.py`, kept for compatibility):

* immutable facts use ``INSERT ... ON CONFLICT DO NOTHING`` and then verify the
  stored row equals the offered row — a completed fact is never silently replaced
  (SQLite ``REPLACE`` deletes + reinserts, which violates append-only);
* mutable execution/queue state has explicit transition updates;
* observations carry ``available`` — missing evidence is recorded as unavailable,
  never as a best-possible ``0.0``;
* only nodes in the same ``cohort`` may enter one consensus/ranking operation;
* WAL + busy timeout so the queue and web processes can share the file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

ROLES = ("source", "vocal", "accompaniment", "auxiliary",
         "challenge_mixture", "challenge_target", "measurement_copy", "delivery")

COHORTS = ("instrumental-production-v1", "vocal-evaluation-v1",
           "challenge-output-v1", "delivery-v1")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifact_node (
    node_id              TEXT PRIMARY KEY,
    node_type            TEXT NOT NULL,
    role                 TEXT NOT NULL,
    cohort               TEXT,
    operation            TEXT NOT NULL,
    parents_json         TEXT NOT NULL,
    recipe_json          TEXT NOT NULL,
    artifact_pcm_sha256  TEXT,
    container_sha256     TEXT,
    sample_rate_hz       INTEGER,
    channels_json        TEXT,
    frames               INTEGER,
    status               TEXT NOT NULL,
    created_at           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS execution (
    execution_id         TEXT PRIMARY KEY,
    node_id              TEXT NOT NULL,
    execution_json       TEXT NOT NULL,
    started_at           TEXT,
    finished_at          TEXT,
    status               TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event (
    event_id              TEXT PRIMARY KEY,
    event_type            TEXT NOT NULL,
    start_sample          INTEGER NOT NULL,
    end_sample            INTEGER NOT NULL,
    features_json         TEXT NOT NULL,
    detector_node_id      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observation (
    node_id               TEXT NOT NULL,
    scope_id              TEXT NOT NULL,
    metric                TEXT NOT NULL,
    value                 REAL,
    uncertainty           REAL,
    available             INTEGER NOT NULL,
    unit                  TEXT,
    details_json          TEXT NOT NULL,
    PRIMARY KEY (node_id, scope_id, metric)
);
CREATE TABLE IF NOT EXISTS challenge_case (
    challenge_id          TEXT PRIMARY KEY,
    challenge_type        TEXT NOT NULL,
    mixture_node_id       TEXT NOT NULL,
    target_node_id        TEXT,
    recipe_json           TEXT NOT NULL,
    class_json            TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS challenge_result (
    challenge_id          TEXT NOT NULL,
    candidate_recipe_id   TEXT NOT NULL,
    result_json           TEXT NOT NULL,
    PRIMARY KEY (challenge_id, candidate_recipe_id)
);
CREATE TABLE IF NOT EXISTS judge_calibration (
    judge_id              TEXT NOT NULL,
    calibration_id        TEXT NOT NULL,
    report_json           TEXT NOT NULL,
    passed_axes_json      TEXT NOT NULL,
    PRIMARY KEY (judge_id, calibration_id)
);
CREATE TABLE IF NOT EXISTS conductor_action (
    action_id             TEXT PRIMARY KEY,
    round                 INTEGER NOT NULL,
    parent_recipe_id      TEXT,
    proposed_json         TEXT NOT NULL,
    validated_recipe_id   TEXT,
    status                TEXT NOT NULL,
    reason                TEXT
);
CREATE TABLE IF NOT EXISTS selection_decision (
    decision_id           TEXT PRIMARY KEY,
    status                TEXT NOT NULL,
    candidate_recipe_id   TEXT,
    selector_version      TEXT NOT NULL,
    report_json           TEXT NOT NULL,
    created_at            TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_state (
    track_id   TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    pi_session TEXT,
    budgets_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT
);
"""


class ImmutableFactError(RuntimeError):
    """An existing completed fact differs from the offered row."""


class ManifestV2:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ManifestV2":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    # --- immutable-fact helper -------------------------------------------
    def _insert_immutable(self, table: str, pk_cols: tuple[str, ...], row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join(f":{c}" for c in row)
        self._conn.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({marks}) ON CONFLICT DO NOTHING", row)
        where = " AND ".join(f"{c}=:{c}" for c in pk_cols)
        stored = self._conn.execute(
            f"SELECT * FROM {table} WHERE {where}", {c: row[c] for c in pk_cols}).fetchone()
        if stored is None:  # pragma: no cover - insert above guarantees presence
            raise ImmutableFactError(f"{table}: row vanished after insert")
        for k, v in row.items():
            if stored[k] != v:
                raise ImmutableFactError(
                    f"{table}: immutable fact conflict on {pk_cols}: column {k!r} "
                    f"stored={stored[k]!r} offered={v!r}")
        self._conn.commit()

    # --- artifact nodes ---------------------------------------------------
    def add_artifact_node(self, *, node_id: str, node_type: str, role: str, operation: str,
                          parents: list[str], recipe: dict, status: str, created_at: str,
                          cohort: str | None = None, artifact_pcm_sha256: str | None = None,
                          container_sha256: str | None = None, sample_rate_hz: int | None = None,
                          channels: list[str] | None = None, frames: int | None = None) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; allowed: {ROLES}")
        self._insert_immutable("artifact_node", ("node_id",), {
            "node_id": node_id, "node_type": node_type, "role": role, "cohort": cohort,
            "operation": operation, "parents_json": json.dumps(parents),
            "recipe_json": json.dumps(recipe, sort_keys=True),
            "artifact_pcm_sha256": artifact_pcm_sha256, "container_sha256": container_sha256,
            "sample_rate_hz": sample_rate_hz,
            "channels_json": json.dumps(channels or []), "frames": frames,
            "status": status, "created_at": created_at,
        })

    def nodes_in_cohort(self, cohort: str, role: str | None = None) -> list[dict]:
        q = "SELECT * FROM artifact_node WHERE cohort=?"
        params: list[Any] = [cohort]
        if role:
            q += " AND role=?"
            params.append(role)
        return [dict(r) for r in self._conn.execute(q + " ORDER BY created_at", params)]

    # --- observations (missing evidence stays 'unavailable') --------------
    def record_observation(self, *, node_id: str, scope_id: str, metric: str,
                           value: float | None, available: bool, uncertainty: float | None = None,
                           unit: str = "", details: dict | None = None) -> None:
        self._insert_immutable("observation", ("node_id", "scope_id", "metric"), {
            "node_id": node_id, "scope_id": scope_id, "metric": metric,
            "value": value, "uncertainty": uncertainty, "available": 1 if available else 0,
            "unit": unit, "details_json": json.dumps(details or {}, sort_keys=True),
        })

    def observations(self, node_id: str, only_available: bool = False) -> list[dict]:
        q = "SELECT * FROM observation WHERE node_id=?"
        if only_available:
            q += " AND available=1"
        return [dict(r) for r in self._conn.execute(q, (node_id,))]

    # --- challenge records ------------------------------------------------
    def add_challenge_case(self, *, challenge_id: str, challenge_type: str, mixture_node_id: str,
                           recipe: dict, klass: dict, target_node_id: str | None = None) -> None:
        self._insert_immutable("challenge_case", ("challenge_id",), {
            "challenge_id": challenge_id, "challenge_type": challenge_type,
            "mixture_node_id": mixture_node_id, "target_node_id": target_node_id,
            "recipe_json": json.dumps(recipe, sort_keys=True),
            "class_json": json.dumps(klass, sort_keys=True),
        })

    def add_challenge_result(self, *, challenge_id: str, candidate_recipe_id: str, result: dict) -> None:
        self._insert_immutable("challenge_result", ("challenge_id", "candidate_recipe_id"), {
            "challenge_id": challenge_id, "candidate_recipe_id": candidate_recipe_id,
            "result_json": json.dumps(result, sort_keys=True),
        })

    def challenge_cases(self, challenge_type: str | None = None) -> list[dict]:
        q = "SELECT * FROM challenge_case"
        params: tuple = ()
        if challenge_type:
            q += " WHERE challenge_type=?"
            params = (challenge_type,)
        return [dict(r) for r in self._conn.execute(q, params)]

    # --- judge calibration ------------------------------------------------
    def add_judge_calibration(self, *, judge_id: str, calibration_id: str,
                              report: dict, passed_axes: list[str]) -> None:
        self._insert_immutable("judge_calibration", ("judge_id", "calibration_id"), {
            "judge_id": judge_id, "calibration_id": calibration_id,
            "report_json": json.dumps(report, sort_keys=True),
            "passed_axes_json": json.dumps(sorted(passed_axes)),
        })

    # --- conductor actions + budgets (restart-safe) -----------------------
    def record_action(self, *, action_id: str, round_no: int, proposed: dict, status: str,
                      parent_recipe_id: str | None = None, validated_recipe_id: str | None = None,
                      reason: str | None = None) -> None:
        self._insert_immutable("conductor_action", ("action_id",), {
            "action_id": action_id, "round": round_no, "parent_recipe_id": parent_recipe_id,
            "proposed_json": json.dumps(proposed, sort_keys=True),
            "validated_recipe_id": validated_recipe_id, "status": status, "reason": reason,
        })

    def action_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM conductor_action GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    # --- selection decisions ----------------------------------------------
    def add_selection_decision(self, *, decision_id: str, status: str, selector_version: str,
                               report: dict, created_at: str,
                               candidate_recipe_id: str | None = None) -> None:
        self._insert_immutable("selection_decision", ("decision_id",), {
            "decision_id": decision_id, "status": status,
            "candidate_recipe_id": candidate_recipe_id, "selector_version": selector_version,
            "report_json": json.dumps(report, sort_keys=True), "created_at": created_at,
        })

    # --- run state (mutable, explicit transitions; session preserved) ------
    def set_run_state(self, track_id: str, state: str, *, pi_session: str | None = None,
                      budgets: dict | None = None, updated_at: str | None = None) -> None:
        row = self._conn.execute("SELECT * FROM run_state WHERE track_id=?", (track_id,)).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO run_state (track_id, state, pi_session, budgets_json, updated_at) "
                "VALUES (?,?,?,?,?)",
                (track_id, state, pi_session, json.dumps(budgets or {}), updated_at))
        else:
            # preserve session/budgets unless explicitly replaced (oracle review)
            new_session = pi_session if pi_session is not None else row["pi_session"]
            new_budgets = json.dumps(budgets) if budgets is not None else row["budgets_json"]
            self._conn.execute(
                "UPDATE run_state SET state=?, pi_session=?, budgets_json=?, updated_at=? "
                "WHERE track_id=?",
                (state, new_session, new_budgets, updated_at, track_id))
        self._conn.commit()

    def run_state(self, track_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM run_state WHERE track_id=?", (track_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["budgets"] = json.loads(d.pop("budgets_json") or "{}")
        return d
