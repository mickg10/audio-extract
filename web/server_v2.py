#!/usr/bin/env python3
"""Web GUI for the audio-extract **v2/v2.1/v3** autonomous library.

Serves a mobile-friendly dashboard + passages/A-B-C view over a v2 ``--lib``
root (the content-addressed tree produced by the ``audio-extract`` CLI — see
``audio_extract/storage.py`` and ``docs/v2``). This is a *sibling* to the v1
``server.py`` (which serves ``convert.py``'s ``lib/`` tree); v1 is untouched.

Pages:

* ``GET /``              pipeline **dashboard**: every run with a state-machine
                         progress strip, screening table, and the local
                         run-launcher panel
* ``GET /r/<track_id>``  the run page (passages timeline, autonomous-decision
                         audit panel, challenge visualizations, A/B/C compare)
* ``GET /runs``          the classic run picker (passages page without a run)
* ``GET /calibration``   Learn-Then-Test calibration report (severity maps,
                         τ table, out-of-sample violations, splits)

Read-only APIs (all defensive about partial/missing data):

* ``GET /api/v2/runs``              every run under --lib (source/state/counts)
* ``GET /api/v2/overview``          dashboard payload: state machine progress,
                                    screening counts, decisions across runs
* ``GET /api/v2/run/<track_id>``    source.json + passages.v1.json + candidates
                                    (recipe.json + manifest metric costs +
                                    Pareto frontier + backend/adapter info)
* ``GET /api/v2/decision/<track_id>``   latest autonomous selection decision
                                        (v2.1 WP13 audit panel) or {"available": false}
* ``GET /api/v2/challenges/<track_id>`` challenge cases + per-candidate results matrix
                                        (exact-reference metrics flattened,
                                        theft-assay rows, backend badges)
* ``GET /api/v2/actions/<track_id>``    conductor probe/action log
* ``GET /api/v2/calibration``       ``calibration/calibration_v1.json`` under the
                                    lib root (or per-run), or {"available": false}
* ``GET /api/v2/panel``             configs/panel.yaml models merged with the
                                    executed model lock (configs/model-lock.json)
* ``GET /api/v2/audio/<t>/source.m4a``                on-the-fly AAC of the source
* ``GET /api/v2/audio/<t>/candidate/<dir>.m4a``       on-the-fly AAC of a candidate
* ``GET /api/v2/peaks/<t>/...json``                   cached waveform peaks (numpy)
* ``GET /api/v2/spectrogram/<t>/...png``              cached spectrogram (matplotlib)

The ONLY writing surface is the **local run launcher** (a job queue, see
``web/jobs_v2.py``):

* ``POST /api/v2/launch``                   enqueue a run (existing run id or an
                                            uploaded audio file) — the worker runs
                                            the standard CLI verbs sequentially
* ``GET  /api/v2/launcher``                 queue snapshot
* ``POST /api/v2/launcher/<id>/cancel``     terminate the stage's process group
* ``POST /api/v2/launcher/<id>/remove``     clear a finished job

The decision/challenges/actions endpoints read the **data model v2** tables
(``selection_decision``, ``challenge_case``, ``challenge_result``,
``judge_calibration``, ``conductor_action``, ``run_state`` — see
``audio_extract/manifest_v2.py``). The pipeline may write those either into the
run's ``manifest.sqlite`` (alongside the v1 tables) or into a separate
``manifest_v2.sqlite``; both are probed, table by table, and a missing table
degrades to ``{"available": false}`` — the GUI is an audit surface, never a
labeling/decision system.

The peaks/spectrogram/audio endpoints accept ``?start=<sample>&end=<sample>`` so
the UI can render a passage-focused window. Raw float32 PCM is **never** sent to
the browser — audio is always transcoded to AAC/m4a with ffmpeg first. Generated
artifacts are cached under ``<lib>/.gui_cache`` (keyed by source path+mtime+args),
so the second request is a plain static file serve with HTTP Range (seeking).

Run:  uv run python web/server_v2.py --lib /path/to/lib   [--port 8731]

Only Python stdlib + numpy + matplotlib + soundfile + ffmpeg are used (all
already project deps; PyYAML — also a project dep — is imported lazily for
panel.yaml and degrades gracefully); no third-party web framework, no CDN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import posixpath
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(WEB_DIR)
STATIC_DIR = os.path.join(WEB_DIR, "static")
DEFAULT_PORT = 8731  # v1 lives on 8730
DEFAULT_PANEL = os.path.join(REPO_DIR, "configs", "panel.yaml")
DEFAULT_LOCK = os.path.join(REPO_DIR, "configs", "model-lock.json")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "1024")) * 1024 * 1024

sys.path.insert(0, WEB_DIR)
import jobs_v2  # noqa: E402  (stdlib-only launcher queue, sibling module)

FFMPEG = shutil.which("ffmpeg")

mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("application/json", ".json")

# Damage axes (mirrors audio_extract.panel_runner.AXES) — lower is better on all.
AXES = ["leakage", "holes_db", "pump_depth_db", "brightness_deviation",
        "hall_damage_db", "stereo_deviation"]
AXIS_META = {
    "leakage":              {"label": "Leakage",        "unit": "",   "hint": "vocal bleed into the stem"},
    "holes_db":             {"label": "Spectral holes", "unit": "dB", "hint": "multiband energy deficit vs consensus"},
    "pump_depth_db":        {"label": "Pumping",        "unit": "dB", "hint": "vocal-synced dynamic dips"},
    "brightness_deviation": {"label": "Brightness dev", "unit": "",   "hint": "timbre drift vs consensus"},
    "hall_damage_db":       {"label": "Hall damage",    "unit": "dB", "hint": "reverb-tail truncation after offsets"},
    "stereo_deviation":     {"label": "Stereo dev",     "unit": "",   "hint": "stereo-width drift vs consensus"},
}

CANDIDATE_DIR_RE = re.compile(r"^sha256_[0-9a-f]{8,}$")
TRACK_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
COPY_CHUNK = 256 * 1024

# Serialize ffmpeg / matplotlib generation per output path so two concurrent
# requests for the same uncached asset don't both compute it.
_gen_locks: dict[str, threading.Lock] = {}
_gen_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _gen_locks_guard:
        lk = _gen_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _gen_locks[key] = lk
        return lk


# ---------------------------------------------------------------------------
# v2 on-disk layout (kept inline so this server is self-contained / decoupled)
# ---------------------------------------------------------------------------
class Lib:
    def __init__(self, root: str, cache_dir: str | None = None):
        self.root = os.path.abspath(root)
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else os.path.join(self.root, ".gui_cache")

    # -- per-track paths --
    def track_root(self, tid: str) -> str:
        return os.path.join(self.root, tid)

    def source_dir(self, tid: str) -> str:
        return os.path.join(self.track_root(tid), "source")

    def passages_json(self, tid: str) -> str:
        return os.path.join(self.track_root(tid), "passages", "passages.v1.json")

    def candidates_dir(self, tid: str) -> str:
        return os.path.join(self.track_root(tid), "candidates")

    def manifest_sqlite(self, tid: str) -> str:
        return os.path.join(self.track_root(tid), "manifest.sqlite")

    def is_run(self, tid: str) -> bool:
        r = self.track_root(tid)
        return os.path.isdir(r) and (
            os.path.isfile(os.path.join(self.source_dir(tid), "source.json"))
            or os.path.isfile(self.manifest_sqlite(tid))
        )

    def list_runs(self) -> list[str]:
        try:
            names = sorted(os.listdir(self.root))
        except OSError:
            return []
        return [n for n in names if not n.startswith(".") and TRACK_ID_RE.match(n) and self.is_run(n)]

    def wav_for(self, tid: str, kind: str, recipe_dir: str | None) -> str | None:
        """Map (kind, recipe_dir) -> the disk path of an f32 wav, or None."""
        if kind == "source":
            p = os.path.join(self.source_dir(tid), "canonical.f32.wav")
        elif kind == "provisional_vocal":
            p = os.path.join(self.source_dir(tid), "provisional_vocal.f32.wav")
        elif kind == "candidate":
            if not recipe_dir or not CANDIDATE_DIR_RE.match(recipe_dir):
                return None
            p = os.path.join(self.candidates_dir(tid), recipe_dir, "output.f32.wav")
        else:
            return None
        return p if os.path.isfile(p) else None


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _metrics(manifest_path: str) -> dict:
    """Return {recipe_id: {axis: value}} aggregate costs from the manifest.

    Reads the ``metric`` table (rows written by ``qa score`` with
    ``passage_id='__aggregate__'`` and ``metric='<axis>/v1'``). Read-only, never
    raises: a missing/locked db yields ``{}``.
    """
    if not os.path.isfile(manifest_path):
        return {}
    costs: dict[str, dict[str, float]] = {}
    try:
        conn = sqlite3.connect(f"file:{manifest_path}?mode=ro", uri=True, timeout=1.0)
    except sqlite3.Error:
        return {}
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT recipe_id, passage_id, metric, value FROM metric "
            "WHERE passage_id='__aggregate__'"
        ).fetchall()
        for r in rows:
            metric = r["metric"] or ""
            axis = metric[:-3] if metric.endswith("/v1") else metric
            if axis in AXIS_META and r["value"] is not None:
                costs.setdefault(r["recipe_id"], {})[axis] = float(r["value"])
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return costs


def _job_state(manifest_path: str) -> str | None:
    if not os.path.isfile(manifest_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{manifest_path}?mode=ro", uri=True, timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT state FROM job LIMIT 1").fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# State machine (docs/v2 §6.5) — the dashboard progress strip
# ---------------------------------------------------------------------------
# Canonical strip stops, in pipeline order. Intermediate/legacy states map onto
# the nearest stop so the strip still lights sensibly.
STATE_STRIP = ["INGESTED", "MINING_PASSAGES", "SCREENING", "MEASURING",
               "FINALIST_SELECTED", "RENDERING_DELIVERY", "FINAL_QC", "COMPLETE"]
STATE_STRIP_SHORT = ["ingest", "mine", "screen", "measure",
                     "finalist", "deliver", "qc", "complete"]
STATE_ALIASES = {
    "PLANNING_REFINEMENT": "MEASURING",
    "REFINING": "MEASURING",
    "AWAITING_HUMAN": "MEASURING",
    "RENDERING_FINALISTS": "FINALIST_SELECTED",
    "SELECTED": "FINALIST_SELECTED",       # run_state variants
    "NO_ACCEPTABLE": "MEASURING",
}
FAILED_STATES = {"FAILED", "CANCELLED"}


def state_progress(state: str | None) -> dict:
    """Map a job/run state onto the strip: {index, failed, label}."""
    if not state:
        return {"index": -1, "failed": False, "label": None}
    if state in FAILED_STATES:
        return {"index": -1, "failed": True, "label": state}
    canonical = STATE_ALIASES.get(state, state)
    try:
        idx = STATE_STRIP.index(canonical)
    except ValueError:
        idx = 0 if state else -1
    return {"index": idx, "failed": False, "label": state}


CONTROL_TAGS = {"no_vocal_control", "random_control"}


def _run_state_row(lib: Lib, tid: str) -> dict | None:
    rows = _v2_rows(lib, tid, "run_state",
                    "SELECT * FROM run_state WHERE track_id=?", (tid,))
    if not rows:
        return None
    r = rows[0]
    return {"state": r.get("state"), "updated_at": r.get("updated_at"),
            "budgets": _loads(r.get("budgets_json"), {})}


def _decision_brief(lib: Lib, tid: str) -> dict | None:
    rows = _v2_rows(lib, tid, "selection_decision",
                    "SELECT rowid AS _rid, decision_id, status, candidate_recipe_id, "
                    "selector_version, created_at, report_json FROM selection_decision")
    if not rows:
        return None
    rows.sort(key=lambda r: (r.get("created_at") or "", r["_rid"]))
    latest = rows[-1]
    report = _loads(latest.get("report_json"), {})
    return {
        "status": latest.get("status"),
        "mode": report.get("mode"),
        "candidate_recipe_id": latest.get("candidate_recipe_id") or report.get("candidate_id"),
        "selector_version": latest.get("selector_version"),
        "created_at": latest.get("created_at"),
        "reason": report.get("reason"),
    }


def _challenge_counts(lib: Lib, tid: str) -> dict:
    cases = _v2_rows(lib, tid, "challenge_case", "SELECT challenge_type FROM challenge_case")
    results = _v2_rows(lib, tid, "challenge_result",
                       "SELECT DISTINCT candidate_recipe_id FROM challenge_result")
    by_type: dict[str, int] = {}
    for c in cases or []:
        t = c.get("challenge_type") or "unknown"
        by_type[t] = by_type.get(t, 0) + 1
    return {"n_cases": len(cases) if cases is not None else 0,
            "by_type": by_type,
            "n_models": len(results) if results is not None else 0}


def _passages_screening(lib: Lib, tid: str) -> dict:
    """Control passages (with vocal-energy ratios) + hard-vocal categories fired."""
    doc = read_json(lib.passages_json(tid)) or {}
    passages = doc.get("passages", []) if isinstance(doc, dict) else []
    ratios: list[float] = []
    hard: set[str] = set()
    n_controls = 0
    for p in passages:
        tags = p.get("tags") or []
        if any(t in CONTROL_TAGS for t in tags):
            n_controls += 1
            r = (p.get("features") or {}).get("vocal_energy_ratio")
            if isinstance(r, (int, float)):
                ratios.append(float(r))
        for t in tags:
            if t not in CONTROL_TAGS:
                hard.add(t)
    return {
        "n_passages": len(passages),
        "n_controls": n_controls,
        "control_ratios": sorted(ratios),
        "min_control_ratio": min(ratios) if ratios else None,
        "hard_tags": sorted(hard),
        "timebase": doc.get("activity_timebase") if isinstance(doc, dict) else None,
    }


def overview_payload(lib: Lib, launcher=None) -> dict:
    """The dashboard: every run with state-machine progress + screening counts +
    decision status. All reads are defensive; a partial run renders partially."""
    runs = []
    for tid in lib.list_runs():
        base = run_summary(lib, tid)
        job_state = base.get("state")
        run_state = _run_state_row(lib, tid)
        # the strip follows the v1 job state; run_state supplies launcher detail
        prog = state_progress(job_state)
        if run_state and run_state.get("state") in FAILED_STATES:
            prog = {"index": prog["index"], "failed": True, "label": run_state["state"]}
        screening = _passages_screening(lib, tid)
        runs.append({
            **base,
            "progress": prog,
            "run_state": run_state,
            "screening": screening,
            "challenges": _challenge_counts(lib, tid),
            "decision": _decision_brief(lib, tid),
        })
    payload = {
        "runs": runs,
        "state_machine": STATE_STRIP,
        "state_machine_short": STATE_STRIP_SHORT,
        "ffmpeg": FFMPEG is not None,
        "server_time": int(time.time()),
    }
    if launcher is not None:
        snap = launcher.snapshot()
        payload["launcher_active"] = snap.get("active", 0)
    return payload


# ---------------------------------------------------------------------------
# Calibration artifact (Learn-Then-Test / SeverityMap — v3 Phases K/L)
# ---------------------------------------------------------------------------
def calibration_payload(lib: Lib) -> dict:
    """``calibration/calibration_v1.json`` under the lib root (preferred) or the
    first run that carries one. Absent/malformed -> {"available": false}."""
    candidates = [os.path.join(lib.root, "calibration", "calibration_v1.json")]
    for tid in lib.list_runs():
        candidates.append(os.path.join(lib.track_root(tid), "calibration",
                                       "calibration_v1.json"))
    for path in candidates:
        doc = read_json(path)
        if isinstance(doc, dict):
            rel = os.path.relpath(path, lib.root)
            return {"available": True, "path": rel, "calibration": doc,
                    "server_time": int(time.time())}
    return {"available": False, "path": None, "calibration": None,
            "hint": "no calibration/calibration_v1.json under the lib root — "
                    "seed one with `uv run python web/dev_seed_calibration.py --lib <root>`",
            "server_time": int(time.time())}


# ---------------------------------------------------------------------------
# Panel + executed model lock (for the run launcher)
# ---------------------------------------------------------------------------
_panel_cache: dict = {"key": None, "payload": None}


def _read_yaml_panel(path: str):
    """panel.yaml via PyYAML (an existing project dep, imported lazily so the
    server keeps running without it)."""
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        return None


def _lock_bundles(lock_path: str) -> dict[str, dict]:
    doc = read_json(lock_path)
    out: dict[str, dict] = {}
    if isinstance(doc, dict):
        for b in doc.get("bundles", []) or []:
            if isinstance(b, dict) and b.get("logical_id"):
                out[b["logical_id"]] = b
    return out


def _bundle_summary(b: dict) -> dict:
    weights = next((f for f in b.get("files", []) if f.get("role") == "weights"), {})
    adapter = b.get("adapter", {}) or {}
    return {
        "logical_id": b.get("logical_id"),
        "alias": (b.get("registry", {}) or {}).get("alias"),
        "family": b.get("family"),
        "target": b.get("target_stem"),
        "bundle_sha256": b.get("bundle_sha256"),
        "weights_sha256": weights.get("sha256"),
        "adapter": adapter.get("name"),
        "adapter_version": adapter.get("version"),
    }


def panel_payload(panel_path: str, lock_path: str) -> dict:
    """configs/panel.yaml models merged with the executed model lock: which seed
    models are import-locked (launchable) vs still needing import. Cached on the
    two files' mtimes."""
    def _mt(p):
        try:
            return os.stat(p).st_mtime_ns
        except OSError:
            return None
    key = (panel_path, _mt(panel_path), lock_path, _mt(lock_path))
    if _panel_cache["key"] == key and _panel_cache["payload"] is not None:
        return _panel_cache["payload"]

    bundles = _lock_bundles(lock_path)
    by_weights = {b["weights_sha256"]: b for b in map(_bundle_summary, bundles.values())
                  if b.get("weights_sha256")}
    by_alias = {b["alias"]: b for b in map(_bundle_summary, bundles.values())
                if b.get("alias")}

    raw = _read_yaml_panel(panel_path)
    models = []
    defaults = {}
    schema = None
    matched_lids: set[str] = set()
    if isinstance(raw, dict):
        schema = raw.get("schema")
        defaults = raw.get("defaults") or {}
        for entry in raw.get("models") or []:
            if not isinstance(entry, dict):
                continue
            ckpt = entry.get("checkpoint") or {}
            sha = ckpt.get("sha256") or ""
            filename = ckpt.get("filename") or ""
            bundle = by_weights.get(sha) or by_alias.get(filename)
            if bundle:
                matched_lids.add(bundle["logical_id"])
            models.append({
                "id": entry.get("id"),
                "family": entry.get("family"),
                "adapter": entry.get("adapter"),
                "target": entry.get("target"),
                "checkpoint": filename,
                "sha256": sha,
                "needs_import": sha in ("", "REQUIRED_AT_IMPORT")
                                or (entry.get("config", {}) or {}).get("sha256") == "REQUIRED_AT_IMPORT",
                "constructions": entry.get("constructions") or [],
                "sweep": entry.get("sweep") or {},
                "locked": bundle is not None,
                "bundle": bundle,
            })
    extra = [_bundle_summary(b) for lid, b in sorted(bundles.items())
             if lid not in matched_lids]
    payload = {
        "panel_path": os.path.relpath(panel_path, REPO_DIR) if panel_path.startswith(REPO_DIR) else panel_path,
        "panel_available": bool(models),
        "schema": schema,
        "defaults": defaults,
        "models": models,
        "lock_path": os.path.relpath(lock_path, REPO_DIR) if lock_path.startswith(REPO_DIR) else lock_path,
        "lock_present": bool(bundles),
        "extra_bundles": extra,   # imported-but-not-in-panel models (still launchable)
        "constructions": ["mixture_minus_primary", "native_primary", "native_secondary"],
        "overlaps": [2, 4, 8],
        "server_time": int(time.time()),
    }
    _panel_cache["key"] = key
    _panel_cache["payload"] = payload
    return payload


def _adapter_backend(adapter: str | None, bundle_id: str | None = None) -> dict | None:
    """Classify the executed adapter into a GUI backend chip.

    ``separator-ttnn`` (Tenstorrent) vs ``audio-separator`` (CUDA/CPU box).
    Unknown adapters still render as chips with kind 'other'."""
    name = (adapter or "").strip()
    if not name and bundle_id:
        name = str(bundle_id).split(":")[0]
    if not name:
        return None
    low = name.lower()
    if "ttnn" in low or low.startswith("tt-"):
        kind = "tt"
    elif "audio-separator" in low or "audio_separator" in low or "demucs" in low:
        kind = "cuda"
    else:
        kind = "other"
    return {"adapter": name, "kind": kind,
            "bundle_prefix": (str(bundle_id)[:16] if bundle_id else None)}


# ---------------------------------------------------------------------------
# Data-model-v2 readers (autonomous decision panel — v2.1 WP13, read-only)
# ---------------------------------------------------------------------------
def _loads(text, default):
    """Parse a stored *_json column; malformed/absent data degrades to default."""
    if not text:
        return default
    try:
        v = json.loads(text)
    except (ValueError, TypeError):
        return default
    return v if isinstance(v, type(default)) else default


def _v2_manifest_paths(lib: Lib, tid: str) -> list[str]:
    """Where the v2 tables may live: the run's manifest.sqlite (v2 tables written
    alongside the v1 ones) or a separate manifest_v2.sqlite. Both are probed."""
    root = lib.track_root(tid)
    return [os.path.join(root, "manifest.sqlite"),
            os.path.join(root, "manifest_v2.sqlite")]


def _v2_rows(lib: Lib, tid: str, table: str, sql: str, params=()) -> list[dict] | None:
    """Rows of ``sql`` from the first manifest file that contains ``table``.

    Returns ``None`` when no manifest has the table (pipeline hasn't run the
    autonomous stage yet), ``[]`` when the table exists but is empty. Read-only
    and defensive: a locked/corrupt db behaves like a missing table.
    """
    for path in _v2_manifest_paths(lib, tid):
        if not os.path.isfile(path):
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        except sqlite3.Error:
            continue
        try:
            conn.row_factory = sqlite3.Row
            has = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not has:
                continue
            return [dict(r) for r in conn.execute(sql, params)]
        except sqlite3.Error:
            continue
        finally:
            conn.close()
    return None


def decision_payload(lib: Lib, tid: str) -> dict:
    """Latest ``selection_decision`` (parsed selector report) + judge calibration."""
    rows = _v2_rows(lib, tid, "selection_decision",
                    "SELECT rowid AS _rid, * FROM selection_decision")
    if not rows:  # None (no table) and [] (no rows) render the same empty panel
        payload = {"track_id": tid, "available": False,
                   "reason": ("no autonomous decision recorded yet"
                              if rows is None else "selection_decision table is empty")}
    else:
        rows.sort(key=lambda r: (r.get("created_at") or "", r["_rid"]))
        latest = rows[-1]
        report = _loads(latest.get("report_json"), {})
        payload = {
            "track_id": tid,
            "available": True,
            "decision_id": latest.get("decision_id"),
            "status": latest.get("status"),
            "mode": report.get("mode"),
            "candidate_recipe_id": latest.get("candidate_recipe_id") or report.get("candidate_id"),
            "selector_version": latest.get("selector_version"),
            "created_at": latest.get("created_at"),
            "report": report,          # full selector report: candidates/params/reason/...
            "n_decisions": len(rows),
        }
    judges = _v2_rows(lib, tid, "judge_calibration",
                      "SELECT * FROM judge_calibration ORDER BY judge_id, calibration_id")
    payload["judges"] = [{"judge_id": j.get("judge_id"),
                          "calibration_id": j.get("calibration_id"),
                          "passed_axes": _loads(j.get("passed_axes_json"), [])}
                         for j in (judges or [])]
    return payload


# Preferred "key metric" per challenge-result dict, in order; the flag says
# whether higher is better (SI-SDR) or lower is better (all error measures).
_PRIMARY_METRIC_PREFS = (
    ("si_sdr_db", True),
    ("target_error_db", False),
    ("stft_distance", False),
    ("band_envelope_err_db", False),
    ("theft_broadband", False),
    ("theft_mean", False),
    ("bleed_broadband", False),
    ("invariance", False),
    ("relative_change", False),
    ("passthrough_error", False),
    ("stereo_width_err", False),
)


def _primary_metric(result: dict) -> dict | None:
    for key, hib in _PRIMARY_METRIC_PREFS:
        v = result.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return {"metric": key, "value": float(v), "higher_is_better": hib}
    for k in sorted(result):
        v = result[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return {"metric": k, "value": float(v), "higher_is_better": False}
    return None


def _flatten_challenge_result(res: dict) -> dict:
    """The real engine nests exact-reference metrics under ``exact_reference``;
    lift them to the top level (without clobbering) so the matrix/heatmap sees
    one shape regardless of writer."""
    flat = dict(res)
    er = res.get("exact_reference")
    if isinstance(er, dict):
        for k, v in er.items():
            flat.setdefault(k, v)
    return flat


def _candidate_backends(lib: Lib, tid: str, cand_ids: set[str], lock_path: str | None) -> dict:
    """candidate id -> backend chip info. Ids may be rendered-candidate recipe
    ids (``sha256:...`` — resolved via the on-disk recipe.json) or model-lock
    logical ids (``challenges run`` stores those)."""
    bundles = _lock_bundles(lock_path) if lock_path else {}
    out: dict[str, dict] = {}
    for cid in cand_ids:
        backend = None
        if str(cid).startswith("sha256:"):
            rj = os.path.join(lib.candidates_dir(tid), str(cid).replace(":", "_"),
                              "recipe.json")
            recipe = read_json(rj) or {}
            model = recipe.get("model", {}) or {}
            backend = _adapter_backend(model.get("adapter"),
                                       model.get("executed_bundle_id"))
        elif cid in bundles:
            b = _bundle_summary(bundles[cid])
            backend = _adapter_backend(b.get("adapter"), b.get("bundle_sha256"))
            if backend:
                backend["family"] = b.get("family")
        if backend:
            out[cid] = backend
    return out


def challenges_payload(lib: Lib, tid: str, lock_path: str | None = None) -> dict:
    """Challenge cases + a {challenge_id: {candidate: result}} matrix with a
    pre-extracted key metric per cell (for the GUI's colored grid), plus the
    theft-assay rows and per-candidate backend badges."""
    cases = _v2_rows(lib, tid, "challenge_case",
                     "SELECT rowid AS _rid, * FROM challenge_case ORDER BY challenge_type, rowid")
    results = _v2_rows(lib, tid, "challenge_result",
                       "SELECT rowid AS _rid, * FROM challenge_result ORDER BY rowid")
    if cases is None and results is None:
        return {"track_id": tid, "available": False, "cases": [],
                "candidates": [], "results": {}, "backends": {}}

    matrix: dict[str, dict] = {}
    cand_ids: set[str] = set()
    for r in results or []:
        res = _loads(r.get("result_json"), {})
        cid = r.get("candidate_recipe_id")
        if not cid:
            continue
        cand_ids.add(cid)
        flat = _flatten_challenge_result(res)
        matrix.setdefault(r.get("challenge_id"), {})[cid] = {
            "metrics": flat, "primary": _primary_metric(flat)}

    out_cases, seen = [], set()
    for c in cases or []:
        seen.add(c.get("challenge_id"))
        out_cases.append({
            "challenge_id": c.get("challenge_id"),
            "challenge_type": c.get("challenge_type"),
            "class": _loads(c.get("class_json"), {}),
            "recipe": _loads(c.get("recipe_json"), {}),
            "mixture_node_id": c.get("mixture_node_id"),
            "target_node_id": c.get("target_node_id"),
        })
    for chid in sorted(matrix):          # results whose case row is missing
        if chid in seen:
            continue
        ctype = "theft_assay" if chid == "theft_assay" else "unknown"
        out_cases.append({"challenge_id": chid, "challenge_type": ctype,
                          "class": {}, "recipe": {},
                          "mixture_node_id": None, "target_node_id": None})
    return {"track_id": tid,
            "available": bool(out_cases or matrix),
            "cases": out_cases,
            "candidates": sorted(cand_ids),
            "results": matrix,
            "backends": _candidate_backends(lib, tid, cand_ids, lock_path)}


def actions_payload(lib: Lib, tid: str) -> dict:
    """Conductor probe/action log: round, proposed action, status, reason."""
    rows = _v2_rows(lib, tid, "conductor_action",
                    "SELECT rowid AS _rid, * FROM conductor_action ORDER BY round, rowid")
    if rows is None:
        return {"track_id": tid, "available": False, "actions": [], "counts": {}}
    actions, counts = [], {}
    for r in rows:
        proposed = _loads(r.get("proposed_json"), {})
        status = r.get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1
        actions.append({
            "action_id": r.get("action_id"),
            "round": r.get("round"),
            "type": proposed.get("type"),
            "proposed": proposed,
            "parent_recipe_id": r.get("parent_recipe_id"),
            "validated_recipe_id": r.get("validated_recipe_id"),
            "status": status,
            "reason": r.get("reason"),
        })
    return {"track_id": tid, "available": True, "actions": actions, "counts": counts}


# ---------------------------------------------------------------------------
# Pareto frontier + scalarized rank (pure python; lower is better everywhere)
# ---------------------------------------------------------------------------
def _dominates(a: dict, b: dict, axes, eps: float = 1e-9) -> bool:
    return all(a[x] <= b[x] + eps for x in axes) and any(a[x] < b[x] - eps for x in axes)


def _pareto_and_rank(costs: dict[str, dict], axes) -> tuple[set, dict]:
    """(pareto set of recipe_ids, {recipe_id: rank}) over candidates with a full
    cost vector. Rank is a z-scored equal-weight sum (lower total = better)."""
    ids = [r for r, c in costs.items() if all(a in c for a in axes)]
    if not ids:
        return set(), {}
    front = {r for r in ids if not any(_dominates(costs[o], costs[r], axes) for o in ids if o != r)}
    # z-score each axis across the set, then sum (equal weights).
    means = {a: sum(costs[r][a] for r in ids) / len(ids) for a in axes}
    sds = {}
    for a in axes:
        var = sum((costs[r][a] - means[a]) ** 2 for r in ids) / len(ids)
        sds[a] = (var ** 0.5) or 1.0
    totals = {r: sum((costs[r][a] - means[a]) / sds[a] for a in axes) for r in ids}
    order = sorted(ids, key=lambda r: totals[r])
    rank = {r: i for i, r in enumerate(order)}
    return front, rank


# ---------------------------------------------------------------------------
# Run payloads
# ---------------------------------------------------------------------------
def run_summary(lib: Lib, tid: str) -> dict:
    src = read_json(os.path.join(lib.source_dir(tid), "source.json")) or {}
    frames = src.get("frames")
    sr = src.get("sample_rate_hz")
    duration_s = (frames / sr) if (frames and sr) else None
    passages = read_json(lib.passages_json(tid)) or {}
    n_passages = len(passages.get("passages", [])) if isinstance(passages, dict) else 0
    n_cand = 0
    cdir = lib.candidates_dir(tid)
    if os.path.isdir(cdir):
        for d in os.listdir(cdir):
            if CANDIDATE_DIR_RE.match(d) and os.path.isfile(os.path.join(cdir, d, "output.f32.wav")):
                n_cand += 1
    return {
        "track_id": tid,
        "title": src.get("original_filename") or tid,
        "state": _job_state(lib.manifest_sqlite(tid)),
        "duration_s": duration_s,
        "sample_rate_hz": sr,
        "channels": src.get("channel_layout"),
        "frames": frames,
        "n_passages": n_passages,
        "n_candidates": n_cand,
        "url": "/r/" + urllib.parse.quote(tid),
    }


def _candidate_label(recipe: dict) -> str:
    op = (recipe or {}).get("operation", {}) or {}
    model = (recipe or {}).get("model", {}) or {}
    model_id = model.get("model_id") or "model"
    target = op.get("target") or ""
    construction = op.get("construction") or op.get("type") or ""
    bits = [model_id]
    if target:
        bits.append(target)
    if construction:
        bits.append(construction)
    return " · ".join(bits)


def build_run_payload(lib: Lib, tid: str) -> dict | None:
    if not lib.is_run(tid):
        return None
    src = read_json(os.path.join(lib.source_dir(tid), "source.json")) or {}
    frames = src.get("frames")
    sr = src.get("sample_rate_hz") or 44100
    duration_s = (frames / sr) if (frames and sr) else None

    # passages -> add seconds + primary tag
    passages_doc = read_json(lib.passages_json(tid)) or {}
    passages = []
    for p in (passages_doc.get("passages", []) if isinstance(passages_doc, dict) else []):
        s0, s1 = int(p.get("start_sample", 0)), int(p.get("end_sample", 0))
        tags = p.get("tags") or []
        passages.append({
            "passage_id": p.get("passage_id"),
            "start_sample": s0,
            "end_sample": s1,
            "start_s": s0 / sr if sr else 0.0,
            "end_s": s1 / sr if sr else 0.0,
            "tags": tags,
            "primary_tag": tags[0] if tags else "passage",
            "features": p.get("features", {}),
            "reason": p.get("reason", ""),
        })

    # candidates from disk (recipe.json + immutable output present)
    costs_by_recipe = _metrics(lib.manifest_sqlite(tid))
    pareto, rank = _pareto_and_rank(costs_by_recipe, AXES)

    candidates = []
    cdir = lib.candidates_dir(tid)
    if os.path.isdir(cdir):
        for d in sorted(os.listdir(cdir)):
            if not CANDIDATE_DIR_RE.match(d):
                continue
            out_wav = os.path.join(cdir, d, "output.f32.wav")
            if not os.path.isfile(out_wav):
                continue
            recipe = read_json(os.path.join(cdir, d, "recipe.json")) or {}
            recipe_id = "sha256:" + d[len("sha256_"):]
            costs = costs_by_recipe.get(recipe_id, {})
            has_costs = all(a in costs for a in AXES)
            model_block = recipe.get("model", {}) or {}
            candidates.append({
                "dir": d,
                "recipe_id": recipe_id,
                "short_id": d[len("sha256_"):][:8],
                "label": _candidate_label(recipe),
                "operation": (recipe.get("operation", {}) or {}),
                "model": model_block,
                "backend": _adapter_backend(model_block.get("adapter"),
                                            model_block.get("executed_bundle_id")),
                "effective_config": (recipe.get("effective_config", {}) or {}),
                "costs": costs,
                "has_costs": has_costs,
                "pareto": recipe_id in pareto,
                "rank": rank.get(recipe_id),
                "audio_url": "/api/v2/audio/%s/candidate/%s.m4a" % (urllib.parse.quote(tid), d),
                "peaks_url": "/api/v2/peaks/%s/candidate/%s.json" % (urllib.parse.quote(tid), d),
                "spectrogram_url": "/api/v2/spectrogram/%s/candidate/%s.png" % (urllib.parse.quote(tid), d),
            })

    # order: pareto first, then by scalarized rank, then dir
    candidates.sort(key=lambda c: (0 if c["pareto"] else 1,
                                   c["rank"] if c["rank"] is not None else 1e9, c["dir"]))

    q = urllib.parse.quote(tid)
    have_pv = os.path.isfile(os.path.join(lib.source_dir(tid), "provisional_vocal.f32.wav"))
    return {
        "track_id": tid,
        "title": src.get("original_filename") or tid,
        "source": src,
        "duration_s": duration_s,
        "sample_rate_hz": sr,
        "channels": src.get("channel_layout"),
        "frames": frames,
        "state": _job_state(lib.manifest_sqlite(tid)),
        "axes": AXES,
        "axis_meta": AXIS_META,
        "passages": passages,
        "candidates": candidates,
        "pareto_frontier": sorted(pareto),
        "source_audio_url": "/api/v2/audio/%s/source.m4a" % q,
        "source_peaks_url": "/api/v2/peaks/%s/source.json" % q,
        "source_spectrogram_url": "/api/v2/spectrogram/%s/source.png" % q,
        "provisional_vocal_audio_url": ("/api/v2/audio/%s/provisional_vocal.m4a" % q) if have_pv else None,
        "server_time": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Generators: peaks (numpy) + spectrogram (matplotlib) + audio (ffmpeg)
# ---------------------------------------------------------------------------
def _cache_key(wav_path: str, *parts) -> str:
    try:
        st = os.stat(wav_path)
        sig = "%s|%d|%d|%s" % (wav_path, st.st_size, int(st.st_mtime), "|".join(str(p) for p in parts))
    except OSError:
        sig = "%s|%s" % (wav_path, "|".join(str(p) for p in parts))
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def _cache_file(lib: Lib, namespace: str, key: str, ext: str) -> str:
    d = os.path.join(lib.cache_dir, namespace)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%s.%s" % (key, ext))


def _read_window(wav_path: str, start: int | None, end: int | None):
    """Read a wav (optionally a [start,end) sample window) as (mono float32, sr,
    total_frames). Bounded to the requested window to cap memory."""
    import numpy as np
    import soundfile as sf

    info = sf.info(wav_path)
    total = int(info.frames)
    s0 = 0 if start is None else max(0, min(int(start), total))
    s1 = total if end is None else max(s0, min(int(end), total))
    data, sr = sf.read(wav_path, start=s0, stop=s1, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono), int(sr), total


def generate_peaks(lib: Lib, wav_path: str, start, end, buckets: int = 2000) -> str:
    key = _cache_key(wav_path, "peaks", start, end, buckets)
    out = _cache_file(lib, "peaks", key, "json")
    if os.path.isfile(out):
        return out
    with _lock_for(out):
        if os.path.isfile(out):
            return out
        import numpy as np

        mono, sr, total = _read_window(wav_path, start, end)
        n = len(mono)
        if n == 0:
            payload = {"min": [], "max": [], "n": 0}
        else:
            nb = int(max(1, min(buckets, n)))
            edges = np.linspace(0, n, nb + 1).astype(np.int64)
            starts = edges[:-1]
            mins = np.minimum.reduceat(mono, starts)
            maxs = np.maximum.reduceat(mono, starts)
            payload = {
                "min": [round(float(x), 4) for x in mins],
                "max": [round(float(x), 4) for x in maxs],
                "n": nb,
                "sr": sr,
                "total_frames": total,
            }
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, out)
    return out


def generate_spectrogram(lib: Lib, wav_path: str, start, end) -> str:
    key = _cache_key(wav_path, "spec", start, end)
    out = _cache_file(lib, "spectrogram", key, "png")
    if os.path.isfile(out):
        return out
    with _lock_for(out):
        if os.path.isfile(out):
            return out
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        mono, sr, _ = _read_window(wav_path, start, end)
        fig = plt.figure(figsize=(9, 2.4), dpi=110)
        ax = fig.add_subplot(111)
        bg = "#0a0d12"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        if len(mono) >= 64:
            nfft = 1024
            while nfft > len(mono) and nfft > 64:
                nfft //= 2
            noverlap = int(nfft * 0.75)
            ax.specgram(mono, NFFT=nfft, Fs=sr, noverlap=noverlap, cmap="magma",
                        mode="magnitude", scale="dB", vmin=-100, vmax=-10)
            ax.set_ylim(0, min(sr / 2, 16000))
            t0 = (start or 0) / sr
            # relabel the x axis to absolute track time
            ticks = ax.get_xticks()
            ax.set_xticklabels(["%.1f" % (t + t0) for t in ticks], fontsize=7, color="#9aa7b4")
            ax.set_yticks([0, 4000, 8000, 12000, 16000])
            ax.set_yticklabels(["0", "4k", "8k", "12k", "16k"], fontsize=7, color="#9aa7b4")
            for spine in ax.spines.values():
                spine.set_color("#2a3140")
            ax.tick_params(colors="#2a3140", length=2)
            ax.set_xlabel("time (s)", fontsize=7, color="#6b7684")
            ax.set_ylabel("Hz", fontsize=7, color="#6b7684")
        else:
            ax.text(0.5, 0.5, "window too short", ha="center", va="center",
                    color="#6b7684", transform=ax.transAxes)
            ax.axis("off")
        fig.tight_layout(pad=0.4)
        tmp = out + ".tmp.png"
        fig.savefig(tmp, facecolor=bg)
        plt.close(fig)
        os.replace(tmp, out)
    return out


def generate_aac(lib: Lib, wav_path: str, start, end, bitrate: str = "192k") -> str | None:
    if FFMPEG is None:
        return None
    key = _cache_key(wav_path, "aac", start, end, bitrate)
    out = _cache_file(lib, "audio", key, "m4a")
    if os.path.isfile(out):
        return out
    with _lock_for(out):
        if os.path.isfile(out):
            return out
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
        # Optional sample-accurate trim to a passage window (using an f32 wav so
        # sample offsets convert cleanly to seconds at the file's sample rate).
        if start is not None or end is not None:
            import soundfile as sf
            sr = int(sf.info(wav_path).samplerate)
            ss = (start or 0) / sr
            cmd += ["-ss", "%.6f" % ss, "-i", wav_path]
            if end is not None:
                cmd += ["-t", "%.6f" % (((end - (start or 0)) / sr))]
        else:
            cmd += ["-i", wav_path]
        tmp = out + ".tmp.m4a"
        cmd += ["-ac", "2", "-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart", tmp]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, OSError):
            try:
                os.remove(tmp)
            except OSError:
                pass
            return None
        os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------
# multipart/form-data (launcher upload; same minimal parser as v1 server.py)
# ---------------------------------------------------------------------------
def parse_multipart_formdata(body: bytes, boundary: bytes):
    """Minimal in-memory multipart parser -> [{name, filename, data}]. Uploads
    are bounded by MAX_UPLOAD_BYTES; boundary is browser-chosen per RFC 7578."""
    delim = b"--" + boundary
    parts = []
    for seg in body.split(delim):
        if not seg or seg in (b"--", b"--\r\n", b"\r\n"):
            continue
        if seg.startswith(b"--"):
            continue
        if seg.startswith(b"\r\n"):
            seg = seg[2:]
        if seg.endswith(b"\r\n"):
            seg = seg[:-2]
        head_end = seg.find(b"\r\n\r\n")
        if head_end < 0:
            continue
        raw_headers = seg[:head_end].decode("utf-8", "replace")
        data = seg[head_end + 4:]
        name = filename = None
        for hline in raw_headers.split("\r\n"):
            if hline.lower().startswith("content-disposition:"):
                for m in re.finditer(r'(\w+)="([^"]*)"', hline):
                    if m.group(1) == "name":
                        name = m.group(2)
                    elif m.group(1) == "filename":
                        filename = m.group(2)
        parts.append({"name": name, "filename": filename, "data": data})
    return parts


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "").strip() or "upload"
    return _SAFE_NAME_RE.sub("_", base)[:120]


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
API_RUN_RE = re.compile(r"^/api/v2/run/(.+)$")
DECISION_RE = re.compile(r"^/api/v2/decision/([^/]+)$")
CHALLENGES_RE = re.compile(r"^/api/v2/challenges/([^/]+)$")
ACTIONS_RE = re.compile(r"^/api/v2/actions/([^/]+)$")
AUDIO_RE = re.compile(r"^/api/v2/audio/([^/]+)/(source|provisional_vocal|candidate/[^/]+)\.m4a$")
PEAKS_RE = re.compile(r"^/api/v2/peaks/([^/]+)/(source|candidate/[^/]+)\.json$")
SPEC_RE = re.compile(r"^/api/v2/spectrogram/([^/]+)/(source|candidate/[^/]+)\.png$")
RUN_PAGE_RE = re.compile(r"^/r/([^/]+)/?$")
LAUNCHER_ACTION_RE = re.compile(r"^/api/v2/launcher/([A-Za-z0-9]+)/(cancel|remove)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "audio-extract-web-v2/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers -----------------------------------------------------------
    def _send_headers(self, status, ctype, length=None, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8", len(body),
                           {"Cache-Control": "no-cache"})
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_text(self, text, status=200, ctype="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self._send_headers(status, ctype, len(body), {"Cache-Control": "no-cache"})
        if self.command != "HEAD":
            self.wfile.write(body)

    def _serve_frontend(self, name, ctype):
        path = os.path.join(STATIC_DIR, name)
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            self._send_text("Not found", 404)
            return
        self._send_headers(200, ctype, len(body), {"Cache-Control": "no-cache"})
        if self.command != "HEAD":
            self.wfile.write(body)

    def _serve_file(self, disk_path, cache="no-cache", ctype=None):
        """Serve a file with HTTP Range support (206) so <audio> seeking works."""
        if not disk_path or not os.path.isfile(disk_path):
            self._send_text("Not found", 404)
            return
        try:
            size = os.path.getsize(disk_path)
        except OSError:
            self._send_text("Not found", 404)
            return
        if ctype is None:
            ctype, _ = mimetypes.guess_type(disk_path)
            ctype = ctype or "application/octet-stream"
        extra = {"Accept-Ranges": "bytes", "Cache-Control": cache}
        start, end, partial = 0, size - 1, False
        rng = self.headers.get("Range")
        if rng:
            m = RANGE_RE.match(rng.strip())
            if m:
                g1, g2 = m.group(1), m.group(2)
                if g1 == "" and g2 == "":
                    pass
                elif g1 == "":
                    n = int(g2)
                    if n > 0:
                        start = max(0, size - n)
                        partial = True
                else:
                    start = int(g1)
                    if g2 != "":
                        end = min(int(g2), size - 1)
                    partial = True
                    if start > end or start >= size:
                        self._send_headers(416, ctype, 0,
                                           {"Content-Range": "bytes */%d" % size,
                                            "Accept-Ranges": "bytes"})
                        return
        length = end - start + 1
        if partial:
            extra["Content-Range"] = "bytes %d-%d/%d" % (start, end, size)
            self._send_headers(206, ctype, length, extra)
        else:
            self._send_headers(200, ctype, length, extra)
        if self.command == "HEAD":
            return
        try:
            with open(disk_path, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _win(self, query):
        """Parse ?start=&end= sample bounds from a query dict, or (None, None)."""
        def _int(name):
            v = query.get(name, [None])[0]
            if v is None or v == "":
                return None
            try:
                return max(0, int(float(v)))
            except (TypeError, ValueError):
                return None
        return _int("start"), _int("end")

    def _resolve_kind(self, second: str):
        """'source' | 'provisional_vocal' | 'candidate/<dir>' -> (kind, recipe_dir)."""
        if second.startswith("candidate/"):
            return "candidate", second[len("candidate/"):]
        return second, None

    # -- routing -----------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        lib = self.server.lib
        try:
            if path in ("/", "/index.html", "/dashboard"):
                self._serve_frontend("dashboard.html", "text/html; charset=utf-8")
                return
            if path in ("/runs", "/passages"):   # the classic run picker
                self._serve_frontend("passages.html", "text/html; charset=utf-8")
                return
            if path == "/calibration":
                self._serve_frontend("calibration.html", "text/html; charset=utf-8")
                return
            if RUN_PAGE_RE.match(path):
                self._serve_frontend("passages.html", "text/html; charset=utf-8")
                return
            if path == "/healthz":
                self._send_text("ok")
                return
            if path == "/favicon.ico":
                self._send_headers(204, "image/x-icon", 0)
                return

            if path == "/api/v2/runs":
                self._send_json({"runs": [run_summary(lib, t) for t in lib.list_runs()],
                                 "ffmpeg": FFMPEG is not None, "server_time": int(time.time())})
                return

            if path == "/api/v2/overview":
                self._send_json(overview_payload(lib, getattr(self.server, "launcher", None)))
                return

            if path == "/api/v2/calibration":
                self._send_json(calibration_payload(lib))
                return

            if path == "/api/v2/panel":
                self._send_json(panel_payload(
                    getattr(self.server, "panel_path", DEFAULT_PANEL),
                    getattr(self.server, "lock_path", DEFAULT_LOCK)))
                return

            if path == "/api/v2/launcher":
                launcher = getattr(self.server, "launcher", None)
                if launcher is None:
                    self._send_json({"jobs": [], "active": 0, "enabled": False})
                else:
                    self._send_json({**launcher.snapshot(), "enabled": True})
                return

            m = API_RUN_RE.match(path)
            if m:
                tid = urllib.parse.unquote(m.group(1)).strip("/")
                payload = build_run_payload(lib, tid)
                if payload is None:
                    self._send_json({"error": "run not found", "track_id": tid}, 404)
                else:
                    self._send_json(payload)
                return

            # ---- autonomous decision panel (WP13, read-only audit) ----
            _lock = getattr(self.server, "lock_path", DEFAULT_LOCK)
            _challenges = lambda l, t: challenges_payload(l, t, _lock)  # noqa: E731
            for rx, fn in ((DECISION_RE, decision_payload),
                           (CHALLENGES_RE, _challenges),
                           (ACTIONS_RE, actions_payload)):
                m = rx.match(path)
                if m:
                    tid = urllib.parse.unquote(m.group(1))
                    if not lib.is_run(tid):
                        self._send_json({"error": "run not found", "track_id": tid}, 404)
                    else:
                        self._send_json(fn(lib, tid))
                    return

            m = AUDIO_RE.match(path)
            if m:
                tid = urllib.parse.unquote(m.group(1))
                kind, rdir = self._resolve_kind(urllib.parse.unquote(m.group(2)))
                wav = lib.wav_for(tid, kind, rdir)
                if wav is None:
                    self._send_text("Not found", 404)
                    return
                if FFMPEG is None:
                    self._send_text("ffmpeg not available on server", 503)
                    return
                start, end = self._win(query)
                out = generate_aac(lib, wav, start, end)
                if out is None:
                    self._send_text("transcode failed", 500)
                    return
                self._serve_file(out, cache="public, max-age=3600", ctype="audio/mp4")
                return

            m = PEAKS_RE.match(path)
            if m:
                tid = urllib.parse.unquote(m.group(1))
                kind, rdir = self._resolve_kind(urllib.parse.unquote(m.group(2)))
                wav = lib.wav_for(tid, kind, rdir)
                if wav is None:
                    self._send_json({"min": [], "max": [], "n": 0}, 200)
                    return
                start, end = self._win(query)
                try:
                    out = generate_peaks(lib, wav, start, end)
                except Exception as exc:
                    self._send_json({"error": str(exc), "min": [], "max": []}, 200)
                    return
                self._serve_file(out, cache="public, max-age=3600", ctype="application/json")
                return

            m = SPEC_RE.match(path)
            if m:
                tid = urllib.parse.unquote(m.group(1))
                kind, rdir = self._resolve_kind(urllib.parse.unquote(m.group(2)))
                wav = lib.wav_for(tid, kind, rdir)
                if wav is None:
                    self._send_text("Not found", 404)
                    return
                start, end = self._win(query)
                try:
                    out = generate_spectrogram(lib, wav, start, end)
                except Exception as exc:
                    self._send_text("spectrogram error: %s" % exc, 500)
                    return
                self._serve_file(out, cache="public, max-age=3600", ctype="image/png")
                return

            if path.startswith("/static/"):
                name = posixpath.normpath(path[len("/static/"):]).lstrip("/")
                if name.startswith("..") or os.path.isabs(name):
                    self._send_text("Forbidden", 403)
                    return
                ctype, _ = mimetypes.guess_type(name)
                self._serve_frontend(name, ctype or "application/octet-stream")
                return

            self._send_text("Not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # never let one bad request kill the thread
            try:
                self._send_text("Server error: %s" % exc, 500)
            except Exception:
                pass

    # -- POST: the local run launcher (the ONLY writing surface) -----------
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        launcher = getattr(self.server, "launcher", None)
        try:
            if launcher is None:
                self._send_json({"ok": False, "error": "launcher disabled"}, 503)
                return
            m = LAUNCHER_ACTION_RE.match(path)
            if m:
                jid, action = m.group(1), m.group(2)
                ok, msg = (launcher.cancel(jid) if action == "cancel"
                           else launcher.remove(jid))
                self._send_json({"ok": ok, "message": msg}, 200 if ok else 400)
                return
            if path == "/api/v2/launch":
                self._handle_launch(launcher)
                return
            self._send_text("Not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self._send_json({"ok": False, "error": str(exc)}, 500)
            except Exception:
                pass

    def _handle_launch(self, launcher):
        """POST /api/v2/launch — JSON {run_id, models[], construction, overlap}
        for an existing run, or multipart/form-data with a ``file`` part for a
        fresh ingest. Appends to the local queue (concurrency 1)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send_json({"ok": False, "error": "missing or oversized body "
                             "(max %d MB)" % (MAX_UPLOAD_BYTES // (1024 * 1024))}, 413 if length else 400)
            return
        ctype = self.headers.get("Content-Type") or ""
        body = self.rfile.read(length)

        fields: dict[str, str] = {}
        upload = None
        if ctype.lower().startswith("multipart/form-data"):
            m = re.search(r'boundary="?([^";]+)"?', ctype)
            if not m:
                self._send_json({"ok": False, "error": "missing multipart boundary"}, 400)
                return
            for part in parse_multipart_formdata(body, m.group(1).encode()):
                if part["filename"]:
                    if part["data"]:
                        upload = part
                else:
                    fields[part["name"] or ""] = part["data"].decode("utf-8", "replace").strip()
        else:
            try:
                fields = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._send_json({"ok": False, "error": "body must be JSON or multipart"}, 400)
                return
            if not isinstance(fields, dict):
                self._send_json({"ok": False, "error": "JSON body must be an object"}, 400)
                return

        models = fields.get("models") or []
        if isinstance(models, str):
            models = [x.strip() for x in models.split(",") if x.strip()]
        construction = fields.get("construction") or "mixture_minus_primary"
        if construction not in ("mixture_minus_primary", "native_primary", "native_secondary"):
            self._send_json({"ok": False, "error": "unknown construction %r" % construction}, 400)
            return
        try:
            overlap = int(fields.get("overlap") or 8)
        except (TypeError, ValueError):
            overlap = 8

        source_path = source_label = None
        run_id = (fields.get("run_id") or "").strip()
        if upload is not None:
            fname = _sanitize_filename(upload["filename"])
            if not re.search(r"\.(mp3|wav|flac|m4a|ogg)$", fname, re.I):
                self._send_json({"ok": False, "error": "unsupported upload type "
                                 "(mp3/wav/flac/m4a/ogg)"}, 400)
                return
            if not run_id:
                run_id = re.sub(r"\.[A-Za-z0-9]+$", "", fname)
            run_id = _SAFE_NAME_RE.sub("_", run_id)[:80]
            dest = os.path.join(launcher.uploads_dir, "%d_%s" % (int(time.time()), fname))
            with open(dest, "wb") as fh:
                fh.write(upload["data"])
            source_path = dest
            source_label = upload["filename"]
        if not TRACK_ID_RE.match(run_id or ""):
            self._send_json({"ok": False, "error": "invalid run_id"}, 400)
            return

        jid, msg = launcher.enqueue(run_id=run_id, models=models,
                                    construction=construction, overlap=overlap,
                                    source_path=source_path, source_label=source_label)
        if jid is None:
            self._send_json({"ok": False, "error": msg}, 400)
        else:
            self._send_json({"ok": True, "job_id": jid, "run_id": run_id,
                             "message": msg})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main():
    ap = argparse.ArgumentParser(description="audio-extract v2 web GUI (dashboard + audit + local launcher)")
    ap.add_argument("--lib", required=True, help="v2 library root (contains <track_id>/ dirs)")
    ap.add_argument("--cache-dir", default=None,
                    help="where to cache generated peaks/spectrograms/m4a (default: <lib>/.gui_cache)")
    ap.add_argument("--panel", default=DEFAULT_PANEL, help="panel.yaml (default: configs/panel.yaml)")
    ap.add_argument("--lock", default=DEFAULT_LOCK,
                    help="executed model lock (default: configs/model-lock.json)")
    ap.add_argument("--model-dir", default=None,
                    help="separator model dir passed to launched CLI stages (default: CLI default)")
    ap.add_argument("--no-launcher", action="store_true",
                    help="disable the local run-launcher queue (pure read-only server)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", DEFAULT_PORT)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    args = ap.parse_args()

    lib = Lib(args.lib, args.cache_dir)
    if not os.path.isdir(lib.root):
        sys.stderr.write("WARNING: --lib %s does not exist yet (runs will appear later)\n" % lib.root)
    os.makedirs(lib.cache_dir, exist_ok=True)
    if FFMPEG is None:
        sys.stderr.write("WARNING: ffmpeg not found on PATH — audio endpoints will 503\n")

    httpd = Server((args.host, args.port), Handler)
    httpd.lib = lib
    httpd.panel_path = os.path.abspath(args.panel)
    httpd.lock_path = os.path.abspath(args.lock)
    httpd.launcher = None
    if not args.no_launcher:
        httpd.launcher = jobs_v2.LaunchManager(
            REPO_DIR, lib.root, os.path.join(lib.cache_dir, "launcher"),
            lock_path=httpd.lock_path, model_dir=args.model_dir)
    sys.stderr.write(
        "audio-extract v2 GUI serving lib=%s on http://%s:%d\n"
        "  cache -> %s | ffmpeg -> %s | launcher -> %s\n"
        % (lib.root, args.host, args.port, lib.cache_dir, FFMPEG or "MISSING",
           "LOCAL queue (concurrency 1)" if httpd.launcher else "disabled")
    )

    def _term(_signum, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _term)

    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        sys.stderr.write("\nshutting down\n")
    finally:
        if httpd.launcher is not None:
            httpd.launcher.shutdown()   # never orphan a launched stage
        httpd.server_close()


if __name__ == "__main__":
    main()
