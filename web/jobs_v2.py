#!/usr/bin/env python3
"""Local run-launcher queue for the **v2** GUI (`server_v2.py`).

A single background worker thread executes at most ONE launch job at a time
(concurrency 1 — the separator saturates the machine). Each job runs the
standard autonomous CLI verbs **sequentially** as subprocesses of the project
venv (``uv run audio-extract --lib <lib> ...``):

    ingest -> passages mine -> panel render -> qa score
           -> challenges build -> challenges run -> select autonomous

This is the SAME code path the GPU box uses — the launcher only sequences the
CLI. It is a *local* launcher: the selected models must already be imported into
``configs/model-lock.json`` and present in the model dir; on a laptop a run may
simply be slow.

Design (mirrors the proven v1 ``web/jobs.py``):

* every stage subprocess starts in its OWN session/process group
  (``start_new_session=True``), so **cancel** kills the whole tree
  (``os.killpg`` SIGTERM, SIGKILL escalation) — model workers included;
* jobs persist to ``<state_dir>/launcher_jobs.json`` and survive a server
  restart (a job that was running is marked interrupted on reload);
* stage status is streamed into the run's ``run_state`` table (the data-model-v2
  table, written with the exact ``manifest_v2`` schema via plain sqlite3) so the
  dashboard's state strip animates while the CLI verbs run;
* stdlib only — the CLI subprocesses carry all the heavy deps.

Testing seam: ``AUDIO_EXTRACT_V2_STAGE_CMD`` overrides the per-stage command (a
shell-ish template; ``{stage}``, ``{run_id}``, ``{lib}`` are substituted), so the
queue/cancel/persistence machinery is testable without models.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sqlite3
import subprocess
import threading
import time
import uuid

TERMINAL = {"done", "failed", "cancelled"}
LOG_TAIL_MAX = 60

# The standard autonomous pipeline, in execution order. "ingest" is skipped when
# the job targets an already-ingested run.
STAGES = (
    "ingest",
    "passages_mine",
    "panel_render",
    "qa_score",
    "challenges_build",
    "challenges_run",
    "select_autonomous",
)

# Matches audio_extract/manifest_v2.py _SCHEMA (run_state only) — kept inline so
# the web server never has to import the pipeline package.
_RUN_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_state (
    track_id   TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    pi_session TEXT,
    budgets_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT
);
"""


def _now() -> float:
    return time.time()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _kill_group(proc) -> None:
    """SIGTERM the subprocess's whole group now; SIGKILL after a short grace."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return

    def _escalate():
        time.sleep(3)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    threading.Thread(target=_escalate, daemon=True).start()


class LaunchManager:
    """Queue + worker for local autonomous runs. All public methods are
    thread-safe; the HTTP handler threads call enqueue/snapshot/cancel."""

    def __init__(self, repo_root: str, lib_root: str, state_dir: str,
                 lock_path: str | None = None, model_dir: str | None = None):
        self.repo_root = os.path.abspath(repo_root)
        self.lib_root = os.path.abspath(lib_root)
        self.state_dir = os.path.abspath(state_dir)
        self.lock_path = lock_path or os.path.join(self.repo_root, "configs", "model-lock.json")
        self.model_dir = model_dir  # None -> CLI default (~/audio-extract/models)
        self.jobs_file = os.path.join(self.state_dir, "launcher_jobs.json")
        self.uploads_dir = os.path.join(self.state_dir, "uploads")
        os.makedirs(self.uploads_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.jobs: list[dict] = []
        self._proc: dict[str, subprocess.Popen] = {}
        self._cancel: set[str] = set()
        self._shutting_down = False
        self._load()
        self._worker = threading.Thread(target=self._run_loop, name="v2-launch-worker",
                                        daemon=True)
        self._worker.start()

    # ---- persistence ------------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.jobs_file, "r", encoding="utf-8") as fh:
                jobs = json.load(fh).get("jobs", [])
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            jobs = []
        for j in jobs:
            if j.get("status") == "running":  # can't still be running after restart
                j["status"] = "failed"
                j["error"] = "interrupted by server restart"
                j["ended_at"] = _now()
        self.jobs = jobs

    def _save_locked(self) -> None:
        tmp = self.jobs_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"jobs": self.jobs, "saved_at": _now()}, fh)
            os.replace(tmp, self.jobs_file)
        except OSError:
            pass

    # ---- public API -------------------------------------------------------
    def enqueue(self, *, run_id: str, models: list[str], construction: str,
                overlap: int, source_path: str | None = None,
                source_label: str | None = None) -> tuple[str | None, str]:
        """Queue a launch. ``source_path`` set => a fresh ingest of that file;
        unset => the run must already exist under --lib. Returns (job_id, msg)."""
        run_id = (run_id or "").strip()
        if not run_id:
            return None, "run_id is required"
        if not models:
            return None, "select at least one model"
        stages = list(STAGES)
        if source_path is None:
            stages.remove("ingest")
            src_json = os.path.join(self.lib_root, run_id, "source", "source.json")
            if not os.path.isfile(src_json):
                return None, "run %r is not ingested yet — upload a file instead" % run_id
        job = {
            "id": uuid.uuid4().hex[:12],
            "run_id": run_id,
            "models": list(models),
            "construction": construction,
            "overlap": int(overlap),
            "source_path": source_path,
            "source_label": source_label,
            "stages": stages,
            "stage": None,             # currently-running stage name
            "stage_index": -1,
            "stage_results": {},       # stage -> {"ok": bool, "summary": str}
            "status": "queued",
            "error": None,
            "log_tail": [],
            "created_at": _now(),
            "started_at": None,
            "ended_at": None,
        }
        with self.cv:
            active = [j for j in self.jobs
                      if j["run_id"] == run_id and j["status"] in ("queued", "running")]
            if active:
                return None, "run %r already has an active launch job" % run_id
            self.jobs.append(job)
            self._save_locked()
            self.cv.notify_all()
        return job["id"], "queued"

    def snapshot(self) -> dict:
        now = _now()
        with self.lock:
            out = []
            active = 0
            for j in self.jobs:
                if j["status"] in ("queued", "running"):
                    active += 1
                item = dict(j)
                started, ended = j.get("started_at"), j.get("ended_at")
                item["elapsed_s"] = (ended - started) if (started and ended) \
                    else ((now - started) if started else None)
                out.append(item)
            out.reverse()  # newest first
            return {"jobs": out, "active": active, "concurrency": 1,
                    "server_time": int(now)}

    def cancel(self, job_id: str) -> tuple[bool, str]:
        with self.cv:
            j = next((x for x in self.jobs if x["id"] == job_id), None)
            if not j:
                return False, "not found"
            if j["status"] == "queued":
                j["status"] = "cancelled"
                j["ended_at"] = _now()
                self._save_locked()
                self._write_run_state(j, "CANCELLED", note="launch cancelled while queued")
                return True, "cancelled"
            if j["status"] == "running":
                self._cancel.add(job_id)
                proc = self._proc.get(job_id)
                if proc:
                    _kill_group(proc)  # terminate the stage's process group
                return True, "cancelling"
            return False, "job is not active (%s)" % j["status"]

    def remove(self, job_id: str) -> tuple[bool, str]:
        with self.cv:
            j = next((x for x in self.jobs if x["id"] == job_id), None)
            if not j:
                return False, "not found"
            if j["status"] not in TERMINAL:
                return False, "cannot remove an active job"
            self.jobs.remove(j)
            self._save_locked()
            return True, "removed"

    def shutdown(self) -> None:
        with self.lock:
            self._shutting_down = True
            procs = list(self._proc.values())
        for p in procs:
            _kill_group(p)
        with self.cv:
            self.cv.notify_all()

    # ---- run_state streaming (data-model-v2 table, plain sqlite3) ---------
    def _write_run_state(self, job: dict, state: str, *, note: str = "") -> None:
        """Record launcher progress in the run's run_state table so the GUI can
        animate. Preserves pi_session; owns only the ``launcher`` budget key."""
        man = os.path.join(self.lib_root, job["run_id"], "manifest.sqlite")
        if not os.path.isdir(os.path.dirname(man)):
            return  # run dir not created yet (ingest not run / failed early)
        try:
            conn = sqlite3.connect(man, timeout=5.0)
        except sqlite3.Error:
            return
        try:
            conn.executescript(_RUN_STATE_SCHEMA)
            row = conn.execute("SELECT pi_session, budgets_json FROM run_state WHERE track_id=?",
                               (job["run_id"],)).fetchone()
            pi_session = row[0] if row else None
            try:
                budgets = json.loads(row[1]) if row and row[1] else {}
            except ValueError:
                budgets = {}
            budgets["launcher"] = {
                "job_id": job["id"], "stage": job.get("stage"),
                "stage_index": job.get("stage_index"), "n_stages": len(job["stages"]),
                "status": job["status"], "note": note, "updated_at": _utc(),
            }
            conn.execute(
                "INSERT OR REPLACE INTO run_state (track_id, state, pi_session, budgets_json, updated_at) "
                "VALUES (?,?,?,?,?)",
                (job["run_id"], state, pi_session, json.dumps(budgets), _utc()))
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    # ---- stage commands ---------------------------------------------------
    def _stage_cmd(self, job: dict, stage: str) -> list[str]:
        tmpl = (os.environ.get("AUDIO_EXTRACT_V2_STAGE_CMD") or "").strip()
        if tmpl:  # test seam
            toks = shlex.split(tmpl)
            sub = {"{stage}": stage, "{run_id}": job["run_id"], "{lib}": self.lib_root}
            out = []
            for t in toks:
                for k, v in sub.items():
                    t = t.replace(k, v)
                out.append(t)
            return out
        base = ["uv", "run", "audio-extract", "--lib", self.lib_root]
        rid = job["run_id"]
        models = ",".join(job["models"])
        mdir = ["--model-dir", self.model_dir] if self.model_dir else []
        if stage == "ingest":
            return base + ["ingest", job["source_path"], "--run-id", rid]
        if stage == "passages_mine":
            return base + ["passages", "mine", "--run-id", rid,
                           "--overlap", str(job["overlap"])] + mdir
        if stage == "panel_render":
            return base + ["panel", "render", "--run-id", rid, "--models", models,
                           "--construction", job["construction"],
                           "--overlap", str(job["overlap"]),
                           "--lock", self.lock_path] + mdir
        if stage == "qa_score":
            return base + ["qa", "score", "--run-id", rid]
        if stage == "challenges_build":
            return base + ["challenges", "build", "--run-id", rid]
        if stage == "challenges_run":
            return base + ["challenges", "run", "--run-id", rid, "--models", models,
                           "--lock", self.lock_path] + mdir
        if stage == "select_autonomous":
            return base + ["select", "autonomous", "--run-id", rid]
        raise ValueError("unknown stage %r" % stage)

    # ---- worker -----------------------------------------------------------
    def _next_queued_locked(self):
        for j in self.jobs:
            if j["status"] == "queued":
                return j
        return None

    def _run_loop(self) -> None:
        while True:
            with self.cv:
                if self._shutting_down:
                    return
                job = self._next_queued_locked()
                while job is None:
                    self.cv.wait()
                    if self._shutting_down:
                        return
                    job = self._next_queued_locked()
                job["status"] = "running"
                job["started_at"] = _now()
                self._save_locked()
                jid = job["id"]
            try:
                self._execute(jid)
            except Exception as exc:  # the worker thread must never die
                with self.cv:
                    j = next((x for x in self.jobs if x["id"] == jid), None)
                    if j:
                        j["status"] = "failed"
                        j["error"] = "runner error: %s" % exc
                        j["ended_at"] = _now()
                        self._save_locked()

    def _job(self, jid: str) -> dict | None:
        return next((x for x in self.jobs if x["id"] == jid), None)

    def _execute(self, jid: str) -> None:
        with self.lock:
            job = self._job(jid)
            if job is None:
                return
            stages = list(job["stages"])

        for i, stage in enumerate(stages):
            with self.cv:
                job = self._job(jid)
                if job is None:
                    return
                if jid in self._cancel:
                    break
                job["stage"] = stage
                job["stage_index"] = i
                self._save_locked()
            self._write_run_state(job, "LAUNCH:%s" % stage)
            ok, summary = self._run_stage(jid, stage)
            with self.cv:
                job = self._job(jid)
                if job is None:
                    return
                job["stage_results"][stage] = {"ok": ok, "summary": summary}
                cancelled = jid in self._cancel
                self._save_locked()
            if cancelled:
                break
            if not ok:
                with self.cv:
                    job["status"] = "failed"
                    job["error"] = "stage %s failed: %s" % (stage, summary)
                    job["ended_at"] = _now()
                    self._save_locked()
                self._write_run_state(job, "FAILED", note=job["error"])
                return

        with self.cv:
            job = self._job(jid)
            if job is None:
                return
            cancelled = jid in self._cancel
            self._cancel.discard(jid)
            job["ended_at"] = _now()
            if cancelled:
                job["status"] = "cancelled"
            elif job["status"] == "running":
                job["status"] = "done"
                job["stage"] = None
            self._save_locked()
        if job["status"] == "cancelled":
            self._write_run_state(job, "CANCELLED", note="launch cancelled")
        else:
            self._write_run_state(job, "LAUNCH:done")

    def _run_stage(self, jid: str, stage: str) -> tuple[bool, str]:
        with self.lock:
            job = self._job(jid)
            if job is None:
                return False, "job vanished"
            cmd = self._stage_cmd(job, stage)
        try:
            proc = subprocess.Popen(
                cmd, cwd=self.repo_root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                start_new_session=True,  # own group -> cancel kills the whole tree
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
            )
        except OSError as exc:
            return False, "could not start %s: %s" % (cmd[0], exc)
        with self.lock:
            self._proc[jid] = proc
            # close the Popen->register race: a cancel that arrived while the
            # process was starting hasn't seen it yet — kill it now.
            if jid in self._cancel:
                _kill_group(proc)

        last_json_line = None
        stdout = proc.stdout
        if stdout is not None:                       # PIPE guarantees non-None;
            for raw in iter(stdout.readline, ""):    # guard defensively anyway
                line = raw.rstrip("\n")
                if line.strip():
                    if line.lstrip().startswith("{"):
                        last_json_line = line.strip()
                    with self.cv:
                        j = self._job(jid)
                        if j is not None:
                            tail = j["log_tail"]
                            tail.append("[%s] %s" % (stage, line[:400]))
                            if len(tail) > LOG_TAIL_MAX:
                                del tail[:-LOG_TAIL_MAX]
            stdout.close()
        proc.wait()
        with self.lock:
            self._proc.pop(jid, None)
            cancelled = jid in self._cancel

        if cancelled:
            return False, "cancelled"
        envelope = None
        if last_json_line:
            try:
                envelope = json.loads(last_json_line)
            except ValueError:
                envelope = None
        if proc.returncode != 0:
            msg = (envelope or {}).get("message") or "exit code %s" % proc.returncode
            return False, msg
        if isinstance(envelope, dict) and envelope.get("ok") is False:
            return False, envelope.get("message") or envelope.get("status") or "not ok"
        summary = ""
        if isinstance(envelope, dict):
            bits = []
            for k in ("passage_count", "rendered", "built", "ran", "candidate_count", "status"):
                v = envelope.get(k)
                if isinstance(v, list):
                    bits.append("%s=%d" % (k, len(v)))
                elif v not in (None, ""):
                    bits.append("%s=%s" % (k, v))
            summary = " ".join(bits)[:200]
        return True, summary or "ok"
