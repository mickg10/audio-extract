#!/usr/bin/env python3
"""Background job queue + runner for on-demand file processing.

A single worker thread runs one `convert.py file <path>` subprocess at a time
(the MPS/GPU is single -- never two conversions at once; extra requests queue).
State is persisted to jobs.json so jobs survive a server restart; a job that was
"running" when the server died is marked interrupted on the next load.

Stop / cancel semantics
-----------------------
* Each subprocess is started in its OWN session/process-group (start_new_session
  = setsid). Stopping a running job therefore kills the WHOLE group
  (`os.killpg`), i.e. the `uv`/`python` process AND every child it spawned
  (model workers, ffmpeg) -- not just the top process.
* Cancelling a still-queued job just marks it canceled so the worker skips it.

Stdlib only. The command is overridable via $AUDIO_EXTRACT_JOB_CMD (a template
where `{path}` is substituted, e.g. for testing) -- default is the real
`uv run python convert.py file {path}`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid

# Terminal states (a job here can be Removed from the list).
TERMINAL = {"done", "failed", "stopped", "canceled"}
LOG_TAIL_MAX = 40

# stdout markers (see the processing contract) -> (phase label, coarse progress).
# progress only ever moves forward.
_SLUG_RE = re.compile(r"\[manifest\]\s+(.+?):")


def _default_cmd(path):
    return ["uv", "run", "python", "convert.py", "file", path]


def _build_cmd(path):
    tmpl = (os.environ.get("AUDIO_EXTRACT_JOB_CMD") or "").strip()
    if not tmpl:
        return _default_cmd(path)
    toks = shlex.split(tmpl)
    if any("{path}" in t for t in toks):
        return [t.replace("{path}", path) for t in toks]
    return toks + [path]


def _now():
    return time.time()


class JobManager:
    def __init__(self, root, input_dir, jobs_file):
        self.root = root                # project root: cwd for the subprocess
        self.input_dir = input_dir
        self.jobs_file = jobs_file
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.jobs = []                  # list of dicts, creation order
        self._proc = {}                 # id -> Popen (running only, not persisted)
        self._stop = set()              # ids asked to stop while running
        self._shutting_down = False
        self._load()
        self._worker = threading.Thread(target=self._run_loop, name="job-worker",
                                        daemon=True)
        self._worker.start()

    # ---- persistence --------------------------------------------------
    def _load(self):
        try:
            with open(self.jobs_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            jobs = data.get("jobs", []) if isinstance(data, dict) else []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            jobs = []
        for j in jobs:
            # A job that was mid-run when the server stopped can't still be
            # running -- mark it interrupted so the queue is consistent.
            if j.get("status") == "running":
                j["status"] = "failed"
                j["phase"] = "interrupted (server restarted)"
                j["error"] = "interrupted by server restart"
                j["ended_at"] = _now()
        self.jobs = jobs

    def _save_locked(self):
        tmp = self.jobs_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"jobs": self.jobs, "saved_at": _now()}, fh)
            os.replace(tmp, self.jobs_file)
        except OSError:
            pass

    # ---- public API ---------------------------------------------------
    def enqueue(self, filename, path):
        job = {
            "id": uuid.uuid4().hex[:12],
            "filename": filename,
            "path": path,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "slug": None,
            "created_at": _now(),
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "error": None,
            "log_tail": [],
        }
        with self.cv:
            self.jobs.append(job)
            self._save_locked()
            self.cv.notify_all()
        return job["id"]

    def snapshot(self):
        """A JSON-serialisable view for /api/jobs (newest first, with elapsed)."""
        now = _now()
        with self.lock:
            out = []
            active = 0
            for j in self.jobs:
                if j["status"] in ("queued", "running"):
                    active += 1
                started, ended = j.get("started_at"), j.get("ended_at")
                if started and ended:
                    elapsed = ended - started
                elif started:
                    elapsed = now - started
                else:
                    elapsed = None
                item = dict(j)
                item["elapsed_s"] = elapsed
                out.append(item)
            out.reverse()  # newest first
            return {"jobs": out, "active": active, "server_time": int(now)}

    def action(self, job_id, action):
        """Dispatch stop|cancel|remove. Returns (ok, message)."""
        if action == "remove":
            return self._remove(job_id)
        if action in ("stop", "cancel"):
            return self._stop_or_cancel(job_id, action)
        return False, "unknown action"

    def _find_locked(self, job_id):
        for j in self.jobs:
            if j["id"] == job_id:
                return j
        return None

    def _stop_or_cancel(self, job_id, action):
        with self.cv:
            j = self._find_locked(job_id)
            if not j:
                return False, "not found"
            if j["status"] == "queued":
                j["status"] = "canceled"
                j["phase"] = "canceled"
                j["ended_at"] = _now()
                self._save_locked()
                self.cv.notify_all()
                return True, "canceled"
            if j["status"] == "running":
                self._stop.add(job_id)
                proc = self._proc.get(job_id)
                if proc:
                    _kill_group(proc)      # SIGTERM group now, SIGKILL escalation
                return True, "stopping"
            return False, "job is not active (%s)" % j["status"]

    def _remove(self, job_id):
        with self.cv:
            j = self._find_locked(job_id)
            if not j:
                return False, "not found"
            if j["status"] not in TERMINAL:
                return False, "cannot remove an active job"
            self.jobs.remove(j)
            self._proc.pop(job_id, None)
            self._stop.discard(job_id)
            self._save_locked()
            return True, "removed"

    def shutdown(self):
        """Kill any running subprocess group so we never orphan a conversion."""
        with self.lock:
            self._shutting_down = True
            procs = list(self._proc.values())
        for p in procs:
            _kill_group(p)

    # ---- worker -------------------------------------------------------
    def _next_queued_locked(self):
        for j in self.jobs:
            if j["status"] == "queued":
                return j
        return None

    def _run_loop(self):
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
                job["phase"] = "starting"
                job["progress"] = max(job.get("progress", 0), 0.03)
                self._save_locked()
                jid, path = job["id"], job["path"]
            try:
                self._execute(jid, path)
            except Exception as exc:               # worker must never die
                with self.cv:
                    j = self._find_locked(jid)
                    if j:
                        j["status"] = "failed"
                        j["error"] = "runner error: %s" % exc
                        j["ended_at"] = _now()
                        self._save_locked()

    def _execute(self, jid, path):
        cmd = _build_cmd(path)
        try:
            proc = subprocess.Popen(
                cmd, cwd=self.root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                start_new_session=True,             # own process group -> group kill
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
            )
        except Exception as exc:
            with self.cv:
                j = self._find_locked(jid)
                if j:
                    j["status"] = "failed"
                    j["error"] = "could not start: %s" % exc
                    j["ended_at"] = _now()
                    self._save_locked()
            return

        with self.lock:
            self._proc[jid] = proc

        saw_done = False
        # readline (not `for line in proc.stdout`) so phases update promptly.
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip("\n")
            if "DONE: processed" in line:
                saw_done = True
            self._on_line(jid, line)
        proc.stdout.close()
        proc.wait()
        rc = proc.returncode

        with self.cv:
            j = self._find_locked(jid)
            stopped = jid in self._stop
            self._proc.pop(jid, None)
            self._stop.discard(jid)
            if j:
                j["exit_code"] = rc
                j["ended_at"] = _now()
                if stopped:
                    j["status"] = "stopped"
                    j["phase"] = "stopped"
                elif rc == 0 and saw_done:
                    j["status"] = "done"
                    j["phase"] = "done"
                    j["progress"] = 1.0
                else:
                    j["status"] = "failed"
                    if not j.get("error"):
                        j["error"] = "exited %s without DONE marker" % rc
                self._save_locked()

    def _on_line(self, jid, line):
        phase, prog = _phase_for(line)
        with self.cv:
            j = self._find_locked(jid)
            if not j:
                return
            tail = j["log_tail"]
            if line.strip():
                tail.append(line[:500])
                if len(tail) > LOG_TAIL_MAX:
                    del tail[:-LOG_TAIL_MAX]
            slug_m = _SLUG_RE.search(line)
            if slug_m and not j.get("slug"):
                j["slug"] = slug_m.group(1)
            changed = False
            if phase and phase != j.get("phase"):
                j["phase"] = phase
                changed = True
            if prog is not None and prog > j.get("progress", 0):
                j["progress"] = prog
                changed = True
            if changed:
                self._save_locked()      # persist on phase change, not every line


def _phase_for(line):
    """Map a convert.py stdout line to (phase, progress) or (None, None)."""
    if "JOB: processing" in line:
        return "starting", 0.05
    if "already separated" in line:
        return "separating (models cached)", 0.35
    if line.startswith("=== loading") or "=== loading" in line:
        return "separating (running models)", 0.25
    if "[ensemble]" in line:
        return "ensembling", 0.60
    if "[manifest]" in line:
        return "variants written", 0.75
    if "normalized" in line and "instrumental" in line:
        return "normalizing", 0.85
    if "transcoded" in line and "AAC" in line:
        return "transcoding (AAC)", 0.95
    if "DONE: processed" in line:
        return "finalizing", 0.99
    return None, None


def _kill_group(proc):
    """SIGTERM the process's whole group now; SIGKILL it after a short grace."""
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
