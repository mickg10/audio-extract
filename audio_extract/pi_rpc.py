"""Pi RPC planner (v2.1 WP10 / §15.2) — LF-delimited JSONL over stdin/stdout.

The production planner runs DeepSeek V4 Pro *through Pi* (which owns provider
auth, ``reasoning_content`` replay, and session persistence) — do not extend the
hand-written HTTP ``DeepSeekPlanner`` (kept only as a dev stopgap). One Pi session
per run; the session is advisory, SQLite/manifests stay authoritative.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .planning import validate_planner_response

PI_ARGS = [
    "--mode", "rpc",
    "--provider", "deepseek",
    "--model", "deepseek-v4-pro",        # §15.6: not the legacy deepseek-reasoner alias
    "--thinking", "high",
    "--no-builtin-tools",
]


class PiUnavailableError(RuntimeError):
    """`pi` is not installed/configured; use DeterministicFallbackPlanner."""


class PiRpcPlanner:
    """Minimal JSONL client: send the evidence report, read one planner response.
    Records are split ONLY on \\n (§ Pi RPC contract)."""

    def __init__(self, session_dir: str | Path, extension: str = ".pi/extensions/audio-extract.ts",
                 pi_binary: str = "pi", timeout_s: float = 300.0, max_retries: int = 2):
        if shutil.which(pi_binary) is None:
            raise PiUnavailableError(
                f"{pi_binary!r} not found on PATH; install pi-mono and configure "
                "~/.pi/agent/models.json (see docs/pi/models.json.example)")
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.cmd = [pi_binary, *PI_ARGS,
                    "--session-dir", str(self.session_dir),
                    "--extension", extension, "--approve"]
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def propose(self, report: dict) -> dict:
        last_err = "empty response"
        for attempt in range(self.max_retries + 1):
            try:
                proc = subprocess.run(
                    self.cmd, input=json.dumps(report) + "\n",
                    capture_output=True, text=True, timeout=self.timeout_s)
                for line in proc.stdout.splitlines():   # split ONLY on \n
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ok, reason = validate_planner_response(d)
                    if ok:
                        return d
                    last_err = reason
            except subprocess.TimeoutExpired:
                last_err = f"pi rpc timeout after {self.timeout_s}s (attempt {attempt + 1})"
        return {"status": "no_more_useful_probes",
                "reason": f"planner unavailable/invalid after retries: {last_err}"}
