#!/usr/bin/env python3
"""Read-only web GUI for the audio-extract **v2** deterministic library.

Serves a mobile-friendly passages/A-B-C view over a v2 ``--lib`` root (the
content-addressed tree produced by the ``audio-extract`` CLI — see
``audio_extract/storage.py`` and ``docs/v2``). This is a *sibling* to the v1
``server.py`` (which serves ``convert.py``'s ``lib/`` tree); v1 is untouched.

What it exposes (all read-only, all defensive about partial/missing data):

* ``GET /``                         the passages page (client picks a run)
* ``GET /r/<track_id>``             the passages page, deep-linked to one run
* ``GET /api/v2/runs``              every run under --lib (source/state/counts)
* ``GET /api/v2/run/<track_id>``    source.json + passages.v1.json + candidates
                                    (recipe.json + manifest metric costs +
                                    Pareto frontier)
* ``GET /api/v2/audio/<t>/source.m4a``                on-the-fly AAC of the source
* ``GET /api/v2/audio/<t>/candidate/<dir>.m4a``       on-the-fly AAC of a candidate
* ``GET /api/v2/peaks/<t>/...json``                   cached waveform peaks (numpy)
* ``GET /api/v2/spectrogram/<t>/...png``              cached spectrogram (matplotlib)

The peaks/spectrogram/audio endpoints accept ``?start=<sample>&end=<sample>`` so
the UI can render a passage-focused window. Raw float32 PCM is **never** sent to
the browser — audio is always transcoded to AAC/m4a with ffmpeg first. Generated
artifacts are cached under ``<lib>/.gui_cache`` (keyed by source path+mtime+args),
so the second request is a plain static file serve with HTTP Range (seeking).

Run:  uv run python web/server_v2.py --lib /path/to/lib   [--port 8731]

Only Python stdlib + numpy + matplotlib + soundfile + ffmpeg are used (all already
project deps); no third-party web framework, no CDN.
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
STATIC_DIR = os.path.join(WEB_DIR, "static")
DEFAULT_PORT = 8731  # v1 lives on 8730

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
            candidates.append({
                "dir": d,
                "recipe_id": recipe_id,
                "short_id": d[len("sha256_"):][:8],
                "label": _candidate_label(recipe),
                "operation": (recipe.get("operation", {}) or {}),
                "model": (recipe.get("model", {}) or {}),
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
# HTTP handler
# ---------------------------------------------------------------------------
API_RUN_RE = re.compile(r"^/api/v2/run/(.+)$")
AUDIO_RE = re.compile(r"^/api/v2/audio/([^/]+)/(source|provisional_vocal|candidate/[^/]+)\.m4a$")
PEAKS_RE = re.compile(r"^/api/v2/peaks/([^/]+)/(source|candidate/[^/]+)\.json$")
SPEC_RE = re.compile(r"^/api/v2/spectrogram/([^/]+)/(source|candidate/[^/]+)\.png$")
RUN_PAGE_RE = re.compile(r"^/r/([^/]+)/?$")


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
            if path in ("/", "/index.html"):
                self._serve_frontend("passages.html", "text/html; charset=utf-8")
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

            m = API_RUN_RE.match(path)
            if m:
                tid = urllib.parse.unquote(m.group(1)).strip("/")
                payload = build_run_payload(lib, tid)
                if payload is None:
                    self._send_json({"error": "run not found", "track_id": tid}, 404)
                else:
                    self._send_json(payload)
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
    ap = argparse.ArgumentParser(description="audio-extract v2 read-only web GUI")
    ap.add_argument("--lib", required=True, help="v2 library root (contains <track_id>/ dirs)")
    ap.add_argument("--cache-dir", default=None,
                    help="where to cache generated peaks/spectrograms/m4a (default: <lib>/.gui_cache)")
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
    sys.stderr.write(
        "audio-extract v2 GUI serving lib=%s on http://%s:%d\n  cache -> %s | ffmpeg -> %s\n"
        % (lib.root, args.host, args.port, lib.cache_dir, FFMPEG or "MISSING")
    )

    def _term(_signum, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _term)

    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        sys.stderr.write("\nshutting down\n")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
