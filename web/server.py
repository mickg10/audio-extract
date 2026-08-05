#!/usr/bin/env python3
"""Read-only web GUI for the audio-extract stem-separation library.

Serves a listing page (`/`) and a per-file explorer (`/f/<slug>`) over the
`lib/` tree produced by convert.py (see ../SCHEMA.md). Pure Python standard
library only -- no third-party deps -- so it never disturbs the uv environment
that the background batch is actively using.

Design notes
------------
* ThreadingHTTPServer: concurrent requests so audio streaming (Range) does not
  block API/asset calls.
* Everything is read-only. Data appears incrementally while the batch runs, so
  every read is defensive: missing files, half-written manifest lines and a
  stale index.json are all handled without crashing.
* Two small JSON APIs feed a plain-JS frontend:
    GET /api/index          -> the listing (index.json merged with a live dir scan)
    GET /api/file/<slug>    -> that file's parsed+ordered manifest, with asset URLs
* Static assets under lib/ (audio/png/peaks) are served from /lib/... with full
  HTTP Range support so <audio> seeking works.

Run:  cd ~/audio-extract && uv run python web/server.py   [--port 8730]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import re
import signal
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(WEB_DIR, "static")
LIB_DIR = os.path.abspath(os.path.join(WEB_DIR, os.pardir, "lib"))
ROOT_DIR = os.path.abspath(os.path.join(WEB_DIR, os.pardir))  # convert.py's cwd

DEFAULT_PORT = 8730

# --- upload + jobs config (all overridable by env, e.g. for tests) ---
INPUT_DIR = os.path.abspath(os.environ.get(
    "AUDIO_EXTRACT_INPUT_DIR", os.path.join(ROOT_DIR, "input")))
JOBS_FILE = os.environ.get(
    "AUDIO_EXTRACT_JOBS_FILE", os.path.join(WEB_DIR, "jobs.json"))
ALLOWED_UPLOAD_EXT = {".mp3", ".wav"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "1024")) * 1024 * 1024

# jobs.py sits next to this file; make it importable no matter the cwd.
sys.path.insert(0, WEB_DIR)
from jobs import JobManager  # noqa: E402

# Audio containers we recognise as a file's "original".
ORIGINAL_NAMES = ("original.mp3", "original.wav", "original.flac", "original.m4a")

# Extra MIME types the stdlib map sometimes misses.
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("image/png", ".png")


# ---------------------------------------------------------------------------
# Ordering (per SCHEMA.md): original -> instrumentals -> vocals -> others -> diffs
# ---------------------------------------------------------------------------
def _model_rank(entry: dict) -> int:
    """Sub-order within the instrumentals group: roformer, mdx23c, ensemble,
    de-echo/de-reverb, else. Keyed primarily off the id (descriptions name the
    constituent models, so an ensemble's text contains 'roformer' + 'mdx' and
    must be classified before those substring checks)."""
    idl = (entry.get("id") or "").lower()
    text = "{} {} {}".format(
        entry.get("id", ""), entry.get("description", ""), entry.get("stem", "")
    ).lower()
    if entry.get("kind") == "ensemble" or "ensemble" in idl:
        return 2
    if any(k in idl or k in text for k in ("deecho", "de-echo", "dereverb", "de-reverb")):
        return 3
    if "roformer" in idl:
        return 0
    if "mdx" in idl:
        return 1
    if "roformer" in text:
        return 0
    if "mdx" in text:
        return 1
    return 5


def variant_sort_key(entry: dict):
    """Return a sortable key implementing the SCHEMA display order."""
    kind = (entry.get("kind") or "").lower()
    stem = (entry.get("stem") or "").lower()
    if kind == "source" or entry.get("id") == "original":
        group = 0
    elif kind == "diff":
        group = 4
    elif "instrument" in stem:
        group = 1
    elif "vocal" in stem:
        group = 2
    else:
        group = 3  # any other separate/ensemble output
    return (group, _model_rank(entry), entry.get("id", ""))


GROUP_LABEL = {0: "original", 1: "instrumental", 2: "vocals", 3: "other", 4: "diff"}


# ---------------------------------------------------------------------------
# Data access helpers (all tolerant of partial / missing data)
# ---------------------------------------------------------------------------
def read_index() -> dict:
    """Load lib/index.json, tolerating absence or a mid-write truncated file."""
    path = os.path.join(LIB_DIR, "index.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


def find_original(slug_dir: str):
    """Return (filename, size) of the original audio in a slug dir, or (None, 0)."""
    for name in ORIGINAL_NAMES:
        p = os.path.join(slug_dir, name)
        if os.path.isfile(p):
            try:
                return name, os.path.getsize(p)
            except OSError:
                return name, 0
    return None, 0


def read_manifest(slug: str) -> list:
    """Parse lib/<slug>/manifest.jsonl into a list of dicts.

    Skips blank lines and any line that does not parse as JSON (e.g. the final
    line while convert.py is mid-write). Never raises.
    """
    path = os.path.join(LIB_DIR, slug, "manifest.jsonl")
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written trailing line -- ignore
                if isinstance(obj, dict) and obj.get("id"):
                    entries.append(obj)
    except (FileNotFoundError, OSError):
        pass
    return entries


def asset_url(slug: str, rel: str | None) -> str | None:
    """Build a /lib/<slug>/<rel> URL if <rel> is set and the file exists."""
    if not rel:
        return None
    disk = os.path.join(LIB_DIR, slug, rel)
    if not os.path.isfile(disk):
        return None
    quoted = "/".join(urllib.parse.quote(part) for part in [slug] + rel.split("/"))
    return "/lib/" + quoted


def dir_manifest_count(slug: str) -> int:
    """Cheap count of non-original variants from a manifest without full parse cost."""
    return sum(1 for e in read_manifest(slug) if e.get("id") != "original")


def build_index_payload() -> dict:
    """Merge index.json with a live scan of lib/ so the listing is useful even
    while index.json lags behind the batch."""
    idx = read_index()
    by_slug = {}

    # 1) Seed from index.json (the canonical source once it is regenerated).
    for f in idx.get("files", []) or []:
        slug = f.get("slug")
        if not slug:
            continue
        by_slug[slug] = {
            "slug": slug,
            "title": f.get("title") or slug,
            "original": f.get("original"),
            "duration_s": f.get("duration_s"),
            "n_variants": f.get("n_variants"),
            "from_index": True,
        }

    # 2) Supplement with any slug directory on disk (the batch creates these
    #    before index.json catches up).
    try:
        dir_entries = sorted(os.listdir(LIB_DIR))
    except OSError:
        dir_entries = []
    for name in dir_entries:
        slug_dir = os.path.join(LIB_DIR, name)
        if not os.path.isdir(slug_dir):
            continue
        orig_name, _ = find_original(slug_dir)
        rec = by_slug.get(name)
        has_manifest = os.path.isfile(os.path.join(slug_dir, "manifest.jsonl"))
        if rec is None:
            if orig_name is None and not has_manifest:
                continue  # empty / unrelated dir
            rec = {
                "slug": name,
                "title": name,
                "original": orig_name,
                "duration_s": None,
                "n_variants": None,
                "from_index": False,
            }
            by_slug[name] = rec
        # Fill any gaps from disk truth.
        if not rec.get("original") and orig_name:
            rec["original"] = orig_name
        rec["on_disk"] = os.path.isdir(slug_dir)
        rec["has_manifest"] = has_manifest

    # 3) Enrich each record: resolve original URL, live variant count, status.
    files = []
    for slug, rec in by_slug.items():
        slug_dir = os.path.join(LIB_DIR, slug)
        on_disk = os.path.isdir(slug_dir)
        orig = rec.get("original")
        if orig and not os.path.isfile(os.path.join(slug_dir, orig)):
            # index named an original that is not present yet; re-detect.
            detected, _ = find_original(slug_dir)
            orig = detected or orig
        has_manifest = os.path.isfile(os.path.join(slug_dir, "manifest.jsonl"))

        # Prefer a live manifest count; fall back to index's n_variants.
        n_variants = rec.get("n_variants")
        if has_manifest:
            live = dir_manifest_count(slug)
            if live:
                n_variants = live

        # Duration: index value, else pull from the manifest's source line.
        duration = rec.get("duration_s")
        if duration is None and has_manifest:
            for e in read_manifest(slug):
                if e.get("id") == "original" or e.get("kind") == "source":
                    duration = e.get("duration_s")
                    break

        if not on_disk:
            status = "pending"
        elif has_manifest:
            status = "ready"
        elif orig:
            status = "processing"
        else:
            status = "pending"

        original_url = asset_url(slug, orig) if orig else None
        download_url = None
        if original_url:
            dl_name = "{}{}".format(slug, os.path.splitext(orig)[1])
            download_url = original_url + "?dl=" + urllib.parse.quote(dl_name)

        files.append(
            {
                "slug": slug,
                "title": rec.get("title") or slug,
                "original": orig,
                "original_url": original_url,
                "download_url": download_url,
                "explorer_url": "/f/" + urllib.parse.quote(slug),
                "duration_s": duration,
                "n_variants": n_variants,
                "status": status,
            }
        )

    files.sort(key=lambda f: f["slug"].lower())
    return {
        "generated": idx.get("generated"),
        "count": len(files),
        "index_count": idx.get("count"),
        "server_time": int(time.time()),
        "files": files,
    }


def build_file_payload(slug: str):
    """Return the explorer payload for one slug, or None if the dir is absent."""
    slug_dir = os.path.join(LIB_DIR, slug)
    if not os.path.isdir(slug_dir):
        return None

    entries = read_manifest(slug)
    orig_name, _ = find_original(slug_dir)

    # If the manifest has no source line yet but an original file exists,
    # synthesise a minimal "original" entry so the page still has a player.
    have_source = any(
        e.get("id") == "original" or e.get("kind") == "source" for e in entries
    )
    if not have_source and orig_name:
        entries.insert(
            0,
            {
                "id": "original",
                "kind": "source",
                "stem": "original",
                "description": "Original mix",
                "output": orig_name,
                "spectrogram": os.path.splitext(orig_name)[0] + ".png",
                "peaks": os.path.splitext(orig_name)[0] + ".peaks.json",
                "status": "done",
                "synthetic": True,
            },
        )

    entries.sort(key=variant_sort_key)

    variants = []
    for e in entries:
        out_rel = e.get("output")
        stream_rel = e.get("stream") or out_rel          # compact AAC for playback
        audio_url = asset_url(slug, stream_rel)
        spec_url = asset_url(slug, e.get("spectrogram"))
        peaks_url = asset_url(slug, e.get("peaks"))
        download_url = None
        if audio_url:
            ext = os.path.splitext(stream_rel)[1] or ".m4a"
            dl_name = "{}__{}{}".format(slug, e.get("id", "variant"), ext)
            download_url = audio_url + "?dl=" + urllib.parse.quote(dl_name)
        # lossless WAV as a secondary download (only when the stream is a different file)
        lossless_url = None
        if out_rel and out_rel != stream_rel:
            lu = asset_url(slug, out_rel)
            if lu:
                wext = os.path.splitext(out_rel)[1] or ".wav"
                lossless_url = lu + "?dl=" + urllib.parse.quote(
                    "{}__{}{}".format(slug, e.get("id", "variant"), wext))

        group = variant_sort_key(e)[0]
        v = dict(e)  # keep every manifest field for the UI
        v.update(
            {
                "audio_url": audio_url,
                "spectrogram_url": spec_url,
                "peaks_url": peaks_url,
                "download_url": download_url,
                "lossless_url": lossless_url,
                "audio_ready": audio_url is not None,
                "spec_ready": spec_url is not None,
                "peaks_ready": peaks_url is not None,
                "group": group,
                "group_label": GROUP_LABEL.get(group, "other"),
            }
        )
        variants.append(v)

    # Title / duration from index if we have them.
    idx = read_index()
    title, duration = slug, None
    for f in idx.get("files", []) or []:
        if f.get("slug") == slug:
            title = f.get("title") or slug
            duration = f.get("duration_s")
            break
    if duration is None:
        for e in entries:
            if e.get("id") == "original" or e.get("kind") == "source":
                duration = e.get("duration_s")
                break

    return {
        "slug": slug,
        "title": title,
        "original": orig_name,
        "duration_s": duration,
        "has_manifest": os.path.isfile(os.path.join(slug_dir, "manifest.jsonl")),
        "n_variants": sum(1 for v in variants if v.get("id") != "original"),
        "server_time": int(time.time()),
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")


def sanitize_filename(name: str) -> str:
    """Reduce a client filename to a safe basename (no path, safe chars)."""
    name = os.path.basename((name or "").replace("\\", "/"))
    name = _SAFE_CHARS.sub("_", name).strip().strip(".") or "upload"
    return name[:200]


def unique_path(directory: str, filename: str) -> str:
    """Return a non-colliding path in <directory> (adds _1, _2, ... if needed)."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, "%s_%d%s" % (base, i, ext))
        i += 1
    return candidate


def parse_multipart_formdata(body: bytes, boundary: bytes):
    """Minimal in-memory multipart/form-data parser.

    Returns a list of {name, filename, data}. Kept in memory (uploads are
    audio files of bounded size; the handler enforces MAX_UPLOAD_BYTES). The
    boundary is browser-chosen to not collide with the payload, per RFC 7578.
    """
    delim = b"--" + boundary
    parts = []
    for seg in body.split(delim):
        if not seg or seg in (b"--", b"--\r\n", b"\r\n"):
            continue
        if seg.startswith(b"--"):          # trailing closing boundary
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


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
SLUG_RE = re.compile(r"^/f/([^/]+)/?$")
JOB_ACTION_RE = re.compile(r"^/api/jobs/([A-Za-z0-9]+)/(stop|cancel|remove)$")
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
COPY_CHUNK = 256 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "audio-extract-web/1.0"
    protocol_version = "HTTP/1.1"  # keep-alive; needed for smooth seeking

    # -- small response helpers --------------------------------------------
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
        self._send_headers(
            status,
            "application/json; charset=utf-8",
            len(body),
            {"Cache-Control": "no-cache"},
        )
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_text(self, text, status=200, ctype="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self._send_headers(status, ctype, len(body), {"Cache-Control": "no-cache"})
        if self.command != "HEAD":
            self.wfile.write(body)

    def _serve_static_file(self, disk_path, cache="no-cache", download_name=None):
        """Serve a file from disk with HTTP Range support (206) for seeking."""
        if not os.path.isfile(disk_path):
            self._send_text("Not found", 404)
            return
        try:
            size = os.path.getsize(disk_path)
        except OSError:
            self._send_text("Not found", 404)
            return

        ctype, _ = mimetypes.guess_type(disk_path)
        ctype = ctype or "application/octet-stream"
        extra = {"Accept-Ranges": "bytes", "Cache-Control": cache}
        if download_name:
            safe = download_name.replace('"', "")
            extra["Content-Disposition"] = 'attachment; filename="%s"' % safe

        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if range_header:
            m = RANGE_RE.match(range_header.strip())
            if m:
                g1, g2 = m.group(1), m.group(2)
                if g1 == "" and g2 == "":
                    pass  # malformed -> full body
                elif g1 == "":  # suffix range: last N bytes
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
                        # Unsatisfiable range.
                        self._send_headers(
                            416,
                            ctype,
                            0,
                            {"Content-Range": "bytes */%d" % size,
                             "Accept-Ranges": "bytes"},
                        )
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
            pass  # client aborted (common when seeking) -- ignore quietly

    def _safe_lib_path(self, rel_url_path):
        """Map a /lib/... url path to a disk path inside LIB_DIR (no traversal)."""
        rel = urllib.parse.unquote(rel_url_path)
        rel = posixpath.normpath(rel).lstrip("/")
        disk = os.path.abspath(os.path.join(LIB_DIR, rel))
        if disk != LIB_DIR and not disk.startswith(LIB_DIR + os.sep):
            return None
        return disk

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

    # -- routing ------------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._serve_frontend("listing.html", "text/html; charset=utf-8")
                return
            if path == "/healthz":
                self._send_text("ok")
                return
            if path == "/favicon.ico":
                self._send_headers(204, "image/x-icon", 0)
                return

            if path == "/api/index":
                self._send_json(build_index_payload())
                return

            if path == "/api/jobs":
                self._send_json(self.server.job_manager.snapshot())
                return

            if path == "/jobs" or path == "/jobs.html":
                self._serve_frontend("jobs.html", "text/html; charset=utf-8")
                return

            if path.startswith("/api/file/"):
                slug = urllib.parse.unquote(path[len("/api/file/"):]).strip("/")
                payload = build_file_payload(slug)
                if payload is None:
                    self._send_json({"error": "not found", "slug": slug}, 404)
                else:
                    self._send_json(payload)
                return

            m = SLUG_RE.match(path)
            if m:
                # Explorer shell; the slug is read client-side from the URL.
                self._serve_frontend("explorer.html", "text/html; charset=utf-8")
                return

            if path.startswith("/static/"):
                name = posixpath.normpath(path[len("/static/"):]).lstrip("/")
                ctype, _ = mimetypes.guess_type(name)
                self._serve_frontend(name, ctype or "application/octet-stream")
                return

            if path.startswith("/lib/"):
                disk = self._safe_lib_path(path[len("/lib/"):])
                if disk is None:
                    self._send_text("Forbidden", 403)
                    return
                dl = query.get("dl", [None])[0]
                # Media is immutable once written; allow modest caching. JSON
                # (peaks / manifests) may be rewritten, so keep it fresh.
                cache = "no-cache" if disk.endswith(".json") else "public, max-age=3600"
                self._serve_static_file(disk, cache=cache, download_name=dl)
                return

            self._send_text("Not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # never let one bad request kill the thread
            try:
                self._send_text("Server error: %s" % exc, 500)
            except Exception:
                pass

    # -- POST routing (uploads + job actions) ------------------------------
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/upload":
                self._handle_upload()
                return
            m = JOB_ACTION_RE.match(path)
            if m:
                self._drain_body()  # actions carry no body, but be tidy
                job_id, action = m.group(1), m.group(2)
                ok, msg = self.server.job_manager.action(job_id, action)
                self._send_json({"ok": ok, "message": msg}, 200 if ok else 404)
                return
            self._send_text("Not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self._send_json({"ok": False, "error": str(exc)}, 500)
            except Exception:
                pass

    def _read_body(self, max_bytes):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b"", 0
        if length > max_bytes:
            return None, length
        buf = bytearray()
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(COPY_CHUNK, remaining))
            if not chunk:
                break
            buf += chunk
            remaining -= len(chunk)
        return bytes(buf), length

    def _drain_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        while length > 0:
            chunk = self.rfile.read(min(COPY_CHUNK, length))
            if not chunk:
                break
            length -= len(chunk)

    def _handle_upload(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            self._send_json({"ok": False, "error": "expected multipart/form-data"}, 400)
            return
        bm = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', ctype)
        if not bm:
            self._send_json({"ok": False, "error": "missing multipart boundary"}, 400)
            return
        boundary = (bm.group(1) or bm.group(2)).strip().encode("utf-8")

        body, length = self._read_body(MAX_UPLOAD_BYTES)
        if body is None:
            self._send_json(
                {"ok": False,
                 "error": "upload too large (%d bytes; limit %d MB)"
                          % (length, MAX_UPLOAD_BYTES // (1024 * 1024))},
                413)
            return
        if not body:
            self._send_json({"ok": False, "error": "empty request body"}, 400)
            return

        os.makedirs(INPUT_DIR, exist_ok=True)
        mgr = self.server.job_manager
        accepted, rejected = [], []
        for part in parse_multipart_formdata(body, boundary):
            fn = part.get("filename")
            if not fn:
                continue  # a plain form field, not a file
            ext = os.path.splitext(fn)[1].lower()
            if ext not in ALLOWED_UPLOAD_EXT:
                rejected.append({"filename": fn, "reason": "only .mp3/.wav accepted"})
                continue
            data = part.get("data") or b""
            if not data:
                rejected.append({"filename": fn, "reason": "empty file"})
                continue
            dest = unique_path(INPUT_DIR, sanitize_filename(fn))
            try:
                with open(dest, "wb") as fh:
                    fh.write(data)
            except OSError as exc:
                rejected.append({"filename": fn, "reason": "write failed: %s" % exc})
                continue
            saved = os.path.basename(dest)
            job_id = mgr.enqueue(saved, os.path.abspath(dest))
            accepted.append({"filename": saved, "job_id": job_id, "bytes": len(data)})

        status = 200 if accepted else (400 if rejected else 200)
        self._send_json({"ok": bool(accepted), "accepted": accepted,
                         "rejected": rejected}, status)

    def log_message(self, fmt, *args):  # quieter, single-line logs
        sys.stderr.write(
            "%s - %s\n" % (self.address_string(), fmt % args)
        )


class Server(ThreadingHTTPServer):
    # ThreadingHTTPServer already mixes in ThreadingMixIn; just tune it.
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # A client (or the tailscale proxy) hanging up mid-response raises
        # BrokenPipe/ConnectionReset from the socket machinery outside our
        # handlers -- benign, so don't spam the log with a traceback.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main():
    ap = argparse.ArgumentParser(description="audio-extract read-only web GUI")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", DEFAULT_PORT)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    args = ap.parse_args()

    if not os.path.isdir(LIB_DIR):
        sys.stderr.write("WARNING: lib dir not found at %s (will appear later)\n"
                         % LIB_DIR)

    os.makedirs(INPUT_DIR, exist_ok=True)
    httpd = Server((args.host, args.port), Handler)
    # The job runner lives on the server so request handlers can reach it.
    httpd.job_manager = JobManager(ROOT_DIR, INPUT_DIR, JOBS_FILE)
    sys.stderr.write(
        "audio-extract web GUI serving lib=%s on http://%s:%d\n"
        "  uploads -> %s | jobs -> %s\n"
        % (LIB_DIR, args.host, args.port, INPUT_DIR, JOBS_FILE)
    )

    # Turn SIGTERM (run.sh stop/restart) into a clean shutdown so a running
    # conversion's process group is killed rather than orphaned.
    def _term(_signum, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _term)

    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        sys.stderr.write("\nshutting down\n")
    finally:
        httpd.job_manager.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
