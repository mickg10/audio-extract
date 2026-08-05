# audio-extract web GUI

A self-contained web GUI over `../lib/` (the stem-separation library produced by
`../convert.py` — see `../SCHEMA.md`): a **read-only viewer** of the library
plus an **upload + job queue** to process new files on demand. Pure Python
standard-library backend (no third-party deps, so it never disturbs the uv
environment) + plain offline HTML/CSS/JS frontend (no CDN).

## URL / port

- **Tailnet (private) URL:** https://<your-tailnet-node>.ts.net/
  (exposed with `tailscale serve` — tailnet-only, **not** public funnel)
- **Local:** http://127.0.0.1:8730/
- **Port:** `8730` (override with `PORT=... web/run.sh`)
- **LAN fallback:** by default the server binds to loopback only (so only
  Tailscale can reach it). To also serve the LAN, bind all interfaces:
  `HOST=0.0.0.0 web/run.sh restart` → then http://172.25.5.118:8730/

## Start / stop / restart

Run from anywhere; the script cd's to the project root itself.

```bash
web/run.sh            # restart (default): stop any running instance, start detached
web/run.sh start      # start if not running
web/run.sh stop       # stop
web/run.sh status     # is it running?
```

The server is started detached with `nohup`, so it survives your shell. Logs go
to `web/server.log`. Equivalent manual command:

```bash
cd ~/audio-extract && uv run python web/server.py --port 8730
```

### Re-expose on the tailnet (only needed once, or after a reboot)

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg 8730
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve status      # verify
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --https=443 off   # to remove
```

`tailscale serve` proxies `https://<magicdns>/` → `http://127.0.0.1:8730`,
including HTTP Range, so audio seeking works over HTTPS.

## Pages & features

- **`/` — Listing.** Table of every file from `index.json`, **merged with a live
  scan of `lib/`** so files show up (as `processing…`) before `index.json`
  catches up. Columns: title, duration, # variants, status pill, original
  download, explore link. Filter box, manual **Refresh**, and 5s **auto-refresh**
  (data grows as the batch runs). Reflows to stacked cards on phones.
- **`/f/<slug>` — Explorer.**
  - **Variants tab:** one card per `manifest.jsonl` entry, in SCHEMA order
    (original → instrumentals [roformer, mdx23c, ensemble, de-echo/de-reverb] →
    vocals → diffs). Each card: HTML5 `<audio>` (seekable via Range), a
    **canvas waveform** drawn from `peaks.json` with a moving playhead
    (click/tap to seek), a **zoom** control (−/Fit/+; zoomed view scrolls and
    the playhead auto-follows), the **spectrogram** PNG, a **download** link,
    and the metrics (duration, peak, rms, lufs, sr, channels).
  - **Analysis · A/B/C tab:** pick any three variants; one **transport plays /
    pauses / seeks all three in sync**. Each slot has an audible/mute toggle so
    you can A/B by ear without losing sync, plus its spectrogram. Precomputed
    **diff** variants (`residual_voice`, `*_removed`) are offered as one-click
    **presets** (the "subtraction vs reverb-subtraction" comparisons). An
    **A-vs-B overlay** superimposes the two envelopes and draws their |A−B|
    divergence. Deep-linkable at `/f/<slug>#analysis`.
  - **Keyboard:** `space` = play/pause (active player, or the A/B/C transport on
    the Analysis tab); `←` / `→` = seek the active player ±5s.

Everything degrades gracefully while the batch runs: missing audio / spectrogram
/ peaks show a "pending…" placeholder and are filled in on the next refresh; a
file with no `manifest.jsonl` yet still shows its original.

## Upload & processing jobs

- **Upload** — a drop-zone / file-picker on both `/` and `/jobs` accepts
  **`.mp3` / `.wav`** (other types rejected), multiple at once. Files POST to
  `/api/upload` (multipart), are saved to `../input/<sanitized-name>` (collisions
  get `_1`, `_2`, …), and one **job is enqueued per file**. Default size limit
  1024 MB (`MAX_UPLOAD_MB`).
- **Job queue + runner** — a single background worker thread in the server runs
  **one job at a time** (the MPS/GPU is single — never two `convert.py`
  subprocesses at once; the rest queue in order). Each job runs
  `cd .. && uv run python convert.py file <abspath>` and parses its stdout for a
  coarse phase/progress bar (starting → separating → ensembling → variants
  written → normalizing → transcoding → done).
- **`/jobs` page** (linked from the header, with an active-count badge) — a live
  table (polls `/api/jobs` every 2s), newest first: filename, status pill,
  phase + progress bar, elapsed, and per-job actions:
  - **Stop** a *running* job → **process-group kill**. The subprocess is started
    with `start_new_session=True` (its own session/pgid), so `os.killpg`
    (SIGTERM, then SIGKILL after 3s) takes down the `uv`/`python` process **and
    every child it spawned** (model workers, ffmpeg) — not just the top process.
  - **Cancel** a *queued* job → dequeued (marked `canceled`), never runs.
  - **Remove** a *finished* job → cleared from the list.
  When a job finishes, `convert.py` rebuilds `lib/index.json`, so the new file
  appears in the auto-refreshing listing; the done row links to its explorer.
- **Persistence** — jobs are written to `web/jobs.json` and survive a server
  restart; a job that was `running` when the server stopped is marked
  interrupted (`failed`) on reload. `SIGTERM` (from `run.sh stop/restart`) is
  caught so an in-flight conversion's process group is killed, never orphaned.
- **Statuses:** `queued · running · done · failed · stopped · canceled`.
- **Testing seam:** `AUDIO_EXTRACT_JOB_CMD` overrides the command (a template;
  `{path}` is substituted) — used to test the runner without invoking the real
  models. `AUDIO_EXTRACT_INPUT_DIR` / `AUDIO_EXTRACT_JOBS_FILE` relocate state.

## Mobile

Fully responsive (viewport meta, single-column stack on phones, table→cards,
canvas + images scale to width and redraw on resize, touch/pointer tap-to-seek).
Verified at ~390px (iPhone) and desktop.

## Layout

```
web/
  server.py            # stdlib HTTP server: APIs + static + lib assets w/ Range + upload
  jobs.py              # background job queue + runner (process-group kill, persistence)
  run.sh               # start/stop/restart (detached via nohup)
  server.log           # runtime log (created on start)
  jobs.json            # persisted job queue (created on first job)
  static/
    style.css          # dark theme, responsive
    common.js          # helpers + canvas Waveform class + the uploader widget
    listing.html/.js   # the listing page (+ uploader, jobs badge)
    explorer.html/.js  # the per-file explorer + A/B/C analysis
    jobs.html/.js      # the processing-jobs page
```

### Endpoints

| Route | Purpose |
|-------|---------|
| `GET /` | listing page |
| `GET /f/<slug>` | explorer page |
| `GET /jobs` | processing-jobs page |
| `GET /api/index` | listing data (index.json merged with a live dir scan) |
| `GET /api/file/<slug>` | that file's ordered manifest + asset URLs/ready flags |
| `GET /api/jobs` | job queue snapshot (newest first) + active count |
| `POST /api/upload` | multipart `.mp3`/`.wav` upload → saved to `../input/`, enqueues a job |
| `POST /api/jobs/<id>/stop` | kill a running job (process-group) |
| `POST /api/jobs/<id>/cancel` | dequeue a queued job |
| `POST /api/jobs/<id>/remove` | clear a finished job from the list |
| `GET /lib/<path>` | raw assets (audio/png/peaks) with HTTP **Range** (206); `?dl=<name>` forces download |
| `GET /static/<file>` | frontend assets |
| `GET /healthz` | `ok` |
