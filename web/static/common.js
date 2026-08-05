/* Shared helpers + the canvas Waveform renderer.
   Fully offline, no dependencies. */

/* ----------------------------- formatting ----------------------------- */
function fmtTime(sec) {
  if (sec == null || !isFinite(sec)) return "--:--";
  sec = Math.max(0, sec);
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m + ":" + String(s).padStart(2, "0");
}
function fmtDur(sec) {
  if (sec == null || !isFinite(sec)) return "—";
  if (sec < 1) return sec.toFixed(2) + "s";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? m + "m " + String(s).padStart(2, "0") + "s" : s + "s";
}
function fmtNum(x, digits) {
  if (x == null || !isFinite(x)) return "—";
  return Number(x).toFixed(digits == null ? 2 : digits);
}
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error("HTTP " + r.status + " for " + url);
  return r.json();
}

/* --------------------------------------------------------------------------
 * Waveform
 *   Renders a peaks.json {min[],max[]} envelope onto a canvas that fits its
 *   container by default (responsive / mobile friendly) and can be zoomed to
 *   reveal detail, in which case the container scrolls horizontally and the
 *   playhead auto-scrolls into view. Tap / click to seek (mouse + touch).
 * ------------------------------------------------------------------------ */
class Waveform {
  /**
   * @param {HTMLElement} container  a .wave-scroll element (position:relative, overflow-x:auto)
   * @param {object} opts { duration, color, onSeek(t) }
   */
  constructor(container, opts) {
    opts = opts || {};
    this.container = container;
    this.duration = opts.duration || 0;
    this.color = opts.color || getCSS("--wave") || "#4aa8ff";
    this.onSeek = opts.onSeek || null;
    this.peaks = null;         // {min:[], max:[]}
    this.playT = 0;            // playhead position in seconds
    this.zoomPxPerSec = null;  // null => fit-to-width
    this.dpr = Math.max(1, window.devicePixelRatio || 1);

    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    container.appendChild(this.canvas);

    this.playhead = document.createElement("div");
    this.playhead.className = "playhead";
    this.playhead.style.display = "none";
    container.appendChild(this.playhead);

    this._bindSeek();
    this._ro = new ResizeObserver(() => { if (this.zoomPxPerSec == null) this.render(); });
    this._ro.observe(container);
  }

  async load(url) {
    if (!url) return;
    try {
      const data = await fetchJSON(url);
      if (data && Array.isArray(data.min) && Array.isArray(data.max)) {
        this.peaks = data;
        this.render();
      }
    } catch (e) { /* peaks not generated yet -- leave blank */ }
  }

  setPeaks(data) { this.peaks = data; this.render(); }

  /* Layout width in CSS px: fit container, or duration * zoom. */
  _cssWidth() {
    const fit = Math.max(120, this.container.clientWidth || 300);
    if (this.zoomPxPerSec == null || !this.duration) return fit;
    return Math.max(fit, Math.round(this.duration * this.zoomPxPerSec));
  }

  render() {
    if (!this.peaks) return;
    const cssW = this._cssWidth();
    const cssH = this.container.clientHeight || 96;
    const dpr = this.dpr;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.canvas.style.width = cssW + "px";
    this.canvas.style.height = cssH + "px";

    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // zero line
    const mid = cssH / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid + 0.5); ctx.lineTo(cssW, mid + 0.5); ctx.stroke();

    const mins = this.peaks.min, maxs = this.peaks.max;
    const n = Math.min(mins.length, maxs.length);
    if (!n) return;

    // Map each output pixel column to the range of buckets that fall in it,
    // taking min-of-mins / max-of-maxs so downsampling never loses transients.
    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < cssW; x++) {
      const b0 = Math.floor((x / cssW) * n);
      const b1 = Math.max(b0 + 1, Math.floor(((x + 1) / cssW) * n));
      let lo = 1, hi = -1;
      for (let b = b0; b < b1 && b < n; b++) {
        if (mins[b] < lo) lo = mins[b];
        if (maxs[b] > hi) hi = maxs[b];
      }
      if (hi < lo) { lo = 0; hi = 0; }
      const yTop = mid - hi * (mid - 1);
      const yBot = mid - lo * (mid - 1);
      ctx.moveTo(x + 0.5, yTop);
      ctx.lineTo(x + 0.5, Math.max(yBot, yTop + 0.6));
    }
    ctx.stroke();
    this._positionPlayhead();
  }

  setPlayhead(t) { this.playT = t || 0; this._positionPlayhead(true); }

  _positionPlayhead(autoscroll) {
    if (!this.duration) { this.playhead.style.display = "none"; return; }
    const cssW = parseFloat(this.canvas.style.width) || this.container.clientWidth;
    const x = clamp(this.playT / this.duration, 0, 1) * cssW;
    this.playhead.style.display = "block";
    this.playhead.style.left = x + "px";
    if (autoscroll && this.zoomPxPerSec != null) {
      const view = this.container;
      if (x < view.scrollLeft + 20 || x > view.scrollLeft + view.clientWidth - 20) {
        view.scrollLeft = x - view.clientWidth / 2;
      }
    }
  }

  zoomIn()  { const base = this.zoomPxPerSec || this._fitPxPerSec(); this.zoomPxPerSec = base * 1.8; this.render(); }
  zoomOut() {
    const base = this.zoomPxPerSec || this._fitPxPerSec();
    const next = base / 1.8;
    this.zoomPxPerSec = next <= this._fitPxPerSec() * 1.02 ? null : next;
    this.render();
  }
  zoomFit() { this.zoomPxPerSec = null; this.container.scrollLeft = 0; this.render(); }
  _fitPxPerSec() { return this.duration ? (this.container.clientWidth || 300) / this.duration : 1; }

  _bindSeek() {
    // Pointer events unify mouse + touch. A tap (small movement) seeks; a drag
    // is left to the container's native horizontal scroll.
    let downX = 0, downY = 0, moved = false;
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => { downX = e.clientX; downY = e.clientY; moved = false; });
    c.addEventListener("pointermove", (e) => {
      if (Math.abs(e.clientX - downX) > 8 || Math.abs(e.clientY - downY) > 8) moved = true;
    });
    c.addEventListener("pointerup", (e) => {
      if (moved || !this.onSeek || !this.duration) return;
      const rect = c.getBoundingClientRect();
      const t = clamp((e.clientX - rect.left) / rect.width, 0, 1) * this.duration;
      this.onSeek(t);
    });
  }

  destroy() { try { this._ro.disconnect(); } catch (e) {} }
}

function getCSS(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

function fmtElapsed(sec) {
  if (sec == null || !isFinite(sec)) return "—";
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h > 0 ? h + ":" + mm + ":" + ss : m + ":" + ss;
}

/* POST a job action (stop/cancel/remove) and return the parsed JSON. */
async function postAction(url) {
  const r = await fetch(url, { method: "POST" });
  let body = {};
  try { body = await r.json(); } catch (e) {}
  return { ok: r.ok && body.ok !== false, body };
}

/* ------------------------------------------------------------------------
 * mountUploader: a drop-zone + file picker that uploads .mp3/.wav to
 * /api/upload (multipart) and enqueues a job per file. Shows upload progress.
 *   opts.onComplete(result)  fired after each upload batch.
 * ---------------------------------------------------------------------- */
function mountUploader(container, opts) {
  opts = opts || {};
  container.innerHTML =
    '<div class="uploader" tabindex="0">' +
    '  <div class="uploader-inner">' +
    '    <div class="up-icon">⇪</div>' +
    '    <div class="up-text"><b>Drop .mp3 / .wav to process</b> or ' +
    '      <label class="up-browse">browse<input type="file" accept=".mp3,.wav,audio/mpeg,audio/wav,audio/x-wav" multiple hidden></label></div>' +
    '    <div class="up-hint">each file is queued & processed on demand — separation → ensemble → de-echo → diffs</div>' +
    '  </div>' +
    '  <div class="up-progress hidden"><div class="up-bar"></div></div>' +
    '  <div class="up-status hidden"></div>' +
    "</div>";
  const zone = container.querySelector(".uploader");
  const input = container.querySelector('input[type="file"]');
  const prog = container.querySelector(".up-progress");
  const bar = container.querySelector(".up-bar");
  const status = container.querySelector(".up-status");

  function setStatus(html, cls) {
    status.className = "up-status" + (cls ? " " + cls : "");
    status.innerHTML = html;
    status.classList.remove("hidden");
  }

  function accept(files) {
    const arr = Array.from(files || []);
    const ok = [], bad = [];
    for (const f of arr) {
      if (/\.(mp3|wav)$/i.test(f.name)) ok.push(f); else bad.push(f.name);
    }
    if (!ok.length) {
      setStatus("Only .mp3 / .wav files are accepted." +
        (bad.length ? " Rejected: " + bad.map(escapeHtml).join(", ") : ""), "err");
      return;
    }
    upload(ok, bad);
  }

  function upload(files, badClient) {
    const fd = new FormData();
    for (const f of files) fd.append("files", f, f.name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    prog.classList.remove("hidden");
    bar.style.width = "0%";
    setStatus('<span class="spin"></span> uploading ' + files.length +
      " file" + (files.length === 1 ? "" : "s") + "…");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) bar.style.width = Math.round((e.loaded / e.total) * 100) + "%";
    };
    xhr.onload = () => {
      prog.classList.add("hidden");
      let res = {};
      try { res = JSON.parse(xhr.responseText); } catch (e) {}
      const acc = res.accepted || [], rej = (res.rejected || []).concat(
        (badClient || []).map((n) => ({ filename: n, reason: "not .mp3/.wav" })));
      let msg = "";
      if (acc.length) msg += "Queued <b>" + acc.length + "</b> file" +
        (acc.length === 1 ? "" : "s") + " → <a href=\"/jobs\">view jobs →</a>";
      if (rej.length) msg += (msg ? "<br>" : "") + '<span class="muted">Rejected: ' +
        rej.map((r) => escapeHtml(r.filename) + " (" + escapeHtml(r.reason) + ")").join(", ") + "</span>";
      setStatus(msg || "No files uploaded.", acc.length ? "ok" : "err");
      input.value = "";
      if (opts.onComplete) opts.onComplete(res);
    };
    xhr.onerror = () => {
      prog.classList.add("hidden");
      setStatus("Upload failed (network error).", "err");
    };
    xhr.send(fd);
  }

  input.addEventListener("change", () => accept(input.files));
  zone.addEventListener("click", (e) => { if (e.target === zone || e.target.closest(".uploader-inner") && e.target.tagName !== "INPUT" && !e.target.closest("label")) input.click(); });
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation(); zone.classList.add("drag");
  }));
  ["dragleave", "dragend", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation(); zone.classList.remove("drag");
  }));
  zone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files) accept(e.dataTransfer.files);
  });
}

/* Attach compact zoom controls (- fit +) that drive one or more waveforms. */
function makeZoomControls(waveforms) {
  const wrap = document.createElement("div");
  wrap.className = "wave-zoom";
  const mk = (label, title, fn) => {
    const b = document.createElement("button");
    b.className = "btn small"; b.textContent = label; b.title = title;
    b.addEventListener("click", fn);
    wrap.appendChild(b);
    return b;
  };
  const list = Array.isArray(waveforms) ? waveforms : [waveforms];
  mk("−", "Zoom out", () => list.forEach(w => w && w.zoomOut()));
  mk("Fit", "Fit to width", () => list.forEach(w => w && w.zoomFit()));
  mk("+", "Zoom in", () => list.forEach(w => w && w.zoomIn()));
  return wrap;
}
