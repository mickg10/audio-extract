/* audio-extract v2 GUI — passages timeline + A/B/C compare.
   Depends on common.js (fetchJSON, escapeHtml, fmtTime, clamp). Fully offline. */

/* ------------------------------ tag colors ------------------------------ */
const TAG_COLORS = {
  high_soprano:       "#ff6b9d",
  extreme_soprano:    "#ff3b6b",
  dense_accompaniment:"#ffb454",
  quiet_backing:      "#7ee0c8",
  hall_tail:          "#b48cff",
  no_vocal_control:   "#8a94a6",
  random_control:     "#5b6472",
  vocal_overlap:      "#4aa8ff",
};
function tagColor(tag) {
  if (TAG_COLORS[tag]) return TAG_COLORS[tag];
  // stable hashed hue for any unforeseen tag
  let h = 0;
  for (let i = 0; i < String(tag).length; i++) h = (h * 31 + tag.charCodeAt(i)) & 0xffff;
  return "hsl(" + (h % 360) + ",60%,62%)";
}
function fmtCost(v) {
  if (v == null || !isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e4) return v.toExponential(2);
  return v.toFixed(a < 1 ? 3 : 2);
}

const SLOTS = ["A", "B", "C"];

/* --------------------------------- state -------------------------------- */
const S = {
  runs: [],
  run: null,             // full run payload
  tid: null,
  selPassage: null,      // selected passage object, or null (= whole track)
  timeline: null,
  srcAudio: null,
  slots: { A: null, B: null, C: null },   // recipe_dir per slot
  abAudios: { A: null, B: null, C: null }, // HTMLAudioElement per slot
  active: null,          // slot currently unmuted during audition
  auditioning: false,
};

/* =======================================================================
 * PassageTimeline — peaks + colored passage regions + playhead on a canvas.
 * "Zoom" is windowing: the view [viewStart,viewEnd] maps across the full width
 * (no horizontal scroll — friendlier on phones). Regions are DOM overlays so
 * their labels stay crisp and tappable.
 * ===================================================================== */
class PassageTimeline {
  constructor(container, opts) {
    this.c = container;
    this.onSeek = opts.onSeek || null;
    this.onSelect = opts.onSelect || null;
    this.duration = 0;
    this.viewStart = 0;
    this.viewEnd = 0;
    this.peaks = null;
    this.passages = [];
    this.selectedId = null;
    this.playT = 0;
    this.dpr = Math.max(1, window.devicePixelRatio || 1);

    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.c.appendChild(this.canvas);

    this.playhead = document.createElement("div");
    this.playhead.className = "playhead";
    this.playhead.style.display = "none";
    this.c.appendChild(this.playhead);

    this._regionEls = [];
    this._bindSeek();
    this._ro = new ResizeObserver(() => { this._draw(); this._layoutRegions(); this._positionPlayhead(); });
    this._ro.observe(this.c);
  }

  setDuration(d) { this.duration = d || 0; if (!this.viewEnd) this.viewEnd = this.duration; }
  setPassages(list) { this.passages = list || []; this._layoutRegions(); }
  setSelected(id) { this.selectedId = id; this._layoutRegions(); }

  async setView(start, end, peaksBaseUrl) {
    this.viewStart = start;
    this.viewEnd = end;
    const span = Math.max(1e-6, end - start);
    let url = peaksBaseUrl;
    // request window-specific peaks unless we're viewing the whole track
    if (!(start <= 0.001 && end >= this.duration - 0.001) && this.duration && S.run) {
      const sr = S.run.sample_rate_hz || 44100;
      url = peaksBaseUrl + "?start=" + Math.floor(start * sr) + "&end=" + Math.ceil(end * sr);
    }
    try {
      const data = await fetchJSON(url);
      this.peaks = (data && Array.isArray(data.min)) ? data : null;
    } catch (e) { this.peaks = null; }
    this._draw();
    this._layoutRegions();
    this._positionPlayhead();
  }

  setPlayhead(t) { this.playT = t || 0; this._positionPlayhead(); }

  _span() { return Math.max(1e-6, this.viewEnd - this.viewStart); }
  _xForTime(t) {
    const cssW = this.c.clientWidth || 300;
    return clamp((t - this.viewStart) / this._span(), 0, 1) * cssW;
  }

  _draw() {
    const cssW = Math.max(120, this.c.clientWidth || 300);
    const cssH = this.c.clientHeight || 150;
    const dpr = this.dpr, ctx = this.ctx;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.canvas.style.width = cssW + "px";
    this.canvas.style.height = cssH + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const mid = cssH / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid + 0.5); ctx.lineTo(cssW, mid + 0.5); ctx.stroke();

    if (!this.peaks) return;
    const mins = this.peaks.min, maxs = this.peaks.max;
    const n = Math.min(mins.length, maxs.length);
    if (!n) return;
    ctx.strokeStyle = getCSS("--wave") || "#4aa8ff";
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
  }

  _layoutRegions() {
    for (const el of this._regionEls) el.remove();
    this._regionEls = [];
    const cssW = this.c.clientWidth || 300;
    for (const p of this.passages) {
      if (p.end_s <= this.viewStart || p.start_s >= this.viewEnd) continue; // out of view
      const x0 = this._xForTime(p.start_s);
      const x1 = this._xForTime(p.end_s);
      const w = Math.max(3, x1 - x0);
      const col = tagColor(p.primary_tag);
      const el = document.createElement("div");
      el.className = "pregion" + (p.passage_id === this.selectedId ? " selected" : "");
      el.style.left = x0 + "px";
      el.style.width = w + "px";
      el.style.background = col + "22";
      el.style.borderLeftColor = col;
      el.style.borderRightColor = col;
      el.title = p.primary_tag + " — " + fmtTime(p.start_s) + "–" + fmtTime(p.end_s) +
        (p.tags && p.tags.length > 1 ? "  [" + p.tags.join(", ") + "]" : "");
      const lab = document.createElement("div");
      lab.className = "plabel";
      lab.style.background = col;
      lab.textContent = p.primary_tag;
      el.appendChild(lab);
      el.addEventListener("click", (e) => { e.stopPropagation(); if (this.onSelect) this.onSelect(p); });
      this.c.appendChild(el);
      this._regionEls.push(el);
    }
  }

  _positionPlayhead() {
    if (!this.duration || this.playT < this.viewStart || this.playT > this.viewEnd) {
      this.playhead.style.display = "none"; return;
    }
    this.playhead.style.display = "block";
    this.playhead.style.left = this._xForTime(this.playT) + "px";
  }

  _bindSeek() {
    let downX = 0, downY = 0, moved = false;
    const cvs = this.canvas;
    cvs.addEventListener("pointerdown", (e) => { downX = e.clientX; downY = e.clientY; moved = false; });
    cvs.addEventListener("pointermove", (e) => {
      if (Math.abs(e.clientX - downX) > 8 || Math.abs(e.clientY - downY) > 8) moved = true;
    });
    cvs.addEventListener("pointerup", (e) => {
      if (moved || !this.onSeek) return;
      const rect = cvs.getBoundingClientRect();
      const t = this.viewStart + clamp((e.clientX - rect.left) / rect.width, 0, 1) * this._span();
      this.onSeek(t);
    });
  }
}

/* ------------------------------- routing -------------------------------- */
function runFromUrl() {
  const m = location.pathname.match(/^\/r\/([^/]+)/);
  if (m) return decodeURIComponent(m[1]);
  return new URLSearchParams(location.search).get("run");
}

/* ------------------------------- bootstrap ------------------------------ */
async function init() {
  document.getElementById("refreshBtn").addEventListener("click", () => {
    if (S.tid) loadRun(S.tid); else loadRuns();
  });
  document.getElementById("runPick").addEventListener("change", (e) => {
    const tid = e.target.value;
    if (tid) navRun(tid);
  });
  document.getElementById("fitBtn").addEventListener("click", () => selectPassage(null));
  document.getElementById("srcPlay").addEventListener("click", toggleSource);
  for (const s of SLOTS) {
    document.getElementById("pick" + s).addEventListener("change", (e) => setSlot(s, e.target.value || null));
    document.getElementById("solo" + s).addEventListener("click", () => solo(s));
  }
  document.getElementById("abcStop").addEventListener("click", stopAudition);
  window.addEventListener("popstate", () => routeOrPick());
  document.addEventListener("keydown", onKey);

  await loadRuns();
  routeOrPick();
}

async function loadRuns() {
  try {
    const data = await fetchJSON("/api/v2/runs");
    S.runs = data.runs || [];
    S.ffmpeg = data.ffmpeg;
  } catch (e) { S.runs = []; }
  const sel = document.getElementById("runPick");
  sel.innerHTML = '<option value="">— pick a run —</option>' +
    S.runs.map((r) => '<option value="' + escapeHtml(r.track_id) + '">' + escapeHtml(r.track_id) + "</option>").join("");
}

function routeOrPick() {
  const tid = runFromUrl();
  if (tid) { loadRun(tid); return; }
  if (S.runs.length === 1) { navRun(S.runs[0].track_id, true); return; }
  renderRunPicker();
}

function navRun(tid, replace) {
  const url = "/r/" + encodeURIComponent(tid);
  if (replace) history.replaceState({}, "", url); else history.pushState({}, "", url);
  loadRun(tid);
}

function renderRunPicker() {
  document.getElementById("runView").classList.add("hidden");
  const el = document.getElementById("runPicker");
  el.classList.remove("hidden");
  if (!S.runs.length) {
    el.innerHTML = '<div class="notice">No runs found under this <code>--lib</code> root yet. ' +
      'Ingest one with <code>uv run audio-extract --lib &lt;root&gt; ingest &lt;file&gt; --run-id &lt;id&gt;</code>.</div>';
    return;
  }
  el.innerHTML = '<h2 style="margin:6px 0 14px;font-size:18px;">Runs</h2><div class="run-grid">' +
    S.runs.map((r) => {
      const dur = r.duration_s ? fmtTime(r.duration_s) : "—";
      return '<a class="run-card" href="/r/' + encodeURIComponent(r.track_id) + '">' +
        '<div class="rc-title">' + escapeHtml(r.title || r.track_id) + "</div>" +
        '<div class="rc-meta">' + dur + " · " + (r.sample_rate_hz || "?") + " Hz · " +
        ((r.channels || []).length || "?") + "ch</div>" +
        '<div class="rc-badges">' +
        '<span class="pill">' + (r.n_passages || 0) + " passages</span>" +
        '<span class="pill">' + (r.n_candidates || 0) + " candidates</span>" +
        (r.state ? '<span class="pill ready">' + escapeHtml(r.state) + "</span>" : "") +
        "</div></a>";
    }).join("") + "</div>";
  // intercept clicks for SPA nav
  el.querySelectorAll("a.run-card").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault(); navRun(decodeURIComponent(a.getAttribute("href").split("/r/")[1]));
  }));
}

/* -------------------------------- load run ------------------------------ */
async function loadRun(tid) {
  S.tid = tid;
  stopAudition();
  document.getElementById("runPicker").classList.add("hidden");
  document.getElementById("runView").classList.remove("hidden");
  document.getElementById("trackTitle").textContent = tid;
  document.getElementById("runPick").value = tid;
  let run;
  try {
    run = await fetchJSON("/api/v2/run/" + encodeURIComponent(tid));
  } catch (e) {
    document.getElementById("trackSub").textContent = "failed to load run: " + e.message;
    return;
  }
  if (run.error) { document.getElementById("trackSub").textContent = run.error; return; }
  S.run = run;
  S.selPassage = null;
  S.slots = { A: null, B: null, C: null };

  renderHeader();
  renderState();
  renderLegend();
  renderChips();
  renderCandTable();
  renderSlotPickers();
  renderABGrid();
  renderCostTable();

  // timeline (clear any prior run's canvas/regions before rebuilding)
  const tlEl = document.getElementById("timeline");
  if (S.timeline && S.timeline._ro) { try { S.timeline._ro.disconnect(); } catch (e) {} }
  tlEl.innerHTML = "";
  S.timeline = new PassageTimeline(tlEl, { onSeek: seekSource, onSelect: (p) => selectPassage(p) });
  S.timeline.setDuration(run.duration_s || 0);
  S.timeline.setPassages(run.passages || []);
  await S.timeline.setView(0, run.duration_s || 0, run.source_peaks_url);

  // source audio
  setupSourceAudio();

  // default: auto-load up to 3 candidates (Pareto first — payload is pre-sorted)
  const cands = run.candidates || [];
  SLOTS.forEach((s, i) => { if (cands[i]) setSlot(s, cands[i].dir); });

  selectPassage(null);   // whole-track view + labels
}

function renderHeader() {
  const r = S.run;
  document.getElementById("trackTitle").textContent = r.title || r.track_id;
  const ch = (r.channels || []).join("/") || "?";
  const bits = [
    r.duration_s ? fmtTime(r.duration_s) : "—",
    (r.sample_rate_hz || "?") + " Hz",
    ch,
    (r.passages || []).length + " passages",
    (r.candidates || []).length + " candidates",
  ];
  document.getElementById("trackSub").textContent = bits.join("  ·  ") + "  ·  " + r.track_id;
  if (!S.ffmpeg) {
    document.getElementById("passageHint").innerHTML +=
      ' <span style="color:var(--danger)">ffmpeg not found on the server — audio playback disabled.</span>';
  }
}

function renderState() {
  const el = document.getElementById("runState");
  if (S.run && S.run.state) {
    el.style.display = "";
    el.textContent = S.run.state;
    el.className = "pill ready";
  } else { el.style.display = "none"; }
}

function renderLegend() {
  const tags = new Set();
  (S.run.passages || []).forEach((p) => tags.add(p.primary_tag));
  const el = document.getElementById("tagLegend");
  if (!tags.size) { el.innerHTML = '<span class="muted">no passages mined for this run</span>'; return; }
  el.innerHTML = Array.from(tags).map((t) =>
    '<span class="lg-item"><span class="sw" style="background:' + tagColor(t) + '"></span>' +
    escapeHtml(t) + "</span>").join("");
}

function renderChips() {
  const el = document.getElementById("passageChips");
  const ps = S.run.passages || [];
  let html = '<span class="chip' + (S.selPassage ? "" : " selected") + '" data-idx="-1">' +
    '<span class="dot" style="background:var(--text-mute)"></span>whole track</span>';
  html += ps.map((p, i) =>
    '<span class="chip" data-idx="' + i + '"><span class="dot" style="background:' + tagColor(p.primary_tag) + '"></span>' +
    escapeHtml(p.primary_tag) + ' <span class="ctime">' + fmtTime(p.start_s) + "–" + fmtTime(p.end_s) + "</span></span>"
  ).join("");
  el.innerHTML = html;
  el.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
    const idx = parseInt(c.getAttribute("data-idx"), 10);
    selectPassage(idx < 0 ? null : ps[idx]);
  }));
}

/* ------------------------ candidate overview table ---------------------- */
function renderCandTable() {
  const r = S.run, axes = r.axes, meta = r.axis_meta;
  document.getElementById("candCount").textContent = (r.candidates || []).length + " rendered";
  const nPar = (r.pareto_frontier || []).length;
  document.getElementById("paretoNote").innerHTML = nPar
    ? nPar + " on the Pareto frontier"
    : '<span class="muted">run <code>qa score</code> to populate costs</span>';

  // best (min) per axis across candidates that have costs
  const best = {};
  for (const a of axes) {
    let m = Infinity;
    for (const c of r.candidates) if (c.has_costs && c.costs[a] < m) m = c.costs[a];
    best[a] = m;
  }
  let head = "<thead><tr><th>cand</th><th>model · target · construction</th>" +
    axes.map((a) => '<th class="numc" title="' + escapeHtml(meta[a].hint) + '">' +
      escapeHtml(meta[a].label) + (meta[a].unit ? ' <span class="axis-hint">' + meta[a].unit + "</span>" : "") +
      "</th>").join("") + "<th>A/B/C</th></tr></thead>";
  let body = "<tbody>" + r.candidates.map((c) => {
    const par = c.pareto ? '<span class="pareto-pill">Pareto</span>' : '<span class="dom-pill">#' + ((c.rank ?? 0) + 1) + "</span>";
    const cells = axes.map((a) => {
      const v = c.costs[a];
      const isBest = c.has_costs && isFinite(best[a]) && Math.abs(v - best[a]) < 1e-12;
      return '<td class="numc' + (isBest ? " best" : "") + '">' + fmtCost(v) + "</td>";
    }).join("");
    const btns = SLOTS.map((s) =>
      '<button class="mini-btn slot-btn" data-slot="' + s + '" data-dir="' + c.dir + '">' + s + "</button>").join(" ");
    return "<tr data-dir=\"" + c.dir + "\"><td>" + par + "</td>" +
      '<td class="lbl">' + escapeHtml(c.label) + '<div class="cand-sub" style="font-family:var(--mono);color:var(--text-mute);font-size:11px;">' +
      escapeHtml(c.short_id) + "</div></td>" + cells + "<td>" + btns + "</td></tr>";
  }).join("") + "</tbody>";
  const t = document.getElementById("candTable");
  t.innerHTML = head + body;
  t.querySelectorAll(".slot-btn").forEach((b) => b.addEventListener("click", () => {
    const s = b.getAttribute("data-slot"), dir = b.getAttribute("data-dir");
    setSlot(s, S.slots[s] === dir ? null : dir);  // toggle
  }));
  refreshSlotButtons();
}

function refreshSlotButtons() {
  document.querySelectorAll("#candTable .slot-btn").forEach((b) => {
    const s = b.getAttribute("data-slot"), dir = b.getAttribute("data-dir");
    b.classList.toggle("on", S.slots[s] === dir);
  });
}

/* ------------------------------ slot pickers ---------------------------- */
function candByDir(dir) { return (S.run.candidates || []).find((c) => c.dir === dir) || null; }

function renderSlotPickers() {
  const opts = '<option value="">— none —</option>' + (S.run.candidates || []).map((c) =>
    '<option value="' + c.dir + '">' + (c.pareto ? "★ " : "") + escapeHtml(c.label) + " · " + escapeHtml(c.short_id) + "</option>"
  ).join("");
  for (const s of SLOTS) {
    const sel = document.getElementById("pick" + s);
    sel.innerHTML = opts;
    sel.value = S.slots[s] || "";
  }
}

function setSlot(slot, dir) {
  S.slots[slot] = dir;
  document.getElementById("pick" + slot).value = dir || "";
  // (re)build the audio element for this slot
  const a = S.abAudios[slot];
  if (a) { try { a.pause(); } catch (e) {} }
  if (dir) {
    const c = candByDir(dir);
    const au = new Audio();
    au.preload = "none";
    au.src = c.audio_url;         // whole-track AAC; audition seeks within it
    au.addEventListener("timeupdate", () => onAbTime(slot));
    S.abAudios[slot] = au;
  } else {
    S.abAudios[slot] = null;
  }
  refreshSlotButtons();
  renderABGrid();
  renderCostTable();
}

/* ------------------------------- A/B/C grid ----------------------------- */
function passWindowQS() {
  if (!S.selPassage || !S.run) return "";
  const sr = S.run.sample_rate_hz || 44100;
  return "?start=" + Math.floor(S.selPassage.start_s * sr) + "&end=" + Math.ceil(S.selPassage.end_s * sr);
}

function renderABGrid() {
  const grid = document.getElementById("abGrid");
  const qs = passWindowQS();
  grid.innerHTML = SLOTS.map((s) => {
    const dir = S.slots[s];
    if (!dir) return '<div class="ab-col empty"><span class="slot-letter">' + s + "</span><br><br>empty slot<br><span class=\"muted\">pick a candidate</span></div>";
    const c = candByDir(dir);
    const par = c.pareto ? ' <span class="pareto-pill">Pareto</span>' : "";
    const costs = c.has_costs
      ? '<div class="col-costs">' + S.run.axes.map((a) =>
          '<div class="cc"><span>' + escapeHtml(S.run.axis_meta[a].label) + "</span><b>" + fmtCost(c.costs[a]) + "</b></div>").join("") + "</div>"
      : '<div class="col-costs muted">no costs — run qa score</div>';
    return '<div class="ab-col" data-slot="' + s + '">' +
      '<div><span class="slot-letter">' + s + '</span><span class="cand-label">' + escapeHtml(c.label) + par + "</span>" +
      '<div class="cand-sub">' + escapeHtml(c.short_id) + "</div></div>" +
      '<audio controls preload="none" src="' + c.audio_url + '"></audio>' +
      '<div class="spec"><img loading="lazy" alt="spectrogram" src="' + c.spectrogram_url + qs + '"></div>' +
      costs + "</div>";
  }).join("");
}

/* --------------------------- combined cost table ------------------------ */
function renderCostTable() {
  const r = S.run, axes = r.axes, meta = r.axis_meta;
  const chosen = SLOTS.filter((s) => S.slots[s]).map((s) => ({ s, c: candByDir(S.slots[s]) }));
  const t = document.getElementById("costTable");
  if (!chosen.length) { t.innerHTML = '<tbody><tr><td class="muted">pick candidates above to compare their costs</td></tr></tbody>'; return; }
  let head = "<thead><tr><th>axis</th>" + chosen.map(({ s, c }) =>
    '<th class="numc">' + s + (c.pareto ? ' <span class="pareto-pill">P</span>' : "") +
    '<div class="axis-hint" style="text-align:right">' + escapeHtml(c.short_id) + "</div></th>").join("") + "</tr></thead>";
  let body = "<tbody>" + axes.map((a) => {
    const vals = chosen.map(({ c }) => (c.has_costs ? c.costs[a] : null));
    const finite = vals.filter((v) => v != null && isFinite(v));
    const best = finite.length ? Math.min(...finite) : null;
    return "<tr><td>" + escapeHtml(meta[a].label) +
      (meta[a].unit ? ' <span class="axis-hint">(' + meta[a].unit + ")</span>" : "") + "</td>" +
      vals.map((v) => '<td class="numc' + (best != null && v != null && Math.abs(v - best) < 1e-12 ? " best" : "") + '">' +
        fmtCost(v) + "</td>").join("") + "</tr>";
  }).join("") + "</tbody>";
  t.innerHTML = head + body;
}

/* --------------------------- passage selection -------------------------- */
function selectPassage(p) {
  S.selPassage = p;
  const label = p ? (p.primary_tag + "  " + fmtTime(p.start_s) + "–" + fmtTime(p.end_s)) : "whole track";
  document.getElementById("selPassage").textContent = p ? ("selected: " + label) : "whole track";
  document.getElementById("abcPassage").textContent = label;
  // chips + region highlight
  document.querySelectorAll("#passageChips .chip").forEach((c) => {
    const idx = parseInt(c.getAttribute("data-idx"), 10);
    const on = (idx < 0 && !p) || (p && (S.run.passages[idx] === p));
    c.classList.toggle("selected", !!on);
  });
  if (S.timeline) {
    S.timeline.setSelected(p ? p.passage_id : null);
    const start = p ? p.start_s : 0;
    const end = p ? p.end_s : (S.run.duration_s || 0);
    // small padding around a zoomed passage for context
    const pad = p ? Math.min(2.0, (end - start) * 0.15) : 0;
    S.timeline.setView(Math.max(0, start - pad), Math.min(S.run.duration_s || end, end + pad), S.run.source_peaks_url);
  }
  // source spectrogram (windowed) + A/B/C spectrograms (windowed)
  const qs = passWindowQS();
  document.getElementById("srcSpec").src = S.run.source_spectrogram_url + qs;
  renderABGrid();
  if (S.auditioning) stopAudition();
}

/* ----------------------------- source audio ----------------------------- */
function setupSourceAudio() {
  if (S.srcAudio) { try { S.srcAudio.pause(); } catch (e) {} }
  const au = new Audio();
  au.preload = "none";
  au.src = S.run.source_audio_url;
  au.addEventListener("timeupdate", () => {
    document.getElementById("srcTime").textContent = fmtTime(au.currentTime) + " / " + fmtTime(S.run.duration_s || au.duration);
    if (S.timeline && !S.auditioning) S.timeline.setPlayhead(au.currentTime);
  });
  document.getElementById("srcTime").textContent = "0:00 / " + fmtTime(S.run.duration_s || 0);
  au.addEventListener("ended", () => { document.getElementById("srcPlay").textContent = "▶ Source"; });
  au.addEventListener("play", () => { document.getElementById("srcPlay").textContent = "❚❚ Source"; });
  au.addEventListener("pause", () => { document.getElementById("srcPlay").textContent = "▶ Source"; });
  S.srcAudio = au;
}
function toggleSource() {
  if (!S.srcAudio) return;
  if (S.auditioning) stopAudition();
  if (S.srcAudio.paused) S.srcAudio.play().catch(() => {}); else S.srcAudio.pause();
}
function seekSource(t) {
  if (S.srcAudio) { S.srcAudio.currentTime = t; if (S.timeline) S.timeline.setPlayhead(t); }
}

/* ------------------------------- audition ------------------------------- */
function passBounds() {
  if (S.selPassage) return [S.selPassage.start_s, S.selPassage.end_s];
  return [0, S.run.duration_s || 0];
}
function solo(slot) {
  if (!S.slots[slot]) return;
  if (S.srcAudio) S.srcAudio.pause();
  const [p0] = passBounds();
  if (!S.auditioning) {
    // start all loaded slots together at the passage start; unmute only `slot`
    S.auditioning = true;
    for (const s of SLOTS) {
      const a = S.abAudios[s];
      if (!a) continue;
      try { a.currentTime = p0; } catch (e) {}
      a.muted = (s !== slot);
      a.play().catch(() => {});
    }
  } else {
    // seamless switch: realign the others to the active one, then move the solo
    const cur = S.abAudios[S.active] ? S.abAudios[S.active].currentTime : p0;
    for (const s of SLOTS) {
      const a = S.abAudios[s];
      if (!a) continue;
      if (Math.abs(a.currentTime - cur) > 0.05) { try { a.currentTime = cur; } catch (e) {} }
      a.muted = (s !== slot);
      if (a.paused) a.play().catch(() => {});
    }
  }
  S.active = slot;
  updateSoloButtons();
}
function onAbTime(slot) {
  if (!S.auditioning || slot !== S.active) return;
  const a = S.abAudios[slot];
  const [p0, p1] = passBounds();
  document.getElementById("abcTime").textContent =
    fmtTime(a.currentTime - p0) + " / " + fmtTime(p1 - p0);
  if (S.timeline) S.timeline.setPlayhead(a.currentTime);
  if (a.currentTime >= p1 - 0.02) {
    if (document.getElementById("abcLoop").checked) {
      for (const s of SLOTS) { const x = S.abAudios[s]; if (x) { try { x.currentTime = p0; } catch (e) {} } }
    } else { stopAudition(); }
  }
}
function stopAudition() {
  S.auditioning = false; S.active = null;
  for (const s of SLOTS) { const a = S.abAudios[s]; if (a) { try { a.pause(); } catch (e) {} } }
  updateSoloButtons();
}
function updateSoloButtons() {
  for (const s of SLOTS) {
    const b = document.getElementById("solo" + s);
    b.disabled = !S.slots[s];
    b.classList.toggle("playing", S.auditioning && S.active === s);
  }
}

/* ------------------------------- keyboard ------------------------------- */
function onKey(e) {
  if (e.target && /input|select|textarea/i.test(e.target.tagName)) return;
  if (e.key === " ") { e.preventDefault(); if (S.auditioning) stopAudition(); else toggleSource(); }
  else if (e.key === "a" || e.key === "A") solo("A");
  else if (e.key === "b" || e.key === "B") solo("B");
  else if (e.key === "c" || e.key === "C") solo("C");
  else if (e.key === "Escape") stopAudition();
}

document.addEventListener("DOMContentLoaded", init);
