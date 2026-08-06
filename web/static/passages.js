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
  decision: null,        // /api/v2/decision payload (autonomous audit panel)
  challenges: null,      // /api/v2/challenges payload
  actions: null,         // /api/v2/actions payload
  calibration: null,     // /api/v2/calibration payload (for the theft gate τ)
};

/* Executed-backend chip: audio-separator (CUDA/CPU box) vs separator-ttnn (TT). */
function backendChipHtml(b) {
  if (!b || !b.adapter) return "";
  const cls = b.kind === "tt" ? "tt" : (b.kind === "cuda" ? "cuda" : "");
  const hash = b.bundle_prefix
    ? ' <span class="bh">' + escapeHtml(String(b.bundle_prefix).replace(/^sha256:/, "").slice(0, 8)) + "</span>"
    : "";
  return ' <span class="backend-chip ' + cls + '" title="executed adapter' +
    (b.bundle_prefix ? " · bundle " + escapeHtml(b.bundle_prefix) : "") + '">' +
    escapeHtml(b.adapter) + hash + "</span>";
}
function backendFor(cid) {
  return (S.challenges && S.challenges.backends && S.challenges.backends[cid]) || null;
}

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
  S.decision = S.challenges = S.actions = null;

  renderHeader();
  renderState();
  renderLegend();
  renderChips();
  renderCandTable();
  renderSlotPickers();
  renderABGrid();
  renderCostTable();
  loadDecisionPanel(tid);   // async, non-blocking; read-only audit surface

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
      '<td class="lbl">' + escapeHtml(c.label) + backendChipHtml(c.backend) +
      '<div class="cand-sub" style="font-family:var(--mono);color:var(--text-mute);font-size:11px;">' +
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
      '<div class="cand-sub">' + escapeHtml(c.short_id) + backendChipHtml(c.backend) + "</div></div>" +
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

/* ================= autonomous decision panel (v2.1 WP13) =================
   Read-only audit surface over the data-model-v2 tables: what the autonomous
   selector decided, the per-candidate risk/gate evidence, challenge results
   and the conductor's probe log. Never writes anything; every sub-section
   degrades to a muted note when its table is absent. */

const DEFECT_ABBREV = {
  orchestral_theft: "theft", vocal_leakage: "leakage", event_hole: "holes",
  generic_quality: "generic", brightness: "bright", transients: "trans",
  fullness: "fullness", stereo: "stereo", hall: "hall", severe_artifact: "artifact",
  technical: "tech",
};
function defectAbbrev(d) { return DEFECT_ABBREV[d] || String(d).slice(0, 8); }

function fmtRisk(v) { return (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(2); }

function riskColor(v) {
  if (v == null || !isFinite(v)) return "var(--text-mute)";
  if (v < 0.35) return "var(--accent-2)";
  if (v < 0.65) return "var(--accent-3)";
  return "var(--danger)";
}

/* Map a selector candidate id (recipe_id) onto a rendered candidate, if any. */
function decCandInfo(cid) {
  const cands = (S.run && S.run.candidates) || [];
  const c = cands.find((x) => x.recipe_id === cid || x.dir === cid) || null;
  const short = String(cid || "").replace(/^sha256:/, "").slice(0, 8) || "?";
  return { short, label: c ? c.label : null, cand: c };
}
function decCandName(cid) {
  const i = decCandInfo(cid);
  return (i.label ? escapeHtml(i.label) + " · " : "") +
    '<span class="dec-id">' + escapeHtml(i.short) + "</span>";
}

/* [lower, mean, upper] interval bar on a 0..1 track + acceptable-risk tick. */
function riskBarHtml(lo, mean, hi, ceiling) {
  const pc = (v) => (clamp(v == null ? 0 : v, 0, 1) * 100).toFixed(1) + "%";
  const l = clamp(lo == null ? 0 : lo, 0, 1);
  const h = clamp(hi == null ? l : hi, 0, 1);
  const w = Math.max(0.5, (h - l) * 100).toFixed(1) + "%";
  let html = '<span class="risk-track" title="risk lower/mean/upper: ' +
    fmtRisk(lo) + " / " + fmtRisk(mean) + " / " + fmtRisk(hi) + '">';
  if (ceiling != null && isFinite(ceiling)) {
    html += '<span class="risk-ceiling" style="left:' + pc(ceiling) +
      '" title="acceptable-risk ceiling ' + fmtRisk(ceiling) + '"></span>';
  }
  html += '<span class="risk-band" style="left:' + pc(l) + ";width:" + w +
    ";background:" + riskColor(mean) + '"></span>';
  if (mean != null && isFinite(mean)) {
    html += '<span class="risk-mean" style="left:' + pc(mean) + '"></span>';
  }
  return html + "</span>";
}

function defectBarsHtml(perDefect) {
  const keys = Object.keys(perDefect || {}).sort(
    (a, b) => (perDefect[b] || 0) - (perDefect[a] || 0));
  if (!keys.length) return '<span class="muted">—</span>';
  return '<span class="defect-bars">' + keys.map((d) => {
    const v = clamp(perDefect[d] || 0, 0, 1);
    return '<span class="db-item" title="' + escapeHtml(d) + " risk " + fmtRisk(v) + '">' +
      '<span class="db-name">' + escapeHtml(defectAbbrev(d)) + "</span>" +
      '<span class="db-track"><span class="db-fill" style="width:' + (v * 100).toFixed(0) +
      "%;background:" + riskColor(v) + '"></span></span>' +
      '<span class="db-val">' + fmtRisk(v) + "</span></span>";
  }).join("") + "</span>";
}

function gateChipsHtml(entry) {
  if (!entry) return '<span class="muted">—</span>';
  if (entry.gates_passed) return '<span class="gate-chip ok">gates ok</span>';
  const failed = entry.failed_gates || [];
  if (!failed.length) return '<span class="gate-chip bad">gates failed</span>';
  return failed.map((g) =>
    '<span class="gate-chip bad" title="hard gate exceeded">' + escapeHtml(g) + "</span>").join(" ");
}

/* Why a non-selected candidate lost against the reference (winner / best_available). */
function decWhyLost(entry, refEntry, refCid, cid) {
  if (!entry) return "—";
  if ((entry.failed_gates || []).length) {
    return "failed gate: " + entry.failed_gates.map(escapeHtml).join(", ");
  }
  if (!refEntry || cid === refCid) return "—";
  let worst = null, worstDelta = 0;
  const per = entry.per_defect || {}, refPer = refEntry.per_defect || {};
  for (const d of Object.keys(per)) {
    const delta = (per[d] || 0) - (refPer[d] || 0);
    if (delta > worstDelta) { worstDelta = delta; worst = d; }
  }
  if (worst && worstDelta > 0.005) {
    return "higher " + escapeHtml(worst) + " risk (+" + worstDelta.toFixed(2) + ")";
  }
  const dm = (entry.risk_mean || 0) - (refEntry.risk_mean || 0);
  if (dm > 0.005) return "higher overall risk (+" + dm.toFixed(2) + ")";
  return "bounds overlap with the winner";
}

function decBannerHtml(dec) {
  const rep = (dec && dec.report) || {};
  const status = (dec && dec.status) || rep.status;
  const mode = (dec && dec.mode) || rep.mode;
  const reason = rep.reason ? ' <span class="dec-reason">' + escapeHtml(rep.reason) + "</span>" : "";
  if (status === "final" && mode === "clear_winner") {
    return '<div class="dec-banner final-clear"><span class="dec-status">FINAL · clear winner</span> ' +
      decCandName(dec.candidate_recipe_id) +
      (rep.risk_upper != null ? ' <span class="dec-kv">risk ≤ ' + fmtRisk(rep.risk_upper) + "</span>" : "") +
      reason + "</div>";
  }
  if (status === "final") {  // best_safe (or an unknown final mode — treat as not-certain)
    return '<div class="dec-banner final-safe"><span class="dec-status">FINAL · best safe</span> ' +
      decCandName(dec.candidate_recipe_id) +
      (rep.risk_upper != null ? ' <span class="dec-kv">risk ≤ ' + fmtRisk(rep.risk_upper) + "</span>" : "") +
      ' <span class="dec-warn">bounds overlap — this choice is <b>not certain</b>, it is the ' +
      "safest of the gate-passing candidates</span>" + reason + "</div>";
  }
  if (status === "no_acceptable_candidate") {
    const failed = rep.failed_gates || [];
    return '<div class="dec-banner none"><span class="dec-status">NO ACCEPTABLE CANDIDATE</span>' +
      (rep.best_available ? " best available: " + decCandName(rep.best_available) : "") +
      (failed.length ? ' <span class="dec-kv">failed gates: ' +
        failed.map(escapeHtml).join(", ") + "</span>" : "") +
      reason + "</div>";
  }
  return '<div class="dec-banner unknown"><span class="dec-status">' +
    escapeHtml(status || "unknown status") + "</span>" + reason + "</div>";
}

function decSelectedHtml(dec) {
  const rep = dec.report || {};
  const cands = rep.candidates || {};
  const isFinal = dec.status === "final";
  const cid = isFinal ? dec.candidate_recipe_id : rep.best_available;
  const entry = cid != null ? cands[cid] : null;
  if (!entry) return "";
  const params = rep.params || {};
  const ceiling = params.max_acceptable_risk;
  const head = isFinal ? "Selected candidate" : "Best available (not selected)";
  const paramsBits = [];
  if (params.lambda != null) paramsBits.push("λ=" + params.lambda);
  if (params.alpha != null) paramsBits.push("α=" + params.alpha);
  if (params.delta != null) paramsBits.push("δ=" + params.delta);
  if (ceiling != null) paramsBits.push("risk ceiling=" + ceiling);
  return '<div class="dec-selected' + (isFinal ? "" : " not-final") + '">' +
    '<div class="dec-sel-head">' + head + ": " + decCandName(cid) + "</div>" +
    '<div class="dec-sel-risk">' +
    '<span class="dec-risk-nums">' + fmtRisk(entry.risk_lower) + " – <b>" +
    fmtRisk(entry.risk_mean) + "</b> – " + fmtRisk(entry.risk_upper) + "</span>" +
    riskBarHtml(entry.risk_lower, entry.risk_mean, entry.risk_upper, ceiling) +
    '<span class="muted" style="font-size:11px;">risk lower – mean – upper (bootstrap 5–95%)</span>' +
    "</div>" +
    (paramsBits.length ? '<div class="dec-params muted">selector params: ' +
      paramsBits.join(" · ") + "</div>" : "") +
    "</div>";
}

function decCandTableHtml(dec) {
  const rep = dec.report || {};
  const cands = rep.candidates || {};
  const ids = Object.keys(cands);
  if (!ids.length) return "";
  const refCid = dec.status === "final" ? dec.candidate_recipe_id : rep.best_available;
  const refEntry = refCid != null ? cands[refCid] : null;
  // eligible (gates passed) first, then by mean risk
  ids.sort((a, b) => {
    const ea = cands[a], eb = cands[b];
    if (!!ea.gates_passed !== !!eb.gates_passed) return ea.gates_passed ? -1 : 1;
    return (ea.risk_mean || 0) - (eb.risk_mean || 0);
  });
  const ceiling = (rep.params || {}).max_acceptable_risk;
  const rows = ids.map((cid) => {
    const e = cands[cid];
    const isRef = cid === refCid;
    let verdict;
    if (isRef && dec.status === "final") {
      verdict = '<span class="dec-verdict win">selected · ' +
        (dec.mode === "clear_winner" ? "clear winner" : "best safe (not certain)") + "</span>";
    } else if (isRef) {
      verdict = '<span class="dec-verdict best-avail">best available — still unacceptable</span>';
    } else {
      verdict = decWhyLost(e, refEntry, refCid, cid);
    }
    return "<tr" + (isRef ? ' class="winner"' : "") + "><td>" + decCandName(cid) + "</td>" +
      "<td>" + gateChipsHtml(e) + "</td>" +
      '<td class="db-cell">' + defectBarsHtml(e.per_defect) + "</td>" +
      '<td class="risk-cell"><span class="dec-risk-nums">' + fmtRisk(e.risk_lower) + " – <b>" +
      fmtRisk(e.risk_mean) + "</b> – " + fmtRisk(e.risk_upper) + "</span>" +
      riskBarHtml(e.risk_lower, e.risk_mean, e.risk_upper, ceiling) + "</td>" +
      '<td class="why-cell">' + verdict + "</td></tr>";
  }).join("");
  return '<div class="dec-subhead">Candidates under audit <span class="muted">(sorted by risk; ' +
    "lower is better)</span></div>" +
    '<div class="table-scroll"><table class="cands dec-table">' +
    "<thead><tr><th>candidate</th><th>hard gates</th><th>per-defect risk</th>" +
    "<th>risk (lower – mean – upper)</th><th>why not selected</th></tr></thead>" +
    "<tbody>" + rows + "</tbody></table></div>";
}

/* colored challenge matrix: rows = cases, cols = candidates, cell = key metric */
function heatStyle(v, best, worst) {
  if (v == null || !isFinite(v) || best == null || !isFinite(best)) return "";
  const span = Math.abs(best - worst);
  const g = span < 1e-12 ? 1 : clamp(1 - Math.abs(v - best) / span, 0, 1);
  const r = Math.round(255 + (126 - 255) * g);
  const gg = Math.round(107 + (224 - 107) * g);
  const b = Math.round(107 + (200 - 107) * g);
  return "background:rgba(" + r + "," + gg + "," + b + ",0.20);";
}
function fmtMetric(v) {
  if (v == null || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a < 0.01 && a > 0) return v.toExponential(1);
  return v.toFixed(2);
}
function classLabel(cls) {
  const bits = Object.entries(cls || {}).map(([k, v]) => escapeHtml(k) + "=" + escapeHtml(String(v)));
  return bits.join(" · ");
}

function decMatrixHtml(chal, dec) {
  if (!chal || !chal.available) {
    return '<div class="dec-subhead">Challenge performance</div>' +
      '<div class="muted">no challenge cases recorded for this run</div>';
  }
  const cases = chal.cases || [];
  const results = chal.results || {};
  // column order: decision-report risk order when available, else payload order
  let cols = chal.candidates || [];
  const rep = (dec && dec.report) || {};
  if (rep.candidates) {
    const known = Object.keys(rep.candidates)
      .sort((a, b) => (rep.candidates[a].risk_mean || 0) - (rep.candidates[b].risk_mean || 0))
      .filter((c) => cols.includes(c));
    cols = known.concat(cols.filter((c) => !known.includes(c)));
  }
  if (!cases.length || !cols.length) {
    return '<div class="dec-subhead">Challenge performance</div>' +
      '<div class="muted">' + (cases.length ? "no per-candidate challenge results yet"
        : "no challenge cases recorded for this run") + "</div>";
  }
  let head = "<thead><tr><th>challenge case</th>" + cols.map((c) =>
    '<th class="numc" title="' + escapeHtml(c) + '">' + decCandName(c) +
    backendChipHtml(backendFor(c)) + "</th>").join("") +
    "</tr></thead>";
  const body = cases.map((cs) => {
    const row = results[cs.challenge_id] || {};
    // normalize the heat over cells sharing the row's modal metric
    const prim = cols.map((c) => (row[c] && row[c].primary) || null);
    const metrics = prim.filter(Boolean).map((p) => p.metric);
    const modal = metrics.sort((a, b) =>
      metrics.filter((m) => m === a).length - metrics.filter((m) => m === b).length).pop();
    const vals = prim.filter((p) => p && p.metric === modal).map((p) => p.value);
    const hib = (prim.find((p) => p && p.metric === modal) || {}).higher_is_better;
    const best = vals.length ? (hib ? Math.max.apply(null, vals) : Math.min.apply(null, vals)) : null;
    const worst = vals.length ? (hib ? Math.min.apply(null, vals) : Math.max.apply(null, vals)) : null;
    const cls = classLabel(cs.class);
    const cells = cols.map((c) => {
      const cell = row[c];
      if (!cell || !cell.primary) return '<td class="numc muted">—</td>';
      const p = cell.primary;
      const tip = Object.entries(cell.metrics || {})
        .filter(([, v]) => typeof v === "number")
        .map(([k, v]) => k + "=" + fmtMetric(v)).join("  ");
      const style = p.metric === modal ? heatStyle(p.value, best, worst) : "";
      return '<td class="numc" style="' + style + '" title="' + escapeHtml(tip) + '">' +
        fmtMetric(p.value) + '<span class="cell-metric">' + escapeHtml(p.metric) + "</span></td>";
    }).join("");
    return '<tr><td class="case-cell"><span class="case-type">' + escapeHtml(cs.challenge_type) +
      "</span>" + (cls ? '<span class="case-class">' + cls + "</span>" : "") + "</td>" + cells + "</tr>";
  }).join("");
  return '<div class="dec-subhead">Challenge performance ' +
    '<span class="muted">(exact-reference cases; SI-SDR higher is better, errors lower)</span></div>' +
    '<div class="table-scroll"><table class="cands dec-matrix">' + head +
    "<tbody>" + body + "</tbody></table></div>";
}

/* ============= challenge visualizations (exact-reference evidence) ========
   Three dependency-free inline-SVG views over the /api/v2/challenges payload,
   rendered into #chalVizBody:
     1. SI-SDR heatmap    — exact-reference cases × models (brighter = cleaner)
     2. theft-assay bars  — mean broadband theft per model on a log axis, drawn
                            against the calibrated orchestral-theft gate
     3. SI-SDR spread     — each model's min · median · max across the cases
   Every view is defensive: a missing metric/table degrades to a muted note. */

function chalCaseById(id) {
  const cases = (S.challenges && S.challenges.cases) || [];
  return cases.find((c) => c.challenge_id === id) || null;
}
function chalCaseLabel(cs) {
  if (!cs) return "?";
  const cl = cs.class || {};
  if (cs.challenge_type === "track_remix" && cl.vocal_level_db != null) {
    return "remix " + (cl.vocal_level_db > 0 ? "+" : "") + cl.vocal_level_db + " dB";
  }
  return String(cs.challenge_type || "case").replace(/_/g, " ");
}
function chalCaseSub(cs) {
  const cl = (cs && cs.class) || {};
  if (cl.pan && cl.pan !== "center") return String(cl.pan);
  if (cl.probe) return String(cl.probe);
  return "";
}
function chalModelName(cid) { const i = decCandInfo(cid); return i.label || i.short; }
function chalShort(cid) { return decCandInfo(cid).short; }

/* raw theft_mean gate: invert the orchestral_theft severity map at its certified
   τ (smallest raw whose calibrated severity reaches τ). null when uncertified. */
function theftGateFromCal(cal) {
  const d = cal && cal.calibration && cal.calibration.defects
    ? cal.calibration.defects.orchestral_theft
    : (cal && cal.defects && cal.defects.orchestral_theft);
  const mk = d && d.map_knots;
  if (!d || !mk || !Array.isArray(mk.xs) || !mk.xs.length) return null;
  const ltt = d["learn_then_test_unit=work_model"] || {};
  let tau = null;
  const strongest = d.strongest_certifiable_target;
  if (strongest != null && ltt[String(strongest)] && ltt[String(strongest)].certifiable) {
    tau = ltt[String(strongest)].tau;
  } else {
    for (const k of Object.keys(ltt)) if (ltt[k] && ltt[k].certifiable) { tau = ltt[k].tau; break; }
  }
  if (tau == null) return null;
  const xs = mk.xs, ys = mk.ys || [];
  for (let i = 0; i < xs.length; i++) if (ys[i] >= tau - 1e-9) return { raw: xs[i], tau: tau };
  return null;
}

function median(sorted) {
  const n = sorted.length;
  if (!n) return null;
  return n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

function renderChallengeViz() {
  const host = document.getElementById("chalVizBody");
  const meta = document.getElementById("chalVizMeta");
  if (!host) return;
  if (meta) meta.textContent = "";
  const chal = S.challenges;
  if (!chal || !chal.available) {
    host.innerHTML = '<div class="muted">no challenge cases recorded for this run — run ' +
      "<code>challenges build</code> then <code>challenges run</code>.</div>";
    return;
  }
  const cases = chal.cases || [];
  const results = chal.results || {};
  let cols = (chal.candidates || []).slice();
  if (!cols.length) {
    host.innerHTML = '<div class="muted">challenge cases exist but no per-candidate results yet.</div>';
    return;
  }
  // model column order: mirror the decision-report risk order when available
  const rep = (S.decision && S.decision.report) || {};
  if (rep.candidates) {
    const known = Object.keys(rep.candidates)
      .sort((a, b) => (rep.candidates[a].risk_mean || 0) - (rep.candidates[b].risk_mean || 0))
      .filter((c) => cols.includes(c));
    cols = known.concat(cols.filter((c) => !known.includes(c)));
  }
  const colObjs = cols.map((c) => ({ id: c, label: chalModelName(c), sub: chalShort(c) }));

  host.innerHTML = "";
  let nViews = 0;

  const siOf = (chid, cid) => {
    const cell = (results[chid] || {})[cid];
    const v = cell && cell.metrics && cell.metrics.si_sdr_db;
    return typeof v === "number" && isFinite(v) ? v : null;
  };
  const siCases = cases.filter((cs) => cols.some((c) => siOf(cs.challenge_id, c) != null));

  // ---------------- (1) SI-SDR heatmap: cases × models --------------------
  if (siCases.length) {
    const sec = document.createElement("div");
    sec.className = "chal-viz-section";
    sec.innerHTML = '<div class="dec-subhead">SI-SDR by case × model ' +
      '<span class="muted">(dB vs the exact remix target — brighter is cleaner)</span></div>';
    const heat = svgHeatmap({
      rows: siCases.map((cs) => ({ id: cs.challenge_id, label: chalCaseLabel(cs), sub: chalCaseSub(cs) })),
      cols: colObjs,
      value: (rid, cid) => siOf(rid, cid),
      fmt: (v) => v.toFixed(1),
      higherBetter: true,
      tip: (rid, cid) => {
        const m = ((results[rid] || {})[cid] || {}).metrics || {};
        const bits = [];
        if (typeof m.si_sdr_db === "number") bits.push("SI-SDR <b>" + m.si_sdr_db.toFixed(2) + " dB</b>");
        if (typeof m.stft_distance === "number") bits.push("STFT dist " + m.stft_distance.toFixed(4));
        if (typeof m.band_envelope_err_db === "number") bits.push("band-env err " + m.band_envelope_err_db.toFixed(2) + " dB");
        return chalModelName(cid) + " · " + chalCaseLabel(chalCaseById(rid)) + "<br>" + bits.join("<br>");
      },
    });
    const scroll = document.createElement("div"); scroll.className = "viz-scroll";
    scroll.appendChild(heat.el); sec.appendChild(scroll);
    sec.appendChild(seqLegend(heat.min, heat.max, "SI-SDR (dB) · higher = cleaner extraction", (v) => v.toFixed(1)));
    host.appendChild(sec); nViews++;
  }

  // ---------------- (2) no-vocal theft assay bars -------------------------
  let theftRow = results["theft_assay"] || null;
  if (!theftRow) {
    const tc = cases.find((cs) => cs.challenge_type === "theft_assay");
    if (tc) theftRow = results[tc.challenge_id] || null;
  }
  if (theftRow) {
    const gate = theftGateFromCal(S.calibration);
    const gateRaw = gate ? gate.raw : null;
    const items = cols.filter((c) => theftRow[c]).map((c) => {
      const cell = theftRow[c] || {}; const m = cell.metrics || {};
      let tm = typeof m.theft_mean === "number" ? m.theft_mean
        : (cell.primary && cell.primary.metric === "theft_mean" ? cell.primary.value : null);
      let verdict = null;
      if (gateRaw != null && tm != null) verdict = tm <= gateRaw ? { word: "within", kind: "ok" } : { word: "steals", kind: "crit" };
      return {
        label: chalModelName(c), sub: chalShort(c), value: tm, verdict,
        tip: chalModelName(c) + " · mean broadband theft <b>" + (tm != null ? tm.toFixed(4) : "—") + "</b>" +
          (gateRaw != null ? "<br>gate ≤ <b>" + gateRaw.toFixed(4) + "</b> (τ = " + gate.tau.toFixed(2) + ")" : ""),
      };
    });
    const sec = document.createElement("div");
    sec.className = "chal-viz-section";
    sec.innerHTML = '<div class="dec-subhead">No-vocal theft assay ' +
      '<span class="muted">(mean broadband energy taken from genuine controls — lower is better)</span></div>';
    const scroll = document.createElement("div"); scroll.className = "viz-scroll";
    scroll.appendChild(svgLogBars({
      items, gate: gateRaw,
      gateLabel: gate ? "gate ≤ " + gateRaw.toFixed(3) : null,
      xLabel: "mean broadband theft ratio (log scale)",
      fmt: (v) => (v == null ? "—" : v.toFixed(4)),
    }));
    sec.appendChild(scroll);
    sec.insertAdjacentHTML("beforeend", gate
      ? '<div class="explain">Gate is the raw theft ratio at the certified orchestral-theft threshold ' +
        "τ = " + gate.tau.toFixed(2) + " (inverted through the severity map on the calibration page). " +
        "Bars past it exceed the certified feasibility limit.</div>"
      : '<div class="explain">No certified orchestral-theft gate in the loaded calibration — bars are drawn ' +
        "without a gate line.</div>");
    host.appendChild(sec); nViews++;
  }

  // ---------------- (3) per-model SI-SDR spread ---------------------------
  if (siCases.length) {
    const items = cols.map((c) => {
      const vals = [];
      for (const cs of siCases) { const v = siOf(cs.challenge_id, c); if (v != null) vals.push(v); }
      vals.sort((a, b) => a - b);
      return {
        label: chalModelName(c), sub: chalShort(c),
        min: vals.length ? vals[0] : null, med: median(vals), max: vals.length ? vals[vals.length - 1] : null,
        tip: chalModelName(c) + " · SI-SDR over " + vals.length + " exact-reference case" + (vals.length === 1 ? "" : "s"),
      };
    });
    const sec = document.createElement("div");
    sec.className = "chal-viz-section";
    sec.innerHTML = '<div class="dec-subhead">SI-SDR spread per model ' +
      '<span class="muted">(min · median · max across exact-reference cases — right is better)</span></div>';
    const scroll = document.createElement("div"); scroll.className = "viz-scroll";
    scroll.appendChild(svgRangeBars({ items, xLabel: "SI-SDR (dB)", fmt: (v) => (v == null ? "—" : v.toFixed(1)) }));
    sec.appendChild(scroll);
    host.appendChild(sec); nViews++;
  }

  if (!nViews) {
    host.innerHTML = '<div class="muted">challenge results present but no SI-SDR or theft metrics to plot.</div>';
    return;
  }
  if (meta) meta.textContent = cases.length + " cases · " + cols.length + " models";
}

function decActionsHtml(act) {
  if (!act || !act.available) {
    return '<div class="dec-subhead">Conductor probes &amp; actions</div>' +
      '<div class="muted">no conductor actions recorded for this run</div>';
  }
  const actions = act.actions || [];
  if (!actions.length) {
    return '<div class="dec-subhead">Conductor probes &amp; actions</div>' +
      '<div class="muted">conductor_action table is empty</div>';
  }
  const counts = Object.entries(act.counts || {}).map(([k, n]) => n + " " + escapeHtml(k)).join(" · ");
  const rows = actions.map((a) => {
    const detail = Object.entries(a.proposed || {})
      .filter(([k]) => k !== "type")
      .map(([k, v]) => escapeHtml(k) + "=" + escapeHtml(typeof v === "object" ? JSON.stringify(v) : String(v)))
      .join(" · ");
    const validated = a.validated_recipe_id
      ? '<div class="cand-sub">→ ' + decCandName(a.validated_recipe_id) + "</div>" : "";
    const st = String(a.status || "");
    const cls = /executed|validated|done|ok/i.test(st) ? "ok"
      : /rejected|failed|refused/i.test(st) ? "bad"
      : /skipped|deferred/i.test(st) ? "warn" : "dim";
    return '<tr><td class="numc">' + (a.round == null ? "—" : a.round) + "</td>" +
      '<td class="act-cell"><b>' + escapeHtml(a.type || "action") + "</b>" +
      (detail ? '<div class="cand-sub">' + detail + "</div>" : "") + validated + "</td>" +
      '<td><span class="act-pill ' + cls + '">' + escapeHtml(st) + "</span></td>" +
      '<td class="why-cell">' + (a.reason ? escapeHtml(a.reason) : '<span class="muted">—</span>') +
      "</td></tr>";
  }).join("");
  return '<div class="dec-subhead">Conductor probes &amp; actions' +
    (counts ? ' <span class="muted">(' + counts + ")</span>" : "") + "</div>" +
    '<div class="table-scroll"><table class="cands dec-actions">' +
    "<thead><tr><th>round</th><th>proposed action</th><th>status</th><th>reason</th></tr></thead>" +
    "<tbody>" + rows + "</tbody></table></div>";
}

function decFooterHtml(dec) {
  const bits = [];
  if (dec && dec.available) {
    if (dec.selector_version) bits.push("selector " + escapeHtml(dec.selector_version));
    if (dec.decision_id) bits.push("decision " + escapeHtml(String(dec.decision_id).slice(0, 24)));
    if (dec.n_decisions > 1) bits.push(dec.n_decisions + " decisions recorded (latest shown)");
  }
  const judges = (dec && dec.judges) || [];
  if (judges.length) {
    bits.push("judge calibration: " + judges.map((j) =>
      escapeHtml(j.judge_id) + (j.passed_axes && j.passed_axes.length
        ? " (" + j.passed_axes.map(escapeHtml).join(", ") + ")" : " (no axes passed)")).join(" · "));
  }
  return bits.length ? '<div class="dec-footer muted">' + bits.join("  ·  ") + "</div>" : "";
}

async function loadDecisionPanel(tid) {
  const body = document.getElementById("decisionBody");
  const chalBody = document.getElementById("chalVizBody");
  if (chalBody) chalBody.innerHTML = '<div class="muted"><span class="spin"></span> loading challenges…</div>';
  const get = (url) => fetchJSON(url).catch(() => null);
  const q = encodeURIComponent(tid);
  const [dec, chal, act, cal] = await Promise.all([
    get("/api/v2/decision/" + q),
    get("/api/v2/challenges/" + q),
    get("/api/v2/actions/" + q),
    S.calibration ? Promise.resolve(S.calibration) : get("/api/v2/calibration"),
  ]);
  if (S.tid !== tid) return;   // user already switched runs
  S.decision = dec; S.challenges = chal; S.actions = act; S.calibration = cal;
  renderDecisionPanel(body);
  renderChallengeViz();
  renderCandTable();           // backend chips may now resolve via the lock
}

function renderDecisionPanel(body) {
  body = body || document.getElementById("decisionBody");
  const dec = S.decision, chal = S.challenges, act = S.actions;
  const meta = document.getElementById("decMeta");
  meta.textContent = (dec && dec.available)
    ? [dec.selector_version, dec.created_at].filter(Boolean).join(" · ") : "";
  const nothing = (!dec || !dec.available) && (!chal || !chal.available) && (!act || !act.available);
  if (nothing) {
    body.innerHTML = '<div class="dec-banner none-yet">no autonomous decision recorded yet ' +
      "for this run — the panel fills in once the v2 pipeline records " +
      "<code>selection_decision</code> / <code>challenge_case</code> / " +
      "<code>conductor_action</code> facts</div>";
    return;
  }
  let html = "";
  if (dec && dec.available) {
    html += decBannerHtml(dec);
    html += decSelectedHtml(dec);
    html += decCandTableHtml(dec);
  } else {
    html += '<div class="dec-banner none-yet">no selection decision recorded yet</div>';
  }
  html += decMatrixHtml(chal, dec);
  html += decActionsHtml(act);
  html += decFooterHtml(dec);
  body.innerHTML = html;
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
