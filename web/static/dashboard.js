/* dashboard.js — pipeline overview: state-machine strips, screening table,
   local run launcher + queue. Depends on common.js (fetchJSON, escapeHtml,
   fmtTime, fmtElapsed) + charts.js (VIZ tokens only). */

const TAG_COLORS_D = {   // mirror of the run page's tag palette
  high_soprano: "#ff6b9d", extreme_soprano: "#ff3b6b", dense_accompaniment: "#ffb454",
  quiet_backing: "#7ee0c8", hall_tail: "#b48cff", no_vocal_control: "#8a94a6",
  random_control: "#5b6472", vocal_overlap: "#4aa8ff",
};
function tagColorD(tag) {
  if (TAG_COLORS_D[tag]) return TAG_COLORS_D[tag];
  let h = 0;
  for (let i = 0; i < String(tag).length; i++) h = (h * 31 + tag.charCodeAt(i)) & 0xffff;
  return "hsl(" + (h % 360) + ",60%,62%)";
}

const D = {
  overview: null,
  panel: null,
  queue: null,
  srcMode: "existing",
  pollTimer: null,
};

/* ------------------------------ overview -------------------------------- */
function decisionChip(dec) {
  if (!dec) return '<span class="kchip dec-missing">no decision yet</span>';
  if (dec.status === "final" && dec.mode === "clear_winner")
    return '<span class="kchip dec-final">FINAL · clear winner</span>';
  if (dec.status === "final")
    return '<span class="kchip dec-safe">FINAL · ' + escapeHtml(dec.mode || "best safe") + "</span>";
  if (dec.status === "no_acceptable_candidate")
    return '<span class="kchip dec-none">NO ACCEPTABLE</span>';
  return '<span class="kchip">' + escapeHtml(dec.status || "?") + "</span>";
}

function stripHtml(run, states, shorts) {
  const prog = run.progress || { index: -1, failed: false };
  const segs = states.map((st, i) => {
    let cls = "";
    if (prog.failed) {
      cls = i <= prog.index ? "done" : (i === Math.max(0, prog.index + 1) ? "fail" : "");
    } else if (prog.index >= 0) {
      if (i < prog.index) cls = "done";
      else if (i === prog.index) cls = (st === "COMPLETE" ? "complete-done" : "cur");
    }
    return '<div class="seg ' + cls + '" title="' + escapeHtml(st) + '">' +
      '<div class="bar"></div><div class="lab">' + escapeHtml(shorts[i]) + "</div></div>";
  }).join("");
  let note = "";
  if (prog.failed) {
    note = '<div class="strip-note"><span class="fail-word">' + escapeHtml(prog.label || "FAILED") +
      "</span> — pipeline stopped</div>";
  } else if (prog.label && prog.label !== states[prog.index]) {
    note = '<div class="strip-note">state: <b>' + escapeHtml(prog.label) + "</b></div>";
  }
  // live launcher stage from run_state (budgets.launcher)
  const l = run.run_state && run.run_state.budgets && run.run_state.budgets.launcher;
  if (l && l.status === "running" && l.stage) {
    note += '<div class="strip-note"><span class="kchip launching">launcher: ' +
      escapeHtml(l.stage) + " (" + ((l.stage_index || 0) + 1) + "/" + (l.n_stages || "?") + ")</span></div>";
  }
  return '<div class="strip">' + segs + "</div>" + note;
}

function renderRuns() {
  const el = document.getElementById("runsWrap");
  const ov = D.overview;
  document.getElementById("runCount").textContent = (ov.runs || []).length + " runs";
  if (!ov.runs || !ov.runs.length) {
    el.innerHTML = '<div class="notice">No runs under this <code>--lib</code> yet. ' +
      "Launch one below, or seed demo data with " +
      "<code>uv run python web/dev_seed_decision.py --lib &lt;root&gt;</code>.</div>";
    return;
  }
  el.innerHTML = ov.runs.map((r) => {
    const scr = r.screening || {};
    const ch = r.challenges || {};
    const dur = r.duration_s ? fmtTime(r.duration_s) : "—";
    const ratio = scr.min_control_ratio != null
      ? ' <b>· min ratio ' + scr.min_control_ratio.toFixed(3) + "</b>" : "";
    return '<a class="card run-row" style="display:block;color:inherit;" href="' + r.url + '">' +
      '<div class="rr-head"><span class="rr-title">' + escapeHtml(r.title || r.track_id) + "</span>" +
      '<span class="rr-meta">' + dur + " · " + (r.sample_rate_hz || "?") + " Hz · " +
      escapeHtml(r.track_id) + "</span></div>" +
      stripHtml(r, ov.state_machine, ov.state_machine_short) +
      '<div class="rr-chips">' +
      '<span class="kchip"><b>' + (r.n_passages || 0) + "</b> passages</span>" +
      '<span class="kchip"><b>' + (scr.n_controls || 0) + "</b> controls" + ratio + "</span>" +
      '<span class="kchip"><b>' + (r.n_candidates || 0) + "</b> candidates</span>" +
      '<span class="kchip"><b>' + (ch.n_cases || 0) + "</b> challenge cases · <b>" +
        (ch.n_models || 0) + "</b> models</span>" +
      decisionChip(r.decision) +
      "</div></a>";
  }).join("");
}

function renderScreening() {
  const t = document.getElementById("screeningTable");
  const runs = (D.overview && D.overview.runs) || [];
  if (!runs.length) { t.innerHTML = ""; return; }
  const head = "<thead><tr><th>track</th><th>duration</th>" +
    "<th>controls <span class=\"axis-hint\">(vocal-energy ratios)</span></th>" +
    "<th>hard-vocal categories fired</th><th>decision</th></tr></thead>";
  const body = "<tbody>" + runs.map((r) => {
    const scr = r.screening || {};
    const ratios = (scr.control_ratios || []).map((x) => x.toFixed(3)).join(", ");
    const tags = (scr.hard_tags || []).map((tg) =>
      '<span class="tag-mini"><span class="tagdot" style="background:' + tagColorD(tg) + '"></span>' +
      escapeHtml(tg) + "</span>").join(" ") || '<span class="muted">none</span>';
    return "<tr><td><a href=\"" + r.url + '">' + escapeHtml(r.track_id) + "</a></td>" +
      '<td class="numc">' + (r.duration_s ? fmtTime(r.duration_s) : "—") + "</td>" +
      '<td><b>' + (scr.n_controls || 0) + "</b>" +
      (ratios ? ' <div class="ratio-list">[' + ratios + "]</div>" : "") + "</td>" +
      '<td style="white-space:normal;max-width:340px;">' + tags + "</td>" +
      "<td>" + decisionChip(r.decision) + "</td></tr>";
  }).join("") + "</tbody>";
  t.innerHTML = head + body;
}

/* ------------------------------ launcher -------------------------------- */
function renderPanel() {
  const el = document.getElementById("modelList");
  const p = D.panel;
  if (!p || !p.panel_available) {
    el.innerHTML = '<span class="muted">configs/panel.yaml could not be read on the server</span>';
    return;
  }
  document.getElementById("lockNote").textContent = p.lock_present
    ? "lock: " + p.lock_path : "no model-lock.json — import models to enable launching";
  const rows = [];
  for (const m of p.models) {
    const enabled = !!m.locked;
    const right = enabled
      ? '<span class="lock-badge">🔒 locked <span class="bh">' +
        escapeHtml(String((m.bundle || {}).bundle_sha256 || "").slice(0, 10)) + "</span></span>"
      : '<span class="import-hint">' + (m.needs_import ? "needs import (hash placeholder)" : "import to enable") + "</span>";
    const value = enabled ? (m.bundle.logical_id || m.id) : "";
    rows.push('<label class="model-item' + (enabled ? "" : " disabled") + '">' +
      '<input type="checkbox" name="model" value="' + escapeHtml(value) + '"' +
      (enabled ? "" : " disabled") + ">" +
      '<span><span class="mi-name">' + escapeHtml(m.id) + '</span><br>' +
      '<span class="mi-fam">' + escapeHtml(m.family || "?") + " · " + escapeHtml(m.target || "?") +
      " · " + escapeHtml(m.checkpoint || "") + "</span></span>" +
      '<span class="mi-right">' + right + "</span></label>");
  }
  for (const b of p.extra_bundles || []) {   // imported models not in panel.yaml
    rows.push('<label class="model-item">' +
      '<input type="checkbox" name="model" value="' + escapeHtml(b.logical_id) + '">' +
      '<span><span class="mi-name">' + escapeHtml(b.logical_id) + '</span><br>' +
      '<span class="mi-fam">' + escapeHtml(b.family || "?") + " · " + escapeHtml(b.target || "?") +
      ' · not in panel.yaml</span></span>' +
      '<span class="mi-right"><span class="lock-badge">🔒 locked <span class="bh">' +
      escapeHtml(String(b.bundle_sha256 || "").slice(0, 10)) + "</span></span></span></label>");
  }
  el.innerHTML = rows.join("");
}

function renderTrackPick() {
  const sel = document.getElementById("trackPick");
  const runs = (D.overview && D.overview.runs) || [];
  sel.innerHTML = runs.length
    ? runs.map((r) => '<option value="' + escapeHtml(r.track_id) + '">' + escapeHtml(r.track_id) + "</option>").join("")
    : '<option value="">— no ingested runs; upload a file —</option>';
}

function setSrcMode(mode) {
  D.srcMode = mode;
  document.getElementById("srcExisting").classList.toggle("on", mode === "existing");
  document.getElementById("srcUpload").classList.toggle("on", mode === "upload");
  document.getElementById("srcExistingWrap").classList.toggle("hidden", mode !== "existing");
  document.getElementById("srcUploadWrap").classList.toggle("hidden", mode !== "upload");
}

function launchStatus(msg, cls) {
  const el = document.getElementById("launchStatus");
  el.className = "launch-status " + (cls || "");
  el.innerHTML = msg;
  el.classList.remove("hidden");
}

async function submitLaunch() {
  const models = Array.from(document.querySelectorAll('#modelList input[name=model]:checked'))
    .map((i) => i.value).filter(Boolean);
  if (!models.length) { launchStatus("Pick at least one locked model.", "err"); return; }
  const construction = document.getElementById("constructionPick").value;
  const overlap = document.getElementById("overlapPick").value;
  const btn = document.getElementById("launchBtn");
  btn.disabled = true;
  launchStatus('<span class="spin"></span> submitting…', "");
  try {
    let resp;
    if (D.srcMode === "upload") {
      const f = document.getElementById("fileInput").files[0];
      if (!f) { launchStatus("Choose an audio file to upload.", "err"); btn.disabled = false; return; }
      const fd = new FormData();
      fd.append("file", f, f.name);
      fd.append("run_id", document.getElementById("runIdInput").value || "");
      fd.append("models", models.join(","));
      fd.append("construction", construction);
      fd.append("overlap", overlap);
      resp = await fetch("/api/v2/launch", { method: "POST", body: fd });
    } else {
      const run_id = document.getElementById("trackPick").value;
      if (!run_id) { launchStatus("No existing run selected.", "err"); btn.disabled = false; return; }
      resp = await fetch("/api/v2/launch", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id, models, construction, overlap }),
      });
    }
    const body = await resp.json();
    if (body.ok) {
      launchStatus("Queued <b>" + escapeHtml(body.run_id) + "</b> (job " +
        escapeHtml(body.job_id) + ") — watch the queue below.", "ok");
      pollQueue();
    } else {
      launchStatus("Launch refused: " + escapeHtml(body.error || "unknown error"), "err");
    }
  } catch (e) {
    launchStatus("Launch failed: " + escapeHtml(e.message), "err");
  }
  btn.disabled = false;
}

/* ------------------------------ queue ----------------------------------- */
function stageDots(job) {
  return '<div class="lq-stage-dots">' + (job.stages || []).map((st, i) => {
    const res = (job.stage_results || {})[st];
    let cls = "";
    if (res) cls = res.ok ? "ok" : "bad";
    else if (job.status === "running" && i === job.stage_index) cls = "run";
    return '<span class="sd ' + cls + '" title="' + escapeHtml(st + (res ? " — " + (res.ok ? "ok" : "FAILED") : "")) + '"></span>';
  }).join("") + "</div>";
}

function renderQueue() {
  const t = document.getElementById("queueTable");
  const q = D.queue;
  if (!q || !q.enabled) {
    t.innerHTML = '<tbody><tr><td class="muted">launcher disabled on this server (--no-launcher)</td></tr></tbody>';
    return;
  }
  if (!q.jobs.length) {
    t.innerHTML = '<tbody><tr><td class="muted">no launch jobs yet</td></tr></tbody>';
    return;
  }
  const head = "<thead><tr><th>run</th><th>models · options</th><th>stages</th>" +
    "<th>status</th><th>elapsed</th><th></th></tr></thead>";
  const body = "<tbody>" + q.jobs.map((j) => {
    const pillCls = { queued: "job-queued", running: "job-running", done: "job-done",
                      failed: "job-failed", cancelled: "job-canceled" }[j.status] || "";
    const stageTxt = j.status === "running" && j.stage
      ? '<div class="job-phase">' + escapeHtml(j.stage) + " (" + (j.stage_index + 1) + "/" + j.stages.length + ")</div>"
      : (j.error ? '<div class="job-err" title="' + escapeHtml(j.error) + '">' + escapeHtml(j.error) + "</div>" : "");
    const act = (j.status === "queued" || j.status === "running")
      ? '<button class="btn small danger-btn" data-act="cancel" data-id="' + j.id + '">Cancel</button>'
      : '<button class="btn small" data-act="remove" data-id="' + j.id + '">Remove</button>';
    const log = (j.log_tail && j.log_tail.length)
      ? '<details class="lq-details"><summary>log tail</summary><div class="lq-log">' +
        escapeHtml(j.log_tail.slice(-18).join("\n")) + "</div></details>" : "";
    return "<tr><td><a href=\"/r/" + encodeURIComponent(j.run_id) + '">' + escapeHtml(j.run_id) + "</a>" +
      (j.source_label ? '<div class="cand-sub">' + escapeHtml(j.source_label) + "</div>" : "") + "</td>" +
      '<td style="white-space:normal;max-width:260px;"><span class="mi-fam">' +
      escapeHtml((j.models || []).join(", ")) + "<br>" + escapeHtml(j.construction) +
      " · overlap " + j.overlap + "</span></td>" +
      "<td>" + stageDots(j) + stageTxt + log + "</td>" +
      '<td><span class="pill ' + pillCls + '">' + escapeHtml(j.status) + "</span></td>" +
      '<td class="numc">' + fmtElapsed(j.elapsed_s) + "</td>" +
      '<td class="job-actions">' + act + "</td></tr>";
  }).join("") + "</tbody>";
  t.innerHTML = head + body;
  t.querySelectorAll("button[data-act]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    await postAction("/api/v2/launcher/" + b.getAttribute("data-id") + "/" + b.getAttribute("data-act"));
    pollQueue();
    loadOverview();
  }));
}

/* ------------------------------ polling --------------------------------- */
async function loadOverview() {
  try {
    D.overview = await fetchJSON("/api/v2/overview");
  } catch (e) {
    document.getElementById("runsWrap").innerHTML =
      '<div class="notice">failed to load overview: ' + escapeHtml(e.message) + "</div>";
    return;
  }
  renderRuns();
  renderScreening();
  renderTrackPick();
}

async function loadPanel() {
  try { D.panel = await fetchJSON("/api/v2/panel"); } catch (e) { D.panel = null; }
  renderPanel();
}

async function pollQueue() {
  try { D.queue = await fetchJSON("/api/v2/launcher"); } catch (e) { D.queue = null; }
  renderQueue();
  const active = D.queue && D.queue.active;
  clearTimeout(D.pollTimer);
  D.pollTimer = setTimeout(() => { pollQueue(); if (active) loadOverview(); }, active ? 2500 : 10000);
}

async function init() {
  document.getElementById("refreshBtn").addEventListener("click", () => { loadOverview(); loadPanel(); pollQueue(); });
  document.getElementById("srcExisting").addEventListener("click", () => setSrcMode("existing"));
  document.getElementById("srcUpload").addEventListener("click", () => setSrcMode("upload"));
  document.getElementById("launchBtn").addEventListener("click", submitLaunch);
  document.getElementById("fileInput").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f && !document.getElementById("runIdInput").value) {
      document.getElementById("runIdInput").value =
        f.name.replace(/\.[A-Za-z0-9]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80);
    }
  });
  await loadOverview();
  loadPanel();
  pollQueue();
}

document.addEventListener("DOMContentLoaded", init);
