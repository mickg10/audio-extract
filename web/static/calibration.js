/* calibration.js — the scientific heart, made readable. Renders the REAL
   `audio-extract/calibration/v1` artifact (or a shape-identical dev seed):
     • per-defect severity CURVES  (defects[d].map_knots isotonic knots, SVG)
     • the τ table                 (learn_then_test at 3 targets + strongest cert)
     • out-of-sample violations    (held_out_violation: claimed vs observed)
     • split membership            (splits, per work)
     • in-domain reselection       (reselect_v3 decisions, when present)
   Defensive throughout: a missing block degrades to a muted note, an
   uncertified defect is rendered honestly (no fabricated τ line).
   Depends on common.js + charts.js. */

const C = { cal: null };
const TARGETS = ["0.1", "0.15", "0.2"];

function fmtP(v, digits) {
  return (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(digits == null ? 3 : digits);
}
function pct(v) { return (v == null || !isFinite(v)) ? "?" : Math.round(v * 100) + "%"; }

/* What raw measurement each defect's severity map is fitted from (from the
   artifact's truth_definitions; labels kept short for the axis). */
const RAW_LABELS = {
  orchestral_theft: { metric: "theft_mean", unit: "ratio",
    blurb: "mean broadband stem-energy theft measured on genuine no-vocal controls" },
  event_hole: { metric: "SI-SDR deficit proxy", unit: "0–1",
    blurb: "clip(1 − min control SI-SDR / 30 dB) — deployable proxy for missing orchestral events" },
  fullness: { metric: "band-envelope error proxy", unit: "0–1",
    blurb: "clip(mean control band-envelope error / 30 dB)" },
};
function rawLabel(name) {
  return RAW_LABELS[name] || { metric: "raw proxy", unit: "", blurb: "" };
}

/* learn_then_test entries (work×model unit) for a defect, keyed by target. */
function lttOf(defect) { return defect["learn_then_test_unit=work_model"] || {}; }
function sensitivityOf(defect) { return defect["learn_then_test_unit=work (sensitivity)"] || {}; }
function certTargets(defect) {
  const ltt = lttOf(defect);
  return TARGETS.filter((t) => ltt[t] && ltt[t].certifiable);
}
function strongestEntry(defect) {
  const ltt = lttOf(defect);
  const t = defect.strongest_certifiable_target;
  if (t != null && ltt[String(t)] && ltt[String(t)].certifiable) return ltt[String(t)];
  const c = certTargets(defect);
  return c.length ? ltt[c[0]] : null;
}

/* ------------------------- severity map curves --------------------------- */
function severityCurves(cal) {
  const defects = cal.defects || {};
  const names = Object.keys(defects);
  if (!names.length) return strDiv("no defects in this calibration artifact");
  const wrap = document.createElement("div");
  wrap.className = "chart-grid";
  for (const name of names) {
    const d = defects[name] || {};
    const mk = d.map_knots || {};
    const xs = mk.xs || [], ys = mk.ys || [];
    const rl = rawLabel(name);
    const strong = strongestEntry(d);
    const tau = strong ? strong.tau : null;
    const spread = ys.length ? Math.max.apply(null, ys) - Math.min.apply(null, ys) : 0;
    const flat = ys.length > 0 && spread < 0.2;

    const cell = document.createElement("div");
    cell.className = "chart-cell";
    const h = document.createElement("h4"); h.textContent = name;
    const sub = document.createElement("div"); sub.className = "cc-sub";
    sub.textContent = "raw: " + rl.metric + (rl.unit ? " (" + rl.unit + ")" : "") +
      " · " + xs.length + " isotonic knots · " + (d.fit_units || "?") + " fit units" +
      (d.fit_truly_bad != null ? " (" + d.fit_truly_bad + " truly-bad)" : "");
    cell.appendChild(h); cell.appendChild(sub);
    cell.appendChild(svgLineChart({
      xs, ys, w: 320, h: 190,
      xLabel: rl.metric + (rl.unit ? " (" + rl.unit + ")" : ""),
      yLabel: "calibrated severity",
      yDomain: [0, 1],
      yRef: tau != null ? { value: tau, label: "τ = " + fmtP(tau, 2) } : null,
    }));
    if (tau != null) {
      cell.insertAdjacentHTML("beforeend",
        '<div class="explain">' + escapeHtml(rl.blurb || "raw metric") +
        ", fitted monotonically (pool-adjacent-violators) to a 0–1 severity. Certified at " +
        "target risk " + pct(d.strongest_certifiable_target) + "; raw values whose severity clears the dashed " +
        "τ are refused by the feasibility gate.</div>");
    } else {
      cell.insertAdjacentHTML("beforeend",
        '<div class="cal-refused"><b>certification refused</b> — ' +
        (flat ? "near-flat severity map: a structurally uninformative proxy (it labels almost " +
          "everything the same, so no threshold separates good from bad). "
          : "the proxy does not track true damage on held-out groups. ") +
        "This defect keeps its hard gate only; its calibrated feasibility gate is off.</div>");
    }
    wrap.appendChild(cell);
  }
  return wrap;
}

/* ------------------------------ τ table ---------------------------------- */
function certBadge(entry, label) {
  if (!entry) return '<span class="cal-badge">' + label + ": —</span>";
  return entry.certifiable
    ? '<span class="cal-badge yes" title="τ=' + fmtP(entry.tau) + ", risk UCB ≤ " + fmtP(entry.risk_ucb) +
      ", " + (entry.n_groups != null ? entry.n_groups + " groups" : "") + '">' + label + " ✓</span>"
    : '<span class="cal-badge no" title="risk UCB 1.000 — not certifiable">' + label + " ✗</span>";
}

function tauTable(cal) {
  const defects = cal.defects || {};
  const names = Object.keys(defects);
  if (!names.length) return strDiv("no defects in this calibration artifact");
  let html = '<div class="table-scroll"><table class="cands">' +
    "<thead><tr><th>defect</th><th class=\"numc\">τ <span class=\"axis-hint\">(strongest cert.)</span></th>" +
    '<th class="numc">risk UCB <span class="axis-hint">Clopper–Pearson</span></th>' +
    '<th class="numc">groups</th><th>certifiable at target risk</th></tr></thead><tbody>';
  for (const name of names) {
    const d = defects[name] || {};
    const ltt = lttOf(d);
    const strong = strongestEntry(d);
    const badges = TARGETS.map((t) => certBadge(ltt[t], pct(parseFloat(t)))).join(" ");
    html += "<tr><td><b>" + escapeHtml(name) + "</b></td>" +
      '<td class="numc">' + (strong ? fmtP(strong.tau) : '<span class="muted">— not certified</span>') + "</td>" +
      '<td class="numc">' + (strong ? "≤ " + fmtP(strong.risk_ucb) : "1.000") + "</td>" +
      '<td class="numc">' + (strong && strong.n_groups != null ? strong.n_groups : "—") + "</td>" +
      "<td>" + badges + "</td></tr>";
  }
  html += "</tbody></table></div>";
  html += '<div class="explain">τ is the loosest severity threshold whose certified group still keeps the ' +
    "<i>upper-bounded</i> bad-rate under the target (exact one-sided Clopper–Pearson at δ = " +
    fmtP(cal.delta, 2) + "). Groups are <b>work × model</b> units; a work-only re-grouping (14 groups) is far " +
    "more conservative and certifies nothing here — the honest sensitivity floor. A ✗ defect keeps only its " +
    "hard gate.</div>";
  return html;
}

/* -------------------------- held-out violations --------------------------- */
function violationBlock(cal) {
  const hv = cal.held_out_violation || {};
  const defects = cal.defects || {};
  const names = Object.keys(hv);
  if (!names.length) return strDiv("no held-out violation check recorded");
  const wrap = document.createElement("div");
  for (const name of names) {
    const v = hv[name] || {};
    const per = v.per_target || {};
    const cert = certTargets(defects[name] || {});
    const box = document.createElement("div");
    box.className = "viol-defect";
    let head = '<div class="viol-head"><b>' + escapeHtml(name) + "</b> " +
      '<span class="muted">' + (v.n_units != null ? v.n_units + " held-out units" : "") + "</span></div>";
    if (!cert.length) {
      box.innerHTML = head + '<div class="muted">not certified on the fit split — nothing to check out of sample.</div>';
      wrap.appendChild(box); continue;
    }
    let rows = "";
    for (const t of TARGETS) {
      const p = per[t];
      if (!p || p.claimed_bound == null) continue;
      const observed = p.observed_rate;
      let verdict;
      if (observed == null) verdict = '<span class="muted">—</span>';
      else if (observed <= p.claimed_bound + 1e-9)
        verdict = '<span class="viol-ok">HOLDS ✓</span>';
      else
        verdict = '<span class="viol-bad">VIOLATED ✗</span>';
      rows += "<tr><td>" + pct(parseFloat(t)) + "</td>" +
        '<td class="numc">≤ ' + fmtP(p.claimed_bound) + "</td>" +
        '<td class="numc">' + (observed != null ? fmtP(observed) : "—") + "</td>" +
        '<td class="numc">' + (p.n_certified != null ? p.n_certified : "—") + "</td>" +
        '<td class="numc">' + (p.n_violations != null ? p.n_violations : "—") + "</td>" +
        "<td>" + verdict + "</td></tr>";
    }
    box.innerHTML = head +
      '<div class="table-scroll"><table class="cands"><thead><tr><th>target</th>' +
      '<th class="numc">claimed bound</th><th class="numc">observed rate</th>' +
      '<th class="numc">n certified</th><th class="numc">violations</th><th>verdict</th></tr></thead><tbody>' +
      rows + "</tbody></table></div>";
    // per-unit rows (expandable)
    const perUnit = v.rows || [];
    if (perUnit.length) {
      const det = document.createElement("details");
      det.className = "lq-details";
      det.innerHTML = '<summary>' + perUnit.length + " per-unit predictions</summary>" +
        '<div class="table-scroll"><table class="cands"><thead><tr><th>work</th><th>model</th>' +
        '<th class="numc">pred severity</th><th>truly bad</th></tr></thead><tbody>' +
        perUnit.map((r) =>
          "<tr><td>" + escapeHtml(String(r.work || "")) + "</td>" +
          "<td>" + escapeHtml(String(r.model || "")) + "</td>" +
          '<td class="numc">' + fmtP(r.pred) + "</td>" +
          "<td>" + (r.bad ? '<span class="viol-bad">bad</span>' : '<span class="viol-ok">ok</span>') + "</td></tr>"
        ).join("") + "</tbody></table></div>";
      box.appendChild(det);
    }
    wrap.appendChild(box);
  }
  const note = document.createElement("div");
  note.className = "explain";
  note.innerHTML = "The honesty check: on works the calibration never saw, is the actual bad-rate among " +
    "<i>certified</i> cases still under the promised bound? Green holds; a red row means the guarantee did not " +
    "transfer and that defect's gate must be re-fit before it is trusted.";
  wrap.appendChild(note);
  return wrap;
}

/* ------------------------------- splits ---------------------------------- */
const SPLIT_HINTS = {
  calibration: "fits the curves and picks every τ",
  fit: "fits the curves and picks every τ",
  heldout: "never touched during fitting — checks the bounds above",
  held_out: "never touched during fitting — checks the bounds above",
  regression: "frozen works re-run on every release to catch drift",
  dev: "development material, outside the formal calibration/held-out split",
};

function splitsBlock(cal) {
  const splits = cal.splits || {};
  const groups = {};
  for (const [work, split] of Object.entries(splits)) (groups[split] = groups[split] || []).push(work);
  const names = Object.keys(groups);
  if (!names.length) return strDiv("no split membership recorded");
  const order = ["calibration", "fit", "heldout", "held_out", "regression", "dev"];
  names.sort((a, b) => (order.indexOf(a) + 99) - (order.indexOf(b) + 99) || a.localeCompare(b));
  let html = "";
  for (const g of names) {
    const works = groups[g].slice().sort();
    html += '<div class="split-group"><div class="sg-name">' + escapeHtml(g) +
      ' <span class="axis-hint">(' + works.length + " work" + (works.length === 1 ? "" : "s") + ")</span>" +
      (SPLIT_HINTS[g] ? '<span class="sg-hint">' + SPLIT_HINTS[g] + "</span>" : "") + "</div>" +
      works.map((w) => '<span class="split-chip">' + escapeHtml(w) + "</span>").join("") + "</div>";
  }
  html += '<div class="explain">Splits are per-<i>work</i>, never per-excerpt — excerpts of one recording are ' +
    "correlated, so mixing them across splits would leak and fake the guarantee.</div>";
  return html;
}

/* ---------------------- in-domain reselection (v3) ------------------------ */
function reselectBlock(cal) {
  const rv = cal.reselect_v3;
  if (!rv || !rv.decisions || !Object.keys(rv.decisions).length) return null;
  const wrap = document.createElement("div");
  const passed = Object.keys(rv.taus_passed || {});
  wrap.insertAdjacentHTML("beforeend",
    '<div class="card-desc" style="margin-top:0">Certified τ applied to live tracks: gates <b>' +
    (passed.length ? passed.map(escapeHtml).join(", ") : "none") + "</b>; secondary objective = " +
    escapeHtml(rv.secondary_def || "—") + (rv.note ? " · <i>" + escapeHtml(rv.note) + "</i>" : "") + "</div>");
  for (const [track, dec] of Object.entries(rv.decisions)) {
    const box = document.createElement("div");
    box.className = "viol-defect";
    const cands = dec.candidates || {};
    const chip = dec.certification
      ? '<span class="cal-badge yes">' + escapeHtml(dec.certification) + "</span>"
      : "";
    const dist = dec.distribution ? '<span class="split-chip">' + escapeHtml(dec.distribution) + "</span>" : "";
    let rows = "";
    for (const [model, m] of Object.entries(cands)) {
      const infeasible = (m.infeasible_defects || []);
      const hardFailed = (m.hard_failed || []);
      const ok = !infeasible.length && !hardFailed.length;
      const ucbs = Object.entries(m.ucbs || {})
        .map(([k, val]) => escapeHtml(k) + " " + fmtP(val, 2)).join(" · ");
      rows += "<tr" + (ok ? ' class="winner"' : "") + "><td>" + escapeHtml(model) + "</td>" +
        '<td style="font-family:var(--mono);font-size:11px;">' + (ucbs || "—") + "</td>" +
        "<td>" + (infeasible.length
          ? infeasible.map((x) => '<span class="cal-badge no">' + escapeHtml(x) + "</span>").join(" ")
          : '<span class="viol-ok">feasible ✓</span>') + "</td>" +
        '<td class="numc">' + fmtP(m.secondary, 3) + "</td></tr>";
    }
    box.innerHTML = '<div class="viol-head"><b>' + escapeHtml(track) + "</b> " + chip + " " + dist +
      (dec.reason ? ' <span class="muted">' + escapeHtml(dec.reason) + "</span>" : "") + "</div>" +
      '<div class="table-scroll"><table class="cands"><thead><tr><th>model</th>' +
      '<th>defect risk UCBs</th><th>feasibility</th><th class="numc">secondary</th></tr></thead><tbody>' +
      rows + "</tbody></table></div>";
    wrap.appendChild(box);
  }
  return wrap;
}

/* ------------------------------ truth ------------------------------------ */
function truthBlock(cal) {
  const td = cal.truth_definitions || {};
  const keys = Object.keys(td);
  if (!keys.length) return null;
  let html = '<div class="kv-grid">';
  for (const k of keys) html += '<div class="k">' + escapeHtml(k) + '</div><div class="v">' + escapeHtml(String(td[k])) + "</div>";
  return html + "</div>";
}

/* -------------------------------- page ----------------------------------- */
function strDiv(msg) { return '<div class="muted">' + escapeHtml(msg) + "</div>"; }

function card(title, kind, desc, bodyNode) {
  const c = document.createElement("div");
  c.className = "card";
  c.innerHTML = '<div class="card-head"><span class="name">' + escapeHtml(title) + "</span>" +
    (kind ? '<span class="kind">' + escapeHtml(kind) + "</span>" : "") + "</div>" +
    (desc ? '<div class="card-desc">' + desc + "</div>" : "");
  if (bodyNode == null) return c;
  if (typeof bodyNode === "string") c.insertAdjacentHTML("beforeend", bodyNode);
  else c.appendChild(bodyNode);
  return c;
}

async function load() {
  const body = document.getElementById("calBody");
  let payload;
  try {
    payload = await fetchJSON("/api/v2/calibration");
  } catch (e) {
    body.innerHTML = '<div class="notice">failed to load calibration: ' + escapeHtml(e.message) + "</div>";
    return;
  }
  if (!payload.available) {
    body.innerHTML = '<div class="notice">No calibration artifact found under this lib root ' +
      "(<code>calibration/calibration_v1.json</code>). The autonomous selector then runs with hard " +
      "gates only. Seed a synthetic one for development:<br><code>uv run python " +
      "web/dev_seed_calibration.py --lib &lt;root&gt;</code></div>";
    document.getElementById("calMeta").textContent = "no calibration_v1.json under the lib root";
    return;
  }
  const cal = payload.calibration || {};
  C.cal = cal;
  const nCert = Object.keys(cal.defects || {}).filter((n) => certTargets(cal.defects[n]).length).length;
  document.getElementById("calMeta").textContent =
    payload.path + " · " + (cal.schema || "calibration/v1") +
    " · δ=" + fmtP(cal.delta, 2) + (cal.uncertainty_u != null ? " · u=" + fmtP(cal.uncertainty_u, 2) : "") +
    " · " + nCert + " / " + Object.keys(cal.defects || {}).length + " defects certifiable";

  body.innerHTML = "";
  body.appendChild(card(
    "Severity maps", "raw metric → severity 0–1",
    "One curve per defect: how a raw measurement is translated into calibrated damage, monotone by " +
    "construction (pool-adjacent-violators isotonic fit). The dashed line is that defect's certification " +
    "threshold τ; an uncertified defect is shown honestly with no τ.",
    severityCurves(cal)));
  body.appendChild(card(
    "Certification thresholds (τ)", "learn-then-test",
    "For each defect: the strongest certified threshold, its Clopper–Pearson risk upper bound, the number of " +
    "work×model groups behind it, and whether certification succeeds at a 10 / 15 / 20% target risk.",
    tauTable(cal)));
  body.appendChild(card(
    "Out-of-sample violations", "held-out split",
    "Claimed bound vs the rate actually observed on held-out works, per target. Green holds, red is a broken " +
    "promise. Expand any defect for its per-unit predictions.",
    violationBlock(cal)));
  const reselect = reselectBlock(cal);
  if (reselect) body.appendChild(card(
    "In-domain reselection", "τ applied to live tracks",
    "The certified τ feeding real selection decisions: each candidate's per-defect risk UCBs, whether it is " +
    "feasible, and its secondary-objective score.",
    reselect));
  body.appendChild(card(
    "Split membership", "per work",
    "Which recording went into which split.",
    splitsBlock(cal)));
  const truth = truthBlock(cal);
  if (truth) body.appendChild(card(
    "Truth definitions", "what counts as damage",
    "The ground-truth label and deployable proxy behind each defect's calibration data.",
    truth));
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refreshBtn").addEventListener("click", load);
  load();
});
