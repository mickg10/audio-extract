/* charts.js — dependency-free inline-SVG chart helpers for the v2 GUI.
   No libraries; one validated palette (dark surface #0a0d12):
   categorical slots pass the CVD/lightness/contrast gates; magnitude uses a
   single-hue blue ramp (dark = low, recedes into the surface; bright = high);
   status colors are reserved for pass/fail meaning and always ship with a word,
   never color alone. */

const VIZ = {
  series: ["#3987e5", "#008300", "#d55181", "#c98500"],   // categorical slots 1-4
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
  ink: "#e6edf3",        // primary text
  ink2: "#9aa7b4",       // secondary
  muted: "#6b7684",      // axis labels
  grid: "#212837",       // hairline gridlines
  axis: "#39414f",       // baseline/axis
  surface: "#0a0d12",    // chart surface (gaps + rings are drawn in this)
  // sequential blue, low -> high (palette steps 700..100 reversed for dark mode)
  seq: ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
        "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"],
};

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs, children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v != null) el.setAttribute(k, v);
  }
  for (const c of children || []) el.appendChild(c);
  return el;
}
function svgText(x, y, str, attrs) {
  const t = svgEl("text", Object.assign({ x, y, fill: VIZ.ink2, "font-size": 10.5 }, attrs || {}));
  t.textContent = str;
  return t;
}

/* ------------------------------ scales ---------------------------------- */
function niceTicks(lo, hi, n) {
  if (!(isFinite(lo) && isFinite(hi))) return [];
  if (hi <= lo) hi = lo + 1;
  const span = hi - lo;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / Math.max(2, n))));
  let step = step0;
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (span / (step0 * m) <= n) { step = step0 * m; break; }
  }
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-6 ? 0 : +v.toFixed(10));
  }
  return out;
}
function fmtTick(v) {
  if (v == null || !isFinite(v)) return "";
  const a = Math.abs(v);
  if (a >= 1000) return (v / 1000) + "k";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return +v.toFixed(2) + "";
  if (a === 0) return "0";
  if (a < 0.001) return v.toExponential(0);
  return +v.toFixed(3) + "";
}

/* ------------------------------ tooltip --------------------------------- */
let _tipEl = null;
function _tip() {
  if (!_tipEl) {
    _tipEl = document.createElement("div");
    _tipEl.className = "viz-tip hidden";
    document.body.appendChild(_tipEl);
  }
  return _tipEl;
}
/* Attach the hover layer: any child with data-tip shows the shared tooltip. */
function bindTips(root) {
  const tip = _tip();
  root.addEventListener("pointermove", (e) => {
    const t = e.target.closest ? e.target.closest("[data-tip]") : null;
    if (!t) { tip.classList.add("hidden"); return; }
    tip.innerHTML = t.getAttribute("data-tip");
    tip.classList.remove("hidden");
    const pad = 12;
    let x = e.clientX + pad, y = e.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  });
  root.addEventListener("pointerleave", () => tip.classList.add("hidden"));
}

/* ------------------------- sequential color ----------------------------- */
function _hex2rgb(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}
function seqColor(t) {
  t = Math.max(0, Math.min(1, t));
  const seg = t * (VIZ.seq.length - 1);
  const i = Math.min(VIZ.seq.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = _hex2rgb(VIZ.seq[i]), b = _hex2rgb(VIZ.seq[i + 1]);
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return "rgb(" + c.join(",") + ")";
}
function seqInk(t) { return t >= 0.55 ? "#0b1c33" : VIZ.ink; }

/* A small horizontal ramp legend: [swatch gradient] lo — hi + caption. */
function seqLegend(lo, hi, caption, fmt) {
  fmt = fmt || fmtTick;
  const wrap = document.createElement("div");
  wrap.className = "viz-legend";
  const bar = document.createElement("span");
  bar.className = "viz-ramp";
  const stops = VIZ.seq.map((c, i) => c + " " + (i * 100 / (VIZ.seq.length - 1)).toFixed(0) + "%").join(",");
  bar.style.background = "linear-gradient(90deg," + stops + ")";
  const loEl = document.createElement("span"); loEl.className = "viz-leg-num"; loEl.textContent = fmt(lo);
  const hiEl = document.createElement("span"); hiEl.className = "viz-leg-num"; hiEl.textContent = fmt(hi);
  const cap = document.createElement("span"); cap.className = "viz-leg-cap"; cap.textContent = caption;
  wrap.append(loEl, bar, hiEl, cap);
  return wrap;
}

/* =========================================================================
 * svgLineChart — the severity-map curve (knots of an isotonic fit).
 *   opts: {xs, ys, w, h, color, xLabel, yLabel, yDomain, yRef:{value,label}}
 * ======================================================================= */
function svgLineChart(opts) {
  const w = opts.w || 320, h = opts.h || 190;
  const m = { l: 42, r: 12, t: 10, b: 34 };
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const xs = opts.xs || [], ys = opts.ys || [];
  const x0 = Math.min(...xs, 0), x1 = Math.max(...xs, 1e-9);
  const [y0, y1] = opts.yDomain || [0, 1];
  const X = (v) => m.l + (x1 === x0 ? 0 : (v - x0) / (x1 - x0)) * iw;
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ih;
  const svg = svgEl("svg", { viewBox: "0 0 " + w + " " + h, class: "viz-svg", role: "img" });

  // grid + ticks
  for (const ty of niceTicks(y0, y1, 4)) {
    svg.appendChild(svgEl("line", { x1: m.l, x2: w - m.r, y1: Y(ty), y2: Y(ty), stroke: VIZ.grid, "stroke-width": 1 }));
    svg.appendChild(svgText(m.l - 6, Y(ty) + 3.5, fmtTick(ty), { "text-anchor": "end", fill: VIZ.muted }));
  }
  for (const tx of niceTicks(x0, x1, 5)) {
    svg.appendChild(svgText(X(tx), h - m.b + 14, fmtTick(tx), { "text-anchor": "middle", fill: VIZ.muted }));
    svg.appendChild(svgEl("line", { x1: X(tx), x2: X(tx), y1: h - m.b, y2: h - m.b + 4, stroke: VIZ.axis, "stroke-width": 1 }));
  }
  svg.appendChild(svgEl("line", { x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: VIZ.axis, "stroke-width": 1 }));
  svg.appendChild(svgEl("line", { x1: m.l, x2: m.l, y1: m.t, y2: h - m.b, stroke: VIZ.axis, "stroke-width": 1 }));

  // reference line (e.g. the certification threshold τ) — labeled, dashed
  if (opts.yRef && isFinite(opts.yRef.value)) {
    const yv = Y(Math.max(y0, Math.min(y1, opts.yRef.value)));
    svg.appendChild(svgEl("line", { x1: m.l, x2: w - m.r, y1: yv, y2: yv, stroke: VIZ.critical, "stroke-width": 1.5, "stroke-dasharray": "5 4", opacity: 0.85 }));
    svg.appendChild(svgText(w - m.r, yv - 4, opts.yRef.label, { "text-anchor": "end", fill: VIZ.critical, "font-size": 9.5, "font-weight": 700 }));
  }

  // the curve + knots
  const color = opts.color || VIZ.series[0];
  if (xs.length) {
    const d = xs.map((v, i) => (i ? "L" : "M") + X(v).toFixed(1) + " " + Y(ys[i]).toFixed(1)).join(" ");
    svg.appendChild(svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    xs.forEach((v, i) => {
      svg.appendChild(svgEl("circle", { cx: X(v), cy: Y(ys[i]), r: 3.2, fill: color, stroke: VIZ.surface, "stroke-width": 2 }));
      svg.appendChild(svgEl("circle", {                                     // hover hit target
        cx: X(v), cy: Y(ys[i]), r: 10, fill: "transparent",
        "data-tip": "raw <b>" + fmtTick(v) + "</b> → severity <b>" + fmtTick(ys[i]) + "</b>",
      }));
    });
  }

  if (opts.xLabel) svg.appendChild(svgText(m.l + iw / 2, h - 4, opts.xLabel, { "text-anchor": "middle", fill: VIZ.muted, "font-size": 10 }));
  if (opts.yLabel) {
    const t = svgText(0, 0, opts.yLabel, { "text-anchor": "middle", fill: VIZ.muted, "font-size": 10 });
    t.setAttribute("transform", "translate(11 " + (m.t + ih / 2) + ") rotate(-90)");
    svg.appendChild(t);
  }
  bindTips(svg);
  return svg;
}

/* =========================================================================
 * svgHeatmap — case × model matrix, sequential color, value printed in-cell.
 *   opts: {rows:[{id,label,sub}], cols:[{id,label,sub}], value(rowId,colId),
 *          tip(rowId,colId), fmt, higherBetter (color mapping only), unit}
 *   Returns {el, min, max}.
 * ======================================================================= */
function svgHeatmap(opts) {
  const rows = opts.rows || [], cols = opts.cols || [];
  const fmt = opts.fmt || fmtTick;
  const vals = [];
  for (const r of rows) for (const c of cols) {
    const v = opts.value(r.id, c.id);
    if (v != null && isFinite(v)) vals.push(v);
  }
  const lo = vals.length ? Math.min(...vals) : 0;
  const hi = vals.length ? Math.max(...vals) : 1;
  const cw = 86, ch = 34, gap = 2;                       // 2px surface gap between fills
  const labW = opts.labW || 148, topH = 40;
  const w = labW + cols.length * (cw + gap) + 8;
  const h = topH + rows.length * (ch + gap) + 6;
  const svg = svgEl("svg", { viewBox: "0 0 " + w + " " + h, class: "viz-svg viz-heat", role: "img" });
  svg.style.minWidth = Math.min(w, 900) + "px";

  cols.forEach((c, j) => {
    const cx = labW + j * (cw + gap) + cw / 2;
    svg.appendChild(svgText(cx, 14, c.label, { "text-anchor": "middle", fill: VIZ.ink2, "font-size": 10.5, "font-weight": 600 }));
    if (c.sub) svg.appendChild(svgText(cx, 27, c.sub, { "text-anchor": "middle", fill: VIZ.muted, "font-size": 9 }));
  });
  rows.forEach((r, i) => {
    const cy = topH + i * (ch + gap);
    svg.appendChild(svgText(labW - 8, cy + ch / 2 - (r.sub ? 3 : -3.5), r.label, { "text-anchor": "end", fill: VIZ.ink2, "font-size": 10.5 }));
    if (r.sub) svg.appendChild(svgText(labW - 8, cy + ch / 2 + 9.5, r.sub, { "text-anchor": "end", fill: VIZ.muted, "font-size": 9 }));
    cols.forEach((c, j) => {
      const x = labW + j * (cw + gap);
      const v = opts.value(r.id, c.id);
      if (v == null || !isFinite(v)) {
        svg.appendChild(svgEl("rect", { x, y: cy, width: cw, height: ch, rx: 3, fill: "none", stroke: VIZ.grid, "stroke-width": 1 }));
        svg.appendChild(svgText(x + cw / 2, cy + ch / 2 + 3.5, "—", { "text-anchor": "middle", fill: VIZ.muted }));
        return;
      }
      let t = hi === lo ? 0.5 : (v - lo) / (hi - lo);
      if (opts.higherBetter === false) t = 1 - t;        // bright always = better
      const cell = svgEl("rect", {
        x, y: cy, width: cw, height: ch, rx: 3, fill: seqColor(t),
        "data-tip": opts.tip ? opts.tip(r.id, c.id) : (r.label + " × " + c.label + ": <b>" + fmt(v) + "</b>"),
      });
      svg.appendChild(cell);
      svg.appendChild(svgText(x + cw / 2, cy + ch / 2 + 3.5, fmt(v),
        { "text-anchor": "middle", fill: seqInk(t), "font-size": 11, "font-weight": 600, "pointer-events": "none" }));
    });
  });
  bindTips(svg);
  return { el: svg, min: lo, max: hi };
}

/* =========================================================================
 * svgLogBars — horizontal bars on a log-10 axis with a labeled gate line.
 *   opts: {items:[{label, sub, value, tip, verdict:{word, kind}}],
 *          gate, gateLabel, xLabel, fmt}
 * ======================================================================= */
function svgLogBars(opts) {
  const items = opts.items || [];
  const fmt = opts.fmt || ((v) => (v == null ? "—" : v.toPrecision(3)));
  const labW = opts.labW || 150, valW = 118;
  const rowH = 30, gap = 2;
  const w = opts.w || 720;
  const m = { t: 20, b: 36 };
  const iw = w - labW - valW - 16;
  const h = m.t + items.length * (rowH + gap) + m.b;
  const vals = items.map((d) => d.value).filter((v) => v != null && v > 0 && isFinite(v));
  let lo = Math.min(...vals, opts.gate != null ? opts.gate : Infinity);
  let hi = Math.max(...vals, opts.gate != null ? opts.gate : -Infinity);
  if (!isFinite(lo) || lo <= 0) lo = 1e-3;
  if (!isFinite(hi) || hi <= 0) hi = 1;
  const dLo = Math.floor(Math.log10(lo)), dHi = Math.ceil(Math.log10(hi * 1.15));
  const X = (v) => labW + ((Math.log10(v) - dLo) / Math.max(1e-9, dHi - dLo)) * iw;
  const svg = svgEl("svg", { viewBox: "0 0 " + w + " " + h, class: "viz-svg", role: "img" });
  svg.style.minWidth = "560px";

  // decade gridlines + tick labels
  for (let d = dLo; d <= dHi; d++) {
    const x = X(Math.pow(10, d));
    svg.appendChild(svgEl("line", { x1: x, x2: x, y1: m.t - 4, y2: h - m.b, stroke: VIZ.grid, "stroke-width": 1 }));
    svg.appendChild(svgText(x, h - m.b + 14, (d >= 0 ? Math.pow(10, d) : "1e" + d), { "text-anchor": "middle", fill: VIZ.muted }));
  }
  svg.appendChild(svgEl("line", { x1: labW, x2: labW, y1: m.t - 4, y2: h - m.b, stroke: VIZ.axis, "stroke-width": 1 }));
  svg.appendChild(svgEl("line", { x1: labW, x2: w - valW, y1: h - m.b, y2: h - m.b, stroke: VIZ.axis, "stroke-width": 1 }));

  items.forEach((d, i) => {
    const y = m.t + i * (rowH + gap);
    const bh = Math.min(22, rowH - 8);
    const by = y + (rowH - bh) / 2;
    svg.appendChild(svgText(labW - 8, y + rowH / 2 - (d.sub ? 2 : -3.5), d.label, { "text-anchor": "end", fill: VIZ.ink2, "font-size": 10.5, "font-weight": 600 }));
    if (d.sub) svg.appendChild(svgText(labW - 8, y + rowH / 2 + 10, d.sub, { "text-anchor": "end", fill: VIZ.muted, "font-size": 9 }));
    if (d.value == null || !(d.value > 0)) {
      svg.appendChild(svgText(labW + 6, y + rowH / 2 + 3.5, "no data", { fill: VIZ.muted }));
      return;
    }
    const x1 = Math.max(labW + 2, X(d.value));
    // bar: square at the baseline, 4px rounded data-end
    const r = Math.min(4, (x1 - labW) / 2);
    const dPath = "M" + labW + " " + by + " H" + (x1 - r) + " a" + r + " " + r + " 0 0 1 " + r + " " + r +
      " V" + (by + bh - r) + " a" + r + " " + r + " 0 0 1 -" + r + " " + r + " H" + labW + " Z";
    svg.appendChild(svgEl("path", { d: dPath, fill: VIZ.series[0], "data-tip": d.tip || (d.label + ": <b>" + fmt(d.value) + "</b>") }));
    svg.appendChild(svgText(x1 + 6, y + rowH / 2 + 3.5, fmt(d.value), { fill: VIZ.ink, "font-size": 10.5, "font-weight": 600 }));
    if (d.verdict) {
      const col = d.verdict.kind === "ok" ? VIZ.good : (d.verdict.kind === "warn" ? VIZ.warning : VIZ.critical);
      svg.appendChild(svgText(w - 8, y + rowH / 2 + 3.5, d.verdict.word, { "text-anchor": "end", fill: col, "font-size": 10, "font-weight": 700 }));
    }
  });

  if (opts.gate != null && opts.gate > 0) {
    const gx = X(opts.gate);
    svg.appendChild(svgEl("line", { x1: gx, x2: gx, y1: m.t - 10, y2: h - m.b, stroke: VIZ.critical, "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
    svg.appendChild(svgText(Math.min(gx + 4, w - valW), m.t - 8, opts.gateLabel || ("gate " + fmt(opts.gate)), { fill: VIZ.critical, "font-size": 9.5, "font-weight": 700 }));
  }
  if (opts.xLabel) svg.appendChild(svgText(labW + iw / 2, h - 6, opts.xLabel, { "text-anchor": "middle", fill: VIZ.muted, "font-size": 10 }));
  bindTips(svg);
  return svg;
}

/* =========================================================================
 * svgRangeBars — per-model min · median · max on a shared linear axis.
 *   opts: {items:[{label, sub, min, med, max, tip}], xLabel, fmt}
 * ======================================================================= */
function svgRangeBars(opts) {
  const items = opts.items || [];
  const fmt = opts.fmt || fmtTick;
  const labW = opts.labW || 150, rowH = 30, gap = 2;
  const w = opts.w || 720, m = { t: 14, b: 36, r: 46 };
  const iw = w - labW - m.r;
  const h = m.t + items.length * (rowH + gap) + m.b;
  const all = [];
  for (const d of items) for (const v of [d.min, d.med, d.max]) if (v != null && isFinite(v)) all.push(v);
  let lo = all.length ? Math.min(...all) : 0, hi = all.length ? Math.max(...all) : 1;
  const pad = (hi - lo || 1) * 0.08;
  lo -= pad; hi += pad;
  const X = (v) => labW + ((v - lo) / (hi - lo)) * iw;
  const svg = svgEl("svg", { viewBox: "0 0 " + w + " " + h, class: "viz-svg", role: "img" });
  svg.style.minWidth = "560px";

  for (const tx of niceTicks(lo, hi, 6)) {
    svg.appendChild(svgEl("line", { x1: X(tx), x2: X(tx), y1: m.t - 2, y2: h - m.b, stroke: VIZ.grid, "stroke-width": 1 }));
    svg.appendChild(svgText(X(tx), h - m.b + 14, fmtTick(tx), { "text-anchor": "middle", fill: VIZ.muted }));
  }
  svg.appendChild(svgEl("line", { x1: labW, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: VIZ.axis, "stroke-width": 1 }));

  items.forEach((d, i) => {
    const cy = m.t + i * (rowH + gap) + rowH / 2;
    svg.appendChild(svgText(labW - 8, cy - (d.sub ? 2 : -3.5), d.label, { "text-anchor": "end", fill: VIZ.ink2, "font-size": 10.5, "font-weight": 600 }));
    if (d.sub) svg.appendChild(svgText(labW - 8, cy + 10, d.sub, { "text-anchor": "end", fill: VIZ.muted, "font-size": 9 }));
    if (d.min == null || d.max == null) {
      svg.appendChild(svgText(labW + 6, cy + 3.5, "no data", { fill: VIZ.muted }));
      return;
    }
    const x0 = X(d.min), x1 = X(d.max);
    svg.appendChild(svgEl("line", { x1: x0, x2: x1, y1: cy, y2: cy, stroke: VIZ.series[0], "stroke-width": 2, "stroke-linecap": "round" }));
    for (const xv of [x0, x1]) {
      svg.appendChild(svgEl("line", { x1: xv, x2: xv, y1: cy - 5, y2: cy + 5, stroke: VIZ.series[0], "stroke-width": 2 }));
    }
    if (d.med != null) {
      svg.appendChild(svgEl("circle", { cx: X(d.med), cy, r: 4.5, fill: VIZ.series[0], stroke: VIZ.surface, "stroke-width": 2 }));
    }
    svg.appendChild(svgEl("rect", {   // row-wide hover target
      x: labW, y: cy - rowH / 2, width: iw, height: rowH, fill: "transparent",
      "data-tip": d.tip || (d.label + ": min <b>" + fmt(d.min) + "</b> · median <b>" + fmt(d.med) + "</b> · max <b>" + fmt(d.max) + "</b>"),
    }));
    // selective direct labels: the extremes; the median lives in the tooltip
    svg.appendChild(svgText(x0 - 6, cy + 3.5, fmt(d.min), { "text-anchor": "end", fill: VIZ.muted, "font-size": 9.5 }));
    svg.appendChild(svgText(x1 + 6, cy + 3.5, fmt(d.max), { fill: VIZ.muted, "font-size": 9.5 }));
  });
  if (opts.xLabel) svg.appendChild(svgText(labW + iw / 2, h - 6, opts.xLabel, { "text-anchor": "middle", fill: VIZ.muted, "font-size": 10 }));
  bindTips(svg);
  return svg;
}
