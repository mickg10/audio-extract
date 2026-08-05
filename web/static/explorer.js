/* Per-file explorer.
 *   Tab 1 "Variants": one card per manifest entry (player + canvas waveform +
 *   spectrogram + metrics + download), grouped per SCHEMA order, appended
 *   incrementally as the batch writes more.
 *   Tab 2 "Analysis": pick A/B/C, one transport drives all three in sync,
 *   spectrograms side by side, precomputed diff variants as first-class picks,
 *   plus a cheap in-browser A-vs-B overlaid-envelope view.
 */
(function () {
  const slug = decodeURIComponent(
    location.pathname.replace(/^\/f\//, "").replace(/\/+$/, "")
  );

  // ---- state ----------------------------------------------------------
  let data = null;
  const byId = {};
  const cardNode = {};       // id -> card element (variants tab)
  const cardComplete = {};   // id -> was audio ready when rendered
  const vWave = {};          // id -> Waveform (variants tab)
  const vAudio = {};         // id -> <audio> (variants tab)
  let activeAudioId = null;  // last-played, for the space bar
  let currentTab = "variants";
  const groupBody = {};      // group idx -> body element
  const peaksCache = {};     // id -> {min,max}

  // ---- analysis state -------------------------------------------------
  const AB = { A: null, B: null, C: null };
  const abAudio = {};        // slot -> audio
  const abWave = {};         // slot -> Waveform
  const abCol = {};          // slot -> host element
  let overlay = null;        // OverlayWave
  let masterPlaying = false;
  let masterT = 0;
  let raf = null;
  let sliderDragging = false;

  // ---- dom refs -------------------------------------------------------
  const el = (id) => document.getElementById(id);
  const variantsTab = el("variantsTab");
  const analysisTab = el("analysisTab");

  // =====================================================================
  //  Load + top-level render
  // =====================================================================
  async function load(initial) {
    let payload;
    try {
      payload = await fetchJSON("/api/file/" + encodeURIComponent(slug));
    } catch (e) {
      if (initial) {
        variantsTab.innerHTML =
          '<div class="notice">Could not load this file: ' + escapeHtml(e.message) +
          '. <a href="/">Back to library</a></div>';
      }
      return;
    }
    data = payload;
    for (const v of data.variants) byId[v.id] = v;

    el("fileTitle").textContent = data.title || slug;
    el("fileTitle").title = data.title || slug;
    document.title = (data.title || slug) + " · Stem Library";

    const bits = [];
    bits.push(fmtDur(data.duration_s) + " duration");
    bits.push(data.n_variants + " variant" + (data.n_variants === 1 ? "" : "s"));
    if (!data.has_manifest) bits.push("manifest pending — showing original only");
    el("fileSub").textContent = bits.join("  ·  ");
    el("fileMeta").innerHTML = data.has_manifest
      ? data.variants.length + " entries"
      : '<span class="spin"></span> processing';

    renderVariants();
    refreshAnalysisOptions();

    // Deep-link support: /f/<slug>#analysis opens the comparison tab directly.
    if (initial && location.hash === "#analysis") switchTab("analysis");
  }

  // =====================================================================
  //  Variants tab
  // =====================================================================
  const GROUP_ORDER = [0, 1, 2, 3, 4];
  const GROUP_TITLE = {
    0: "Original", 1: "Instrumentals", 2: "Vocals", 3: "Other stems", 4: "Diffs — what was removed",
  };

  function ensureSkeleton() {
    if (variantsTab.dataset.ready) return;
    variantsTab.innerHTML = "";
    for (const g of GROUP_ORDER) {
      const sec = document.createElement("div");
      sec.className = "group-section hidden";
      sec.dataset.group = g;
      const h = document.createElement("div");
      h.className = "group-head";
      h.textContent = GROUP_TITLE[g];
      const body = document.createElement("div");
      sec.appendChild(h);
      sec.appendChild(body);
      variantsTab.appendChild(sec);
      groupBody[g] = body;
    }
    variantsTab.dataset.ready = "1";
  }

  function renderVariants() {
    ensureSkeleton();
    if (!data.variants.length) {
      variantsTab.innerHTML = '<div class="empty-state">No variants yet for this file.</div>';
      variantsTab.dataset.ready = "";
      return;
    }
    for (const v of data.variants) {
      const wasComplete = cardComplete[v.id];
      if (cardNode[v.id] && wasComplete) continue;          // already fully rendered
      const node = buildCard(v);
      if (cardNode[v.id]) {
        cardNode[v.id].replaceWith(node);                   // upgrade placeholder
      } else {
        groupBody[v.group].appendChild(node);
      }
      cardNode[v.id] = node;
      cardComplete[v.id] = v.audio_ready;
      const sec = variantsTab.querySelector('.group-section[data-group="' + v.group + '"]');
      if (sec) sec.classList.remove("hidden");
    }
    if (!activeAudioId) {
      const first = data.variants.find((v) => v.audio_ready);
      if (first) activeAudioId = first.id;
    }
  }

  function metricsHtml(v) {
    const m = [];
    const push = (lbl, val) => m.push('<span class="metric"><span class="lbl">' + lbl + "</span> <b>" + val + "</b></span>");
    push("dur", fmtDur(v.duration_s));
    if (v.peak != null) push("peak", fmtNum(v.peak, 3));
    if (v.rms != null) push("rms", fmtNum(v.rms, 3));
    if (v.lufs_approx != null) push("lufs", fmtNum(v.lufs_approx, 1));
    if (v.sr) push("sr", (v.sr / 1000) + "k");
    if (v.channels) push("ch", v.channels);
    return m.join("");
  }

  function buildCard(v) {
    const card = document.createElement("div");
    card.className = "card" + (v.group === 4 ? " diff" : v.group === 0 ? " source" : "");
    card.id = "card-" + v.id;

    const head = document.createElement("div");
    head.className = "card-head";
    head.innerHTML =
      '<span class="name">' + escapeHtml(v.id) + "</span>" +
      '<span class="kind">' + escapeHtml(v.kind || "?") + " · " + escapeHtml(v.stem || "") + "</span>";
    card.appendChild(head);

    if (v.description) {
      const d = document.createElement("div");
      d.className = "card-desc";
      d.textContent = v.description;
      card.appendChild(d);
    }

    const met = document.createElement("div");
    met.className = "metrics";
    met.innerHTML = metricsHtml(v);
    card.appendChild(met);

    // ---- player + waveform ----
    const row = document.createElement("div");
    row.className = "player-row";
    let audio = null;
    if (v.audio_ready) {
      audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = v.audio_url;
      vAudio[v.id] = audio;
      row.appendChild(audio);
    } else {
      const note = document.createElement("span");
      note.className = "metric";
      note.innerHTML = '<span class="spin"></span> audio pending';
      row.appendChild(note);
    }
    card.appendChild(row);

    let wf = null;
    if (v.peaks_ready) {
      const scroll = document.createElement("div");
      scroll.className = "wave-scroll";
      card.appendChild(scroll);
      wf = new Waveform(scroll, {
        duration: v.duration_s || 0,
        color: v.group === 4 ? getCSS("--accent-3") : v.group === 0 ? getCSS("--accent-2") : getCSS("--wave"),
        onSeek: (t) => { if (audio) { audio.currentTime = t; } wf.setPlayhead(t); },
      });
      wf.load(v.peaks_url);
      vWave[v.id] = wf;
      row.appendChild(makeZoomControls(wf));
    } else {
      const ph = document.createElement("div");
      ph.className = "wave-empty";
      ph.textContent = "waveform pending…";
      card.appendChild(ph);
    }

    // ---- audio<->waveform wiring ----
    if (audio) {
      audio.addEventListener("play", () => { activeAudioId = v.id; });
      const sync = () => { if (wf) wf.setPlayhead(audio.currentTime); };
      audio.addEventListener("timeupdate", sync);
      audio.addEventListener("seeking", sync);
      audio.addEventListener("loadedmetadata", () => {
        if (isFinite(audio.duration) && audio.duration > 0 && wf) {
          wf.duration = audio.duration; wf.render();
        }
      });
    }

    // ---- spectrogram ----
    const spec = document.createElement("div");
    spec.className = "spec";
    if (v.spec_ready) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "spectrogram of " + v.id;
      img.src = v.spectrogram_url;
      spec.appendChild(img);
    } else {
      spec.innerHTML = '<div class="spec-missing">spectrogram pending…</div>';
    }
    card.appendChild(spec);

    // ---- links ----
    if (v.download_url) {
      const links = document.createElement("div");
      links.className = "links";
      var _ext = ((v.stream || v.output || "").split(".").pop() || "").toLowerCase();
      links.innerHTML = '<a href="' + v.download_url + '">↓ Download' +
        (_ext ? " (" + escapeHtml(_ext) + ")" : "") + "</a>";
      if (v.lossless_url) {
        links.innerHTML += ' <a href="' + v.lossless_url + '" class="lossless">lossless wav</a>';
      }
      card.appendChild(links);
    }
    return card;
  }

  // =====================================================================
  //  Analysis tab
  // =====================================================================
  function optLabel(v) {
    const tag = v.group === 4 ? "△ " : v.group === 0 ? "◎ " : "";
    return tag + v.id + (v.stem ? "  (" + v.stem + ")" : "");
  }

  function fillSelect(sel, selectedId) {
    const prev = selectedId != null ? selectedId : sel.value;
    let html = '<option value="">— none —</option>';
    for (const v of data.variants) {
      html += '<option value="' + escapeHtml(v.id) + '">' + escapeHtml(optLabel(v)) + "</option>";
    }
    sel.innerHTML = html;
    if (prev && byId[prev]) sel.value = prev;
  }

  function pickDefault(kindTest, fallbackIdx) {
    const found = data.variants.find(kindTest);
    if (found) return found.id;
    return data.variants[fallbackIdx] ? data.variants[fallbackIdx].id : "";
  }

  let analysisBuilt = false;
  function refreshAnalysisOptions() {
    if (!data.variants.length) return;
    const selA = el("pickA"), selB = el("pickB"), selC = el("pickC");

    if (!analysisBuilt) {
      // sensible defaults for the "subtraction" story
      AB.A = pickDefault((v) => v.id === "original" || v.group === 0, 0);
      AB.B = pickDefault((v) => v.group === 1 && /ensemble/i.test(v.id), 0) ||
             pickDefault((v) => v.group === 1, 1);
      AB.C = pickDefault((v) => v.id === "residual_voice", 0) ||
             pickDefault((v) => v.group === 4, 2);
    }
    fillSelect(selA, AB.A);
    fillSelect(selB, AB.B);
    fillSelect(selC, AB.C);

    if (!analysisBuilt) {
      selA.addEventListener("change", () => setSlot("A", selA.value));
      selB.addEventListener("change", () => setSlot("B", selB.value));
      selC.addEventListener("change", () => setSlot("C", selC.value));
      buildPresets();
      buildColumns();
      wireTransport();
      analysisBuilt = true;
    } else {
      buildPresets(); // new diffs may have appeared
    }
  }

  function buildPresets() {
    const bar = el("presetBar");
    bar.innerHTML = "";
    const add = (label, a, b, c) => {
      const btn = document.createElement("button");
      btn.className = "btn small";
      btn.textContent = label;
      btn.addEventListener("click", () => applyPreset(a, b, c));
      bar.appendChild(btn);
    };
    // Precomputed diffs -> first-class "what was removed" comparisons.
    const diffs = data.variants.filter((v) => v.group === 4 && Array.isArray(v.input) && v.input.length === 2);
    let n = 0;
    for (const d of diffs) {
      const [pre, post] = d.input;
      if (byId[pre] && byId[post]) { add("△ " + d.id, pre, post, d.id); if (++n >= 5) break; }
    }
    if (!diffs.length) {
      const inst = data.variants.find((v) => v.group === 1);
      const voc = data.variants.find((v) => v.group === 2);
      if (inst) add("orig / instr / vocals", "original", inst.id, voc ? voc.id : "");
    }
  }

  function applyPreset(a, b, c) {
    setSlot("A", a || ""); setSlot("B", b || ""); setSlot("C", c || "");
    el("pickA").value = a || ""; el("pickB").value = b || ""; el("pickC").value = c || "";
  }

  function buildColumns() {
    const grid = el("abGrid");
    grid.innerHTML = "";
    for (const slot of ["A", "B", "C"]) {
      const host = document.createElement("div");
      grid.appendChild(host);
      abCol[slot] = host;
      renderSlot(slot, AB[slot]);
    }
  }

  const SLOT_AUDIBLE_DEFAULT = { A: true, B: false, C: false };

  function renderSlot(slot, id) {
    AB[slot] = id || null;
    const host = abCol[slot];
    if (abWave[slot]) { abWave[slot].destroy(); abWave[slot] = null; }
    abAudio[slot] = null;
    host.innerHTML = "";

    const col = document.createElement("div");
    if (!id || !byId[id]) {
      col.className = "ab-col empty";
      col.innerHTML = '<div class="slot-tag">' + slot + "</div><div>— empty —</div>";
      host.appendChild(col);
      recomputeMaster();
      updateOverlay();
      return;
    }
    const v = byId[id];
    col.className = "ab-col";

    const head = document.createElement("div");
    head.className = "card-head";
    head.innerHTML = '<span class="slot-tag">' + slot + "</span><span class=\"name\">" + escapeHtml(v.id) + "</span>";
    const audible = document.createElement("button");
    audible.className = "btn small";
    audible.style.marginLeft = "auto";
    const startAudible = SLOT_AUDIBLE_DEFAULT[slot];
    audible.textContent = startAudible ? "🔊" : "🔇";
    audible.title = "Toggle audible (kept in sync either way)";
    head.appendChild(audible);
    col.appendChild(head);

    const desc = document.createElement("div");
    desc.className = "card-desc";
    desc.textContent = (v.stem ? v.stem + " · " : "") + (v.description || "");
    col.appendChild(desc);

    const met = document.createElement("div");
    met.className = "metrics";
    met.innerHTML = metricsHtml(v);
    col.appendChild(met);

    let audio = null;
    if (v.audio_ready) {
      audio = document.createElement("audio");
      audio.preload = "auto";
      audio.src = v.audio_url;
      audio.muted = !startAudible;
      audio.style.display = "none";
      col.appendChild(audio);
      abAudio[slot] = audio;
      audio.addEventListener("loadedmetadata", () => {
        if (isFinite(audio.duration) && abWave[slot]) { abWave[slot].duration = audio.duration; abWave[slot].render(); }
        recomputeMaster();
      });
    } else {
      const note = document.createElement("div");
      note.className = "metric";
      note.innerHTML = '<span class="spin"></span> audio pending';
      col.appendChild(note);
    }
    audible.addEventListener("click", () => {
      if (!audio) return;
      audio.muted = !audio.muted;
      audible.textContent = audio.muted ? "🔇" : "🔊";
    });

    const scroll = document.createElement("div");
    scroll.className = "wave-scroll";
    col.appendChild(scroll);
    const wf = new Waveform(scroll, {
      duration: v.duration_s || 0,
      color: v.group === 4 ? getCSS("--accent-3") : v.group === 0 ? getCSS("--accent-2") : getCSS("--wave"),
      onSeek: (t) => seekMaster(t),
    });
    if (v.peaks_ready) wf.load(v.peaks_url);
    abWave[slot] = wf;

    const zrow = document.createElement("div");
    zrow.className = "player-row";
    zrow.appendChild(makeZoomControls(wf));
    col.appendChild(zrow);

    const spec = document.createElement("div");
    spec.className = "spec";
    if (v.spec_ready) {
      spec.innerHTML = '<img loading="lazy" alt="spectrogram" src="' + v.spectrogram_url + '">';
    } else {
      spec.innerHTML = '<div class="spec-missing">spectrogram pending…</div>';
    }
    col.appendChild(spec);

    if (v.download_url) {
      const links = document.createElement("div");
      links.className = "links";
      links.innerHTML = '<a href="' + v.download_url + '">↓ Download</a>';
      col.appendChild(links);
    }

    host.appendChild(col);
    recomputeMaster();
    if (slot === "A" || slot === "B") updateOverlay();
  }

  function setSlot(slot, id) {
    const wasPlaying = masterPlaying;
    if (masterPlaying) pauseMaster();
    renderSlot(slot, id);
    if (wasPlaying) playMaster();
  }

  // ---- synchronized transport ----
  function activeAbAudios() {
    return ["A", "B", "C"].map((s) => abAudio[s]).filter((a) => a);
  }
  function masterDuration() {
    let d = 0;
    for (const s of ["A", "B", "C"]) {
      const v = AB[s] && byId[AB[s]];
      if (v && v.duration_s) d = Math.max(d, v.duration_s);
      const a = abAudio[s];
      if (a && isFinite(a.duration)) d = Math.max(d, a.duration);
    }
    return d;
  }
  function recomputeMaster() {
    const d = masterDuration();
    el("masterTime").textContent = fmtTime(masterT) + " / " + fmtTime(d);
  }

  function wireTransport() {
    el("masterPlay").addEventListener("click", () => (masterPlaying ? pauseMaster() : playMaster()));
    const seek = el("masterSeek");
    const fromSlider = () => {
      const d = masterDuration() || 1;
      seekMaster((seek.value / 1000) * d);
    };
    seek.addEventListener("input", () => { sliderDragging = true; fromSlider(); });
    seek.addEventListener("change", () => { sliderDragging = false; fromSlider(); });
  }

  function playMaster() {
    const auds = activeAbAudios();
    if (!auds.length) return;
    for (const a of auds) { try { a.currentTime = masterT; } catch (e) {} a.play().catch(() => {}); }
    masterPlaying = true;
    el("masterPlay").textContent = "❚❚ Pause all";
    tick();
  }
  function pauseMaster() {
    for (const a of activeAbAudios()) a.pause();
    masterPlaying = false;
    el("masterPlay").textContent = "▶ Play all";
    if (raf) cancelAnimationFrame(raf), (raf = null);
  }
  function seekMaster(t) {
    const d = masterDuration();
    masterT = clamp(t, 0, d || t);
    for (const a of activeAbAudios()) { try { a.currentTime = masterT; } catch (e) {} }
    paintPlayheads();
    updateSliderUI();
  }
  function tick() {
    const auds = activeAbAudios();
    const clock = auds.find((a) => !a.paused) || auds[0];
    if (clock) {
      masterT = clock.currentTime;
      // gently resync the rest to the clock
      for (const a of auds) if (a !== clock && Math.abs(a.currentTime - masterT) > 0.18) {
        try { a.currentTime = masterT; } catch (e) {}
      }
      if (clock.ended) { pauseMaster(); masterT = 0; seekMaster(0); return; }
    }
    paintPlayheads();
    if (!sliderDragging) updateSliderUI();
    if (masterPlaying) raf = requestAnimationFrame(tick);
  }
  function paintPlayheads() {
    for (const s of ["A", "B", "C"]) if (abWave[s]) abWave[s].setPlayhead(masterT);
    if (overlay) overlay.setPlayhead(masterT);
  }
  function updateSliderUI() {
    const d = masterDuration() || 1;
    el("masterSeek").value = Math.round((masterT / d) * 1000);
    el("masterTime").textContent = fmtTime(masterT) + " / " + fmtTime(d);
  }

  // ---- A vs B overlay (cheap in-browser difference view) ----
  async function getPeaks(id) {
    if (!id || !byId[id] || !byId[id].peaks_ready) return null;
    if (peaksCache[id]) return peaksCache[id];
    try {
      const p = await fetchJSON(byId[id].peaks_url);
      if (p && Array.isArray(p.min) && Array.isArray(p.max)) { peaksCache[id] = p; return p; }
    } catch (e) {}
    return null;
  }
  async function updateOverlay() {
    const wrap = el("overlayWrap");
    el("legA").textContent = "A" + (AB.A ? " · " + AB.A : "");
    el("legB").textContent = "B" + (AB.B ? " · " + AB.B : "");
    if (!overlay) overlay = new OverlayWave(wrap);
    const [pa, pb] = await Promise.all([getPeaks(AB.A), getPeaks(AB.B)]);
    overlay.duration = masterDuration();
    overlay.setSeries(pa, pb);
  }

  // =====================================================================
  //  Tabs + keyboard
  // =====================================================================
  function switchTab(name) {
    if (name === currentTab) return;
    // avoid two tabs playing at once
    pauseMaster();
    for (const id in vAudio) if (vAudio[id]) vAudio[id].pause();
    currentTab = name;
    el("tabVariants").classList.toggle("active", name === "variants");
    el("tabAnalysis").classList.toggle("active", name === "analysis");
    variantsTab.classList.toggle("hidden", name !== "variants");
    analysisTab.classList.toggle("hidden", name !== "analysis");
    try {
      history.replaceState(null, "", name === "analysis" ? "#analysis" : location.pathname);
    } catch (e) {}
    if (name === "analysis") {
      // canvases need a real width; (re)render now that they're visible
      for (const s of ["A", "B", "C"]) if (abWave[s]) abWave[s].render();
      if (overlay) overlay.render();
    }
  }
  el("tabVariants").addEventListener("click", () => switchTab("variants"));
  el("tabAnalysis").addEventListener("click", () => switchTab("analysis"));

  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea" || e.target.isContentEditable) return;
    if (e.code === "Space" || e.key === " ") {
      e.preventDefault();
      if (currentTab === "analysis") { masterPlaying ? pauseMaster() : playMaster(); }
      else {
        const a = vAudio[activeAudioId] || Object.values(vAudio)[0];
        if (a) { a.paused ? a.play().catch(() => {}) : a.pause(); }
      }
    } else if (currentTab === "variants" && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
      const a = vAudio[activeAudioId] || Object.values(vAudio)[0];
      if (a && isFinite(a.duration)) {
        e.preventDefault();
        a.currentTime = clamp(a.currentTime + (e.key === "ArrowRight" ? 5 : -5), 0, a.duration);
      }
    }
  });

  el("refreshBtn").addEventListener("click", () => load(false));

  // =====================================================================
  //  init + incremental auto-refresh
  // =====================================================================
  load(true);
  setInterval(() => { if (!document.hidden) load(false); }, 6000);
})();

/* ------------------------------------------------------------------------
 * OverlayWave: two envelopes superimposed + their |A-B| divergence, with a
 * shared playhead. Responsive (fits container, redraws on resize).
 * ---------------------------------------------------------------------- */
class OverlayWave {
  constructor(container) {
    this.container = container;
    this.duration = 0;
    this.a = null; this.b = null;
    this.playT = 0;
    this.dpr = Math.max(1, window.devicePixelRatio || 1);
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    container.appendChild(this.canvas);
    this.playhead = document.createElement("div");
    this.playhead.className = "playhead";
    this.playhead.style.display = "none";
    container.appendChild(this.playhead);
    this._ro = new ResizeObserver(() => this.render());
    this._ro.observe(container);
  }
  setSeries(a, b) { this.a = a; this.b = b; this.render(); }
  setPlayhead(t) {
    this.playT = t || 0;
    if (!this.duration) { this.playhead.style.display = "none"; return; }
    const w = parseFloat(this.canvas.style.width) || this.container.clientWidth;
    this.playhead.style.display = "block";
    this.playhead.style.left = (Math.max(0, Math.min(1, this.playT / this.duration)) * w) + "px";
  }
  render() {
    const cssW = Math.max(120, this.container.clientWidth || 300);
    const cssH = this.container.clientHeight || 120;
    const dpr = this.dpr;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.canvas.style.width = cssW + "px";
    this.canvas.style.height = cssH + "px";
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const mid = cssH / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.beginPath(); ctx.moveTo(0, mid + 0.5); ctx.lineTo(cssW, mid + 0.5); ctx.stroke();

    if (!this.a && !this.b) {
      ctx.fillStyle = getCSS("--text-mute");
      ctx.font = "13px -apple-system, sans-serif";
      ctx.fillText("load peaks into slots A and B to compare", 12, mid);
      return;
    }
    const col = (varName, alpha) => {
      const c = getCSS(varName);
      return c || "#888";
    };
    // draw A then B translucent, then divergence line
    this._series(this.a, getCSS("--wave"), 0.55, cssW, cssH, mid);
    this._series(this.b, getCSS("--wave-b"), 0.55, cssW, cssH, mid);
    this._divergence(cssW, cssH, mid);
    this.setPlayhead(this.playT);
  }
  _env(peaks, x, cssW) {
    const n = Math.min(peaks.min.length, peaks.max.length);
    const b0 = Math.floor((x / cssW) * n);
    const b1 = Math.max(b0 + 1, Math.floor(((x + 1) / cssW) * n));
    let lo = 1, hi = -1;
    for (let b = b0; b < b1 && b < n; b++) { if (peaks.min[b] < lo) lo = peaks.min[b]; if (peaks.max[b] > hi) hi = peaks.max[b]; }
    if (hi < lo) { lo = 0; hi = 0; }
    return [lo, hi];
  }
  _series(peaks, color, alpha, cssW, cssH, mid) {
    if (!peaks) return;
    const ctx = this.ctx;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let x = 0; x < cssW; x++) {
      const [lo, hi] = this._env(peaks, x, cssW);
      ctx.moveTo(x + 0.5, mid - hi * (mid - 1));
      ctx.lineTo(x + 0.5, mid - lo * (mid - 1));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  _divergence(cssW, cssH, mid) {
    if (!this.a || !this.b) return;
    const ctx = this.ctx;
    ctx.strokeStyle = getCSS("--accent-3");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = 0; x < cssW; x++) {
      const [la, ha] = this._env(this.a, x, cssW);
      const [lb, hb] = this._env(this.b, x, cssW);
      const d = Math.max(Math.abs(ha - hb), Math.abs(la - lb)); // 0..2
      const y = cssH - 1 - (d / 2) * (cssH - 2);
      if (x === 0) ctx.moveTo(x + 0.5, y); else ctx.lineTo(x + 0.5, y);
    }
    ctx.stroke();
    ctx.lineWidth = 1;
  }
}
