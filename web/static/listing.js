/* Listing page: renders the file table from /api/index, auto-refreshes,
   and reflows to stacked cards on phones (CSS handles the reflow; we set
   data-label on each cell so the card view stays labelled). */
(function () {
  let files = [];
  let timer = null;

  const host = document.getElementById("tableHost");
  const genMeta = document.getElementById("genMeta");
  const countEl = document.getElementById("count");
  const searchEl = document.getElementById("search");
  const autoEl = document.getElementById("autoRefresh");
  const badge = document.getElementById("jobsBadge");

  async function refreshBadge() {
    try {
      const d = await fetchJSON("/api/jobs");
      const n = d.active || 0;
      if (n > 0) { badge.textContent = n; badge.classList.remove("hidden"); }
      else { badge.classList.add("hidden"); }
    } catch (e) { /* jobs endpoint unavailable -- ignore */ }
  }

  function statusPill(status) {
    const s = status || "pending";
    const label = s === "ready" ? "ready" : s === "processing" ? "processing…" : "pending";
    return '<span class="pill ' + s + '">' + label + "</span>";
  }

  function render() {
    const q = (searchEl.value || "").toLowerCase().trim();
    const shown = files.filter(f => !q || (f.title || f.slug).toLowerCase().includes(q));

    if (!files.length) {
      host.innerHTML = '<div class="empty-state">No files yet. The batch may still be starting — data appears here as it runs.</div>';
      return;
    }

    let rows = "";
    for (const f of shown) {
      const title = escapeHtml(f.title || f.slug);
      const dur = fmtDur(f.duration_s);
      const nv = f.n_variants == null ? "—" : f.n_variants;
      const dl = f.download_url
        ? '<a class="dl" href="' + f.download_url + '">↓ ' + escapeHtml(f.original || "original") + "</a>"
        : '<span class="dl" style="color:var(--text-mute)">—</span>';
      const open = '<a href="' + f.explorer_url + '">Explore →</a>';
      rows +=
        '<tr>' +
        '<td class="title" data-label="File"><a href="' + f.explorer_url + '">' + title + "</a></td>" +
        '<td class="num" data-label="Duration">' + dur + "</td>" +
        '<td class="num" data-label="Variants">' + nv + "</td>" +
        '<td data-label="Status">' + statusPill(f.status) + "</td>" +
        '<td data-label="Original">' + dl + "</td>" +
        '<td data-label="" style="text-align:right">' + open + "</td>" +
        "</tr>";
    }

    host.innerHTML =
      '<table class="files"><thead><tr>' +
      "<th>File</th><th style=\"text-align:right\">Duration</th><th style=\"text-align:right\">Variants</th>" +
      "<th>Status</th><th>Original</th><th></th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table>";

    countEl.textContent = shown.length + (q ? " / " + files.length : "") + " file" + (files.length === 1 ? "" : "s");
  }

  async function refresh() {
    try {
      const data = await fetchJSON("/api/index");
      files = data.files || [];
      const parts = [];
      if (data.generated) {
        parts.push("index " + new Date(data.generated * 1000).toLocaleString());
      }
      parts.push(files.length + " files");
      genMeta.textContent = parts.join("  ·  ");
      render();
      refreshBadge();
    } catch (e) {
      genMeta.textContent = "error loading index";
      if (!files.length) {
        host.innerHTML = '<div class="notice">Could not load <code>/api/index</code>: ' + escapeHtml(e.message) + "</div>";
      }
    }
  }

  function scheduleAuto() {
    if (timer) clearInterval(timer);
    if (autoEl.checked) timer = setInterval(refresh, 5000);
  }

  searchEl.addEventListener("input", render);
  document.getElementById("refreshBtn").addEventListener("click", refresh);
  autoEl.addEventListener("change", scheduleAuto);
  mountUploader(document.getElementById("uploader"), {
    onComplete: () => { refresh(); refreshBadge(); },
  });

  refresh();
  scheduleAuto();
})();
