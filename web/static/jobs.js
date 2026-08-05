/* Jobs page: live queue table (poll /api/jobs every 2s) with per-job
   Stop / Cancel / Remove actions, plus the uploader. Mobile-friendly:
   the table reflows to stacked cards (CSS) with data-labels set here. */
(function () {
  let jobs = [];
  let timer = null;
  const host = document.getElementById("tableHost");
  const meta = document.getElementById("jobsMeta");
  const autoEl = document.getElementById("autoRefresh");

  const PILL = {
    queued: "queued", running: "running", done: "done",
    failed: "failed", stopped: "stopped", canceled: "canceled",
  };
  const ACTIVE = new Set(["queued", "running"]);
  const TERMINAL = new Set(["done", "failed", "stopped", "canceled"]);

  function pill(status) {
    const s = PILL[status] ? status : "queued";
    const spin = s === "running" ? '<span class="spin"></span> ' : "";
    return '<span class="pill job-' + s + '">' + spin + s + "</span>";
  }

  function actionsFor(j) {
    const btns = [];
    if (j.status === "running")
      btns.push('<button class="btn small danger-btn" data-act="stop" data-id="' + j.id + '">Stop</button>');
    if (j.status === "queued")
      btns.push('<button class="btn small" data-act="cancel" data-id="' + j.id + '">Cancel</button>');
    if (TERMINAL.has(j.status))
      btns.push('<button class="btn small" data-act="remove" data-id="' + j.id + '">Remove</button>');
    if (j.status === "done" && j.slug)
      btns.push('<a class="btn small primary" href="/f/' + encodeURIComponent(j.slug) + '">Open →</a>');
    return btns.join(" ");
  }

  function progressCell(j) {
    const pct = Math.round((j.progress || 0) * 100);
    const phase = escapeHtml(j.phase || j.status || "");
    const err = j.error && (j.status === "failed")
      ? '<div class="job-err" title="' + escapeHtml(j.error) + '">' + escapeHtml(j.error) + "</div>" : "";
    return (
      '<div class="job-phase">' + phase + (ACTIVE.has(j.status) ? " · " + pct + "%" : "") + "</div>" +
      '<div class="progress"><div class="progress-bar ' + (j.status) + '" style="width:' + pct + '%"></div></div>' +
      err
    );
  }

  function render() {
    if (!jobs.length) {
      host.innerHTML = '<div class="empty-state">No jobs yet. Drop a file above to process it.</div>';
      return;
    }
    let rows = "";
    for (const j of jobs) {
      rows +=
        "<tr>" +
        '<td class="title" data-label="File">' + escapeHtml(j.filename) + "</td>" +
        '<td data-label="Status">' + pill(j.status) + "</td>" +
        '<td data-label="Progress" class="progress-td">' + progressCell(j) + "</td>" +
        '<td class="num" data-label="Elapsed">' + fmtElapsed(j.elapsed_s) + "</td>" +
        '<td data-label="" class="job-actions">' + actionsFor(j) + "</td>" +
        "</tr>";
    }
    host.innerHTML =
      '<table class="files jobs-table"><thead><tr>' +
      "<th>File</th><th>Status</th><th>Progress / phase</th>" +
      '<th style="text-align:right">Elapsed</th><th></th>' +
      "</tr></thead><tbody>" + rows + "</tbody></table>";
  }

  async function refresh() {
    try {
      const data = await fetchJSON("/api/jobs");
      jobs = data.jobs || [];
      const active = data.active || 0;
      meta.textContent = jobs.length + " job" + (jobs.length === 1 ? "" : "s") +
        (active ? "  ·  " + active + " active" : "");
      render();
    } catch (e) {
      meta.textContent = "error loading jobs";
    }
  }

  // Event delegation for action buttons.
  host.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const act = btn.getAttribute("data-act");
    const id = btn.getAttribute("data-id");
    btn.disabled = true;
    btn.textContent = act === "stop" ? "Stopping…" : act === "cancel" ? "Cancelling…" : "…";
    await postAction("/api/jobs/" + encodeURIComponent(id) + "/" + act);
    await refresh();
  });

  function scheduleAuto() {
    if (timer) clearInterval(timer);
    if (autoEl.checked) timer = setInterval(refresh, 2000);
  }

  document.getElementById("refreshBtn").addEventListener("click", refresh);
  autoEl.addEventListener("change", scheduleAuto);
  mountUploader(document.getElementById("uploader"), { onComplete: refresh });

  refresh();
  scheduleAuto();
})();
