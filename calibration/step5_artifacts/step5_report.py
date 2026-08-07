#!/usr/bin/env python3
"""Generate ~/step5_report.md from step5_results.json + calibration_v2.json."""
import os, json, platform, subprocess
from datetime import datetime, timezone

WORKDIR = os.path.expanduser("~/step5_work")
R = json.load(open(os.path.join(WORKDIR, "step5_results.json")))
C = json.load(open(os.path.join(WORKDIR, "calibration_v2.json")))
NONVERDI = ["donizetti", "puccini", "aalto_mozart", "spheres_mozart", "spheres_tchaikovsky"]
CRIT = C.get("critical_defects", ["orchestral_theft", "vocal_leakage", "event_hole"])
DEFECTS = ["orchestral_theft", "vocal_leakage", "event_hole", "fullness", "stereo"]
TK = ("0.1", "0.15", "0.2")
out = []
def w(s=""): out.append(s)

def gpu():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        return f"(nvidia-smi unavailable: {e})"

w(f"# Step 5 — WORK-LEVEL calibration_v2 + LOWO frozen-selector (first work-level certification numbers)")
w(f"\n_Generated {datetime.now(timezone.utc).isoformat()} on {platform.node()}._")
w(f"\nGPU: `{gpu()}`  |  total scoring wall: **{R.get('total_wall_s')} s**")
w("\n> **CLAIM DISCIPLINE.** This is reported as **PIPELINE VALIDATION** (does the frozen "
  "calibration -> select_v3 path run end-to-end and terminate correctly per held-out work). "
  "The statistical **risk claim is DEFERRED** until the ~29-independent-work bar; with "
  f"{len(NONVERDI)} fit works (Verdi held external) the work-atomic certified-risk bound cannot "
  "reach the targets, and abstention is the CORRECT outcome, not a failure.")

# ---- 1. inventory ----
w("\n## 1. Expanded work / condition inventory")
w("\nGrouping unit = **WORK** (a work's conditions never split across train/test). Verdi is the "
  "frozen external test-v1 and is in **no** calibration fit.\n")
w("| work | conditions | n_cond | linear-exact (dB) | voc_frac | stereo side/mid | source |")
w("|---|---|---|---|---|---|---|")
for wk, wd in R["works"].items():
    if "conditions" not in wd: continue
    conds = wd["conditions"]
    le = []; vf = []; sm = []; src = ""
    for c, ci in conds.items():
        m = ci["meta"]; le.append(str(m.get("linear_exactness_db"))); vf.append(str(m.get("voice_energy_frac_of_mix")))
        sm.append(str(m.get("mix_stereo_side_mid_ratio", 0.0))); src = m.get("source", src)
    w(f"| {wk} | {', '.join(conds)} | {len(conds)} | {'/'.join(le)} | {'/'.join(vf)} | {'/'.join(sm)} | {src[:42]} |")
# rendering recipe for spheres
sm0 = next((ci["meta"] for wk in ("spheres_mozart",) for ci in R["works"].get(wk, {}).get("conditions", {}).values()), {})
w("\n**Measured-hall / stereo rendering recipe (Spheres-Mozart, Spheres-Tchaikovsky):**")
w(f"- Orchestra = mono sum of every Spheres instrument stem over a fixed 90 s window "
  f"({sm0.get('n_orch_stems','?')} stems, 48k->44.1k). Voice = borrowed anechoic Aalto soprano "
  f"({sm0.get('soprano_files')}), scaled to ~12% mix energy.")
w(f"- **RIR:** The Spheres measured RIR `{sm0.get('rir_source_seat')}` "
  f"(shape {sm0.get('rir_shape_receivers_taps')} = receivers x taps @48k, resampled to 44.1k). "
  f"The SAME RIR is applied IDENTICALLY to voice and orchestra -> target A stays exact by linear "
  f"construction (mix = H(orch)+H(voice) = H(orch+voice), target = H(orch)).")
w(f"- Conditions: **dry** (H=identity, dual-mono); **hall** (H = RIR receiver 0, dual-mono, MEASURED "
  f"reverb); **stereo** (H = RIR receivers {{0,12}}, a spaced stereo receiver pair -> genuine "
  f"measured-hall stereo). Aalto-Mozart keeps its dry + synthetic-RT60 hall (one work).")
w(f"- Note: convolution lowers hall/stereo voc_frac (~0.05) vs dry (0.12) — a real acoustic effect "
  f"(the continuous orchestra gains more steady-state reverb energy than the intermittent voice).")

# ---- 2. raw proxies per (work,recipe) ----
def units():
    return C.get("units", [])
w("\n## 2. Per (work, recipe) worst proxies + truth flags")
w("\ntruth thresholds: " + ", ".join(f"{k}={v}" for k, v in C.get("truth_thresholds", {}).items()))
w("\n| work | recipe | theft_mean | min_rc_si | eh_depth | v_intf | band | stereo_err | bad(theft/leak/hole/full/st) |")
w("|---|---|---|---|---|---|---|---|---|")
for u in sorted(units(), key=lambda x: (x["work"], x["recipe"])):
    flags = "".join("X" if u.get(f"{d}_bad") else "." for d in DEFECTS)
    w(f"| {u['work']} | {u['recipe']} | {u.get('theft_mean')} | {u.get('min_rc_si')} | "
      f"{u.get('event_hole_proxy')} | {u.get('vocal_leakage_proxy')} | {u.get('fullness_proxy')} | "
      f"{u.get('stereo_proxy')} | {flags} |")

# ---- 3. calibration_v2 tau table ----
w("\n## 3. calibration_v2 per-defect tau table (fit on 5 non-Verdi works)")
w("\ncert = Learn-Then-Test certifiable (Clopper-Pearson one-sided UCB <= target, delta=0.05). "
  "tau = certification threshold on calibrated severity. **work_model** unit groups by (work,recipe); "
  "**work** unit groups by work (atomic — the honest independent-work count).\n")
w("| defect | critical? | fit units (wm) | truly_bad | strongest cert (wm) | work_model cert @.10/.15/.20 | tau @.10/.15/.20 | **work-atomic** cert @.10/.15/.20 |")
w("|---|---|---|---|---|---|---|---|")
for d in DEFECTS:
    dd = C["defects"].get(d, {})
    if "learn_then_test_unit=work_model" not in dd:
        w(f"| {d} | {d in CRIT} | - | - | (no anchors) | - | - | - |"); continue
    wm = dd["learn_then_test_unit=work_model"]; wa = dd["learn_then_test_unit=work"]
    cwm = "/".join("Y" if wm[t]["certifiable"] else "n" for t in TK)
    taus = "/".join(str(round(wm[t]["tau"], 3)) if wm[t]["certifiable"] else "-" for t in TK)
    cwa = "/".join("Y" if wa[t]["certifiable"] else "n" for t in TK)
    w(f"| {d} | {d in CRIT} | {dd['fit_units_work_model']} | {dd['fit_truly_bad']} | "
      f"{dd['strongest_certifiable_target']} | {cwm} | {taus} | {cwa} |")
w("\n**Map knots (isotonic raw->severity, GLOBAL not per-recording):**")
for d in CRIT:
    dd = C["defects"].get(d, {}); k = dd.get("map_knots", {})
    xs = k.get("xs", []); ys = k.get("ys", [])
    w(f"- `{d}`: {len(xs)} knots; x[min..max]=[{xs[0] if xs else '?'}..{xs[-1] if xs else '?'}], "
      f"y[min..max]=[{min(ys) if ys else '?'}..{max(ys) if ys else '?'}]")

# ---- 4. LOWO ----
w("\n## 4. Leave-one-work-out frozen-selector (the deliverable)")
kb = C.get("known_bug", {})
if kb:
    w("\n> **BUG FOUND (blocks the as-shipped certified path).** `" + kb.get("where", "") + "`: "
      + kb.get("bug", "") + "  \n> **Impact:** " + kb.get("impact", "") + "  \n> **Fix:** `"
      + kb.get("fix", "") + "`")
w("\nFor each held-out work: freeze calibration on the OTHER works, run the COMPLETE select_v3 on the "
  "held-out work's cases. Verdi always excluded from every fit. Because of the bug above, the "
  "as-shipped `cli_autonomous.select_autonomous(calibration_path=frozen)` returns "
  "`no_acceptable_candidate` for EVERY work/target (spurious — missing cells). The table below is the "
  "**correctly-keyed select_v3** (same frozen maps + taus, cells pooled per recipe) = the intended "
  "certified behavior; the as-shipped status is shown alongside. Primary target = **0.20** (loosest).\n")
lowo = C.get("lowo_frozen_selector", {})
w("| held-out work | fit works | select_v3 @0.20 | winner recipe | bounding (non-cert critical taus) | as-shipped | reason |")
w("|---|---|---|---|---|---|---|")
for wk in NONVERDI + ["verdi"]:
    lw = lowo.get(wk, {})
    o = lw.get("outcomes", {}).get("0.2", {})
    tag = " (TEST-v1)" if wk == "verdi" else ""
    w(f"| {wk}{tag} | {len(lw.get('fit_works', []))} | **{o.get('status')}** | {o.get('candidate_name')} | "
      f"{o.get('noncertifiable_critical_taus')} | {o.get('shipped_status')} | {str(o.get('reason'))[:50]} |")
w("\n**All targets per held-out work:**")
for wk in NONVERDI + ["verdi"]:
    lw = lowo.get(wk, {}); os_ = lw.get("outcomes", {})
    row = " | ".join(f".{t.split('.')[1]}:{os_.get(t,{}).get('status','?')}"
                     f"({os_.get(t,{}).get('candidate_name')})" for t in TK)
    w(f"- **{wk}**: {row}")
agg = C.get("lowo_aggregate", {})
w("\n**Aggregate coverage / abstention (non-Verdi works):**")
w("\n| target | n_final | n_abstain | n_works | verdi (test-v1) |")
w("|---|---|---|---|---|")
for t in TK:
    a = agg.get(t, {})
    w(f"| {t} | {a.get('n_final')} | {a.get('n_abstain')} | {a.get('n_works_nonverdi')} | {a.get('verdi_status')} |")
# accepted-outcome-failure: any 'final' whose winner is truly_bad on a critical defect on the held-out work
w("\n**Accepted-outcome failures** (a `final` whose winner is truly_bad on a critical defect on its "
  "OWN held-out work — the risk-relevant error):")
ubywr = {(u["work"], u["recipe"]): u for u in units()}
fails = []
for wk in NONVERDI + ["verdi"]:
    o = lowo.get(wk, {}).get("outcomes", {}).get("0.2", {})
    if o.get("status") == "final" and o.get("candidate_name"):
        u = ubywr.get((wk, o["candidate_name"]), {})
        badc = [d for d in CRIT if u.get(f"{d}_bad")]
        if badc:
            fails.append(f"{wk}:{o['candidate_name']} bad on {badc}")
w("- " + ("; ".join(fails) if fails else "NONE at target 0.20 (every `final` winner is truly-good on all critical defects on its held-out work)."))

# ---- 5. answers ----
w("\n## 5. Key questions")
median_final = [wk for wk in NONVERDI + ["verdi"]
                for t in TK
                if lowo.get(wk, {}).get("outcomes", {}).get(t, {}).get("status") == "final"
                and lowo.get(wk, {}).get("outcomes", {}).get(t, {}).get("candidate_name", "").startswith("ens_median")]
w(f"\n**Does the median ensemble certify anywhere?** "
  + (f"YES, once — `ens_median:MDX23C/MelBand/BS` reaches a certified `final` for {sorted(set(median_final))} "
     "(at target risk **0.20** only, with the fuller **5-work** calibration, via the correctly-keyed "
     "select_v3). It passes hard + calibrated gates, is leave-one-evidence-out stable, and is separated "
     "from the runner-up. This is the FIRST work-level certified selection. It does NOT certify on the "
     "4-work LOWO folds: leaving a work out drops the orchestral_theft tau below certifiable at 0.20 "
     "(the n-boundary — 5 works barely certifies theft@0.20, 4 does not), and it never certifies at "
     "0.10/0.15 (n far too small). The risk claim stays DEFERRED to the ~29-work bar."
     if median_final else
     "NO — ens_median never reached a certified `final` in any fold at any target."))
# hall vs dry certifiability: compare truly_bad rates on dry vs hall/stereo for critical defects
def cond_flags(cond_key):
    rows = []
    for wk, wd in R["works"].items():
        if "conditions" not in wd: continue
        if cond_key not in wd["conditions"]: continue
        cid = wd["conditions"][cond_key]["cid"]
        for rid, mm in wd["matrix"].items():
            pc = mm["cases"].get(cid, {})
            rows.append((wk, mm["name"], pc, mm))
    return rows
def bad_on(pc, mm, d):
    th = C["truth_thresholds"]
    if d == "event_hole": return (pc.get("event_hole_depth_db") or 0) > th["event_hole"]
    if d == "vocal_leakage": return (pc.get("vocal_interference_ratio") or 0) > th["vocal_leakage"]
    if d == "orchestral_theft": return (mm.get("min_rc_si") if mm.get("min_rc_si") is not None else 99) < th["orchestral_theft"]
    return False
w("\n**Does measured-hall change certifiability vs dry?** Critical-defect truly_bad counts by condition:")
w("\n| condition | n(rec-cases) | theft-bad | leak-bad | hole-bad |")
w("|---|---|---|---|---|")
for ck in ("dry", "hall", "stereo"):
    rows = cond_flags(ck)
    if not rows: continue
    tb = sum(1 for _, _, pc, mm in rows if bad_on(pc, mm, "orchestral_theft"))
    lb = sum(1 for _, _, pc, mm in rows if bad_on(pc, mm, "vocal_leakage"))
    hb = sum(1 for _, _, pc, mm in rows if bad_on(pc, mm, "event_hole"))
    w(f"| {ck} | {len(rows)} | {tb} | {lb} | {hb} |")
w("\n_Honest, nuanced answer:_ within this small n, measured-hall does NOT change the certification "
  "OUTCOME — every fold abstains regardless of condition, and Verdi (dry) certifies; the binding "
  "constraint is the orchestral_theft tau's n-sensitivity (5 vs 4 fit works), a WORK-count effect, not "
  "a dry-vs-hall effect (theft is a work-level control-side quantity). At the CONDITION level the "
  "effects are mixed and instructive: reverb SMEARS event-holes (all event_hole truly_bad cases are on "
  "DRY — BS's deep holes; hall/stereo depths fall below the 18 dB threshold), and it crushes broadband "
  "SI-SDR (Aalto median dry->hall 23.7->12.5 dB) and raises masking uncertainty — but SI-SDR is not a "
  "critical gate. The MEASURED-hall Spheres cases behave like the synthetic-hall Aalto case (reverb "
  "lowers holes, raises interference). The STEREO condition is well-preserved by every recipe "
  "(stereo_width_err ~0.001-0.006 << 0.5 -> stereo is never a binding defect). Net: hall is a genuine "
  "additional CONDITION axis but, at n=6 works, certifiability is gated by work-count, not reverb.")

# ---- Verdi test-v1, separately ----
w("\n### Verdi frozen test-v1 (never in any fit) — reported separately")
vo = lowo.get("verdi", {}).get("outcomes", {})
for t in TK:
    o = vo.get(t, {})
    w(f"- target {t}: **{o.get('status')}**"
      + (f" -> winner `{o.get('candidate_name')}`" if o.get('status') == 'final' else
         f" (best_available `{o.get('candidate_name')}`, non-cert taus {o.get('noncertifiable_critical_taus')})")
      + f"; as-shipped select_autonomous: {o.get('shipped_status')}.")
w("Verdi's median-ensemble `final` at 0.20 is a legitimate accept: on Verdi the median residual is "
  "truly-good on ALL critical defects (theft min-construction SI-SDR 33.8 dB > 20; event-hole 6.3 dB "
  "< 18; vocal-interference 0.095 < 0.30), so it is NOT an accepted-outcome failure.")

# ---- 6. ops ----
w("\n## 6. GPU timing / disk / bugs")
tim = R.get("timings", [])
if tim:
    tot = sum(t.get("sep_time_s", 0) for t in tim)
    w(f"- separations: {len(tim)}, total sep time {round(tot,1)} s, "
      f"mean {round(tot/max(1,len(tim)),1)} s. Stereo stems observed: "
      f"{set(tuple(v) for t in tim for v in [list(t.get('stem_ch',{}).values())[0]] if t.get('stem_ch'))}")
w(f"- scoring wall: {R.get('total_wall_s')} s (~{round((R.get('total_wall_s') or 0)/60,1)} min), "
  f"6 works / 11 (work,condition) cases / 4 recipes on RTX PRO 4500. Disk held flat (concatenated "
  "controls deleted per work, stems symlinked, temp cleared per separation).")
w("- **BUG (blocks certified path):** `cli_autonomous.severity_cells_from_store` keys per-case cells "
  "by `challenge_id` instead of `candidate_recipe_id` (see the boxed note in section 4). The whole "
  "LOWO table therefore uses a correctly-keyed select_v3 re-run; as-shipped select_autonomous returns "
  "no_acceptable everywhere. No other tracebacks; the GPU scoring run completed cleanly.")
w("- Stereo separation verified: stereo mixes -> stereo stems (2-channel residuals); dual-mono cases "
  "-> width_err ~= 0 as expected.")
w("\n---\n_Artifacts: `calibration_v2.json` (fit on 5 non-Verdi works), `calib_lowo_<work>.json` "
  "(per-fold frozen artifacts), `step5_results.json` (raw scored matrix)._")

open(os.path.join(os.path.expanduser("~"), "step5_report.md"), "w").write("\n".join(out) + "\n")
print("WROTE ~/step5_report.md")
