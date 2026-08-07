#!/usr/bin/env python3
"""Step 5 calibration_v2 + leave-one-work-out frozen-selector run (research6).

Fits GLOBAL (not per-recording) severity maps per critical defect over the pooled
FIT works, learns per-defect Learn-Then-Test tau at 0.10/0.15/0.20 (work_model AND
work-atomic), writes calibration_v2.json (calibration_v1 schema shape), then for each
WORK freezes calibration on the OTHER works and runs the COMPLETE select_v3 (via
cli_autonomous.select_autonomous with calibration_path=frozen) on the held-out work.
Verdi is NEVER in any fit (external test-v1).

TRUTH DEFINITIONS (exact, from linear-exact references; anchored so severity=0.5 at
the truly_bad threshold):
  orchestral_theft : proxy=theft_mean(control); truth_sev=clip(1-min_rc_si/40);
                     truly_bad iff min control actual-construction SI-SDR < 20 dB.
  vocal_leakage    : proxy=vocal_interference_ratio |beta| (joint a*A+b*V regression);
                     truth_sev=clip(|beta|/0.60); truly_bad iff |beta| > 0.30.
  event_hole       : proxy=event_hole_depth_db (metrics_v2.event_holes, near-silent-band
                     gate); truth_sev=clip(depth/36); truly_bad iff worst depth > 18 dB.
  fullness (report): proxy=band_envelope_err_db; truth_sev=clip(band/12); bad iff >6 dB.
  stereo (report)  : proxy=stereo_width_err; truth_sev=clip(width/1.0); bad iff >0.50.
Critical (decision-binding, match selector.CRITICAL_DEFECTS minus severe_artifact):
  orchestral_theft, vocal_leakage, event_hole.
"""
import os, sys, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = os.path.expanduser("~/audio-extract"); sys.path.insert(0, REPO)
from audio_extract.selector import SeverityMap, learn_then_test, select_v3
from audio_extract.benchmark import GroupKey, assign_split
from audio_extract import cli_autonomous as cla
from audio_extract.storage import TrackLayout

WORKDIR = os.path.expanduser("~/step5_work")
LIB = os.path.join(WORKDIR, "lib")
RESULTS = os.path.join(WORKDIR, "step5_results.json")
TARGETS = (0.10, 0.15, 0.20); DELTA = 0.05; U = 0.05
NONVERDI = ["donizetti", "puccini", "aalto_mozart", "spheres_mozart", "spheres_tchaikovsky"]
ALL_WORKS = NONVERDI + ["verdi"]

THR = {"orchestral_theft": 20.0, "vocal_leakage": 0.30, "event_hole": 18.0,
       "fullness": 6.0, "stereo": 0.50}
CRITICAL = ["orchestral_theft", "vocal_leakage", "event_hole"]
DEFECTS = ["orchestral_theft", "vocal_leakage", "event_hole", "fullness", "stereo"]
TRUTH_DEFS = {
    "orchestral_theft": "proxy=theft_mean; truth_sev=clip(1-min_rc_si/40); truly_bad iff min "
                        "control actual-construction SI-SDR < 20 dB (A exact for a control).",
    "vocal_leakage": "proxy=vocal_interference_ratio |beta| (joint a*A+b*V lsq); "
                     "truth_sev=clip(|beta|/0.60); truly_bad iff |beta| > 0.30.",
    "event_hole": "proxy=event_hole_depth_db (metrics_v2.event_holes, near-silent-band gate); "
                  "truth_sev=clip(depth/36); truly_bad iff worst-events depth > 18 dB.",
    "fullness": "proxy=band_envelope_err_db; truth_sev=clip(band/12); bad iff >6 dB (reporting).",
    "stereo": "proxy=stereo_width_err; truth_sev=clip(width/1.0); bad iff >0.50 (reporting).",
}


def clip01(x): return float(np.clip(x, 0.0, 1.0))


def load_records(results):
    """One record per (work, recipe): per-condition raw proxies + theft truth."""
    recs = []
    for work, wd in results["works"].items():
        if "matrix" not in wd:
            print(f"  WARN work {work} has no matrix ({wd.get('error','')[:60]})"); continue
        conds = wd["conditions"]
        for rid, mm in wd["matrix"].items():
            cases = mm["cases"]
            eh, vl, full, st = [], [], [], []
            per_cond = {}
            for cond, ci in conds.items():
                pc = cases.get(ci["cid"], {})
                per_cond[cond] = pc
                if pc.get("event_hole_depth_db") is not None: eh.append(float(pc["event_hole_depth_db"]))
                if pc.get("vocal_interference_ratio") is not None: vl.append(float(pc["vocal_interference_ratio"]))
                if pc.get("band_envelope_err_db") is not None: full.append(float(pc["band_envelope_err_db"]))
                if pc.get("stereo_width_err") is not None: st.append(float(pc["stereo_width_err"]))
            recs.append({"work": work, "recipe": mm["name"], "recipe_id": rid,
                         "eh": eh, "vl": vl, "full": full, "st": st, "per_cond": per_cond,
                         "theft_mean": mm.get("theft_mean"), "min_rc_si": mm.get("min_rc_si"),
                         "mean_rc_band": mm.get("mean_rc_band")})
    return recs


# ---- per-defect: map anchors (raw, truth_sev), worst-proxy, truly_bad ----------
def anchors(rec, d):
    if d == "orchestral_theft":
        if rec["theft_mean"] is None or rec["min_rc_si"] is None: return []
        return [(rec["theft_mean"], clip01(1.0 - rec["min_rc_si"] / 40.0))]
    if d == "vocal_leakage":
        return [(v, clip01(v / 0.60)) for v in rec["vl"]]
    if d == "event_hole":
        return [(x, clip01(x / 36.0)) for x in rec["eh"]]
    if d == "fullness":
        return [(x, clip01(x / 12.0)) for x in rec["full"]]
    if d == "stereo":
        return [(x, clip01(x / 1.0)) for x in rec["st"]]
    raise KeyError(d)


def worst_proxy(rec, d):
    if d == "orchestral_theft": return rec["theft_mean"]
    if d == "vocal_leakage": return max(rec["vl"], default=0.0)
    if d == "event_hole": return max(rec["eh"], default=0.0)
    if d == "fullness": return max(rec["full"], default=0.0)
    if d == "stereo": return max(rec["st"], default=0.0)


def truly_bad(rec, d):
    if d == "orchestral_theft":
        return rec["min_rc_si"] is not None and rec["min_rc_si"] < THR[d]
    if d == "vocal_leakage": return any(v > THR[d] for v in rec["vl"])
    if d == "event_hole": return any(x > THR[d] for x in rec["eh"])
    if d == "fullness": return any(x > THR[d] for x in rec["full"])
    if d == "stereo": return any(x > THR[d] for x in rec["st"])


def fit_doc(fit_recs, all_recs, fit_works, held_works, tag):
    splits = {w: assign_split(GroupKey(work=w)) for w in ALL_WORKS}
    doc = {"schema": "audio-extract/calibration/v2", "tag": tag, "delta": DELTA,
           "uncertainty_u": U, "targets": list(TARGETS), "splits": splits,
           "fit_works": sorted(fit_works), "held_out_works": sorted(held_works),
           "critical_defects": CRITICAL, "truth_definitions": TRUTH_DEFS,
           "truth_thresholds": THR, "defects": {}}
    maps = {}
    for d in DEFECTS:
        raw, sev = [], []
        for r in fit_recs:
            for (x, y) in anchors(r, d):
                raw.append(x); sev.append(y)
        if len(raw) < 2:
            doc["defects"][d] = {"error": "insufficient anchors"}; continue
        m = SeverityMap.fit(raw, sev); maps[d] = m
        triples_wm = [(f'{r["work"]}::{r["recipe"]}', m.apply(worst_proxy(r, d)), truly_bad(r, d))
                      for r in fit_recs if worst_proxy(r, d) is not None]
        triples_w = [(r["work"], m.apply(worst_proxy(r, d)), truly_bad(r, d))
                     for r in fit_recs if worst_proxy(r, d) is not None]
        levels_wm, levels_w, strongest = {}, {}, None
        for t in TARGETS:
            rwm = learn_then_test(triples_wm, target_risk=t, delta=DELTA)
            levels_wm[str(t)] = rwm
            levels_w[str(t)] = learn_then_test(triples_w, target_risk=t, delta=DELTA)
            if rwm["certifiable"] and strongest is None:
                strongest = t
        n_bad = sum(1 for *_x, b in triples_wm if b)
        doc["defects"][d] = {
            "map_knots": {"xs": [round(float(x), 6) for x in m.xs],
                          "ys": [round(float(y), 6) for y in m.ys]},
            "fit_units_work_model": len(triples_wm), "fit_truly_bad": n_bad,
            "n_work_groups": len({w for w, *_ in triples_w}),
            "learn_then_test_unit=work_model": levels_wm,
            "learn_then_test_unit=work": levels_w,
            "strongest_certifiable_target": strongest,
            "is_critical": d in CRITICAL,
        }
    # per-work-recipe unit dump (out-of-sample transparency)
    units = []
    for r in all_recs:
        units.append({"work": r["work"], "recipe": r["recipe"], "split": splits[r["work"]],
                      "in_fit": r["work"] in fit_works,
                      **{f"{d}_proxy": (round(worst_proxy(r, d), 5) if worst_proxy(r, d) is not None else None)
                         for d in DEFECTS},
                      **{f"{d}_bad": bool(truly_bad(r, d)) for d in DEFECTS},
                      "min_rc_si": r["min_rc_si"], "theft_mean": r["theft_mean"]})
    doc["units"] = units
    return doc, maps


def build_candidates_correct(W, maps, recs):
    """Correctly-keyed candidates for select_v3 (works around the shipped
    severity_cells_from_store keying bug: it keys per-case cells by challenge_id, not
    candidate_recipe_id, so every recipe loses its event_hole/leakage cells). Here we
    pool each RECIPE's conditions and apply the frozen maps exactly as the certified
    path INTENDS (event_hole u includes masked_hole_uncertainty; theft one cell)."""
    cands = {}
    for r in [x for x in recs if x["work"] == W]:
        cells = {}
        for cond, pc in r["per_cond"].items():
            if pc.get("event_hole_depth_db") is not None:
                mhu = float(pc.get("masked_hole_uncertainty") or 0.0)
                cells.setdefault("event_hole", []).append(
                    (maps["event_hole"].apply(float(pc["event_hole_depth_db"])), U + mhu))
            if pc.get("vocal_interference_ratio") is not None:
                cells.setdefault("vocal_leakage", []).append(
                    (maps["vocal_leakage"].apply(float(pc["vocal_interference_ratio"])), U))
            if pc.get("band_envelope_err_db") is not None:
                cells.setdefault("fullness", []).append(
                    (maps["fullness"].apply(float(pc["band_envelope_err_db"])), U))
            if pc.get("stereo_width_err") is not None:
                cells.setdefault("stereo", []).append(
                    (maps["stereo"].apply(float(pc["stereo_width_err"])), U))
        theft_sev = maps["orchestral_theft"].apply(r["theft_mean"]) if r["theft_mean"] is not None else 1.0
        cells["orchestral_theft"] = [(theft_sev, U)]
        gates = {"technical": 0.0, "orchestral_theft": theft_sev}
        if cells.get("event_hole"):
            gates["event_hole"] = max(s for s, _ in cells["event_hole"])
        secondary = float(np.mean([s for s, _ in cells.get("fullness", [(0.0, 0)])]))
        cands[r["recipe_id"]] = {"cells": cells, "gates": gates, "secondary": secondary,
                                 "recipe_id": r["recipe_id"], "name": r["recipe"]}
    return cands


def main():
    results = json.load(open(RESULTS))
    recs = load_records(results)
    print(f"records={len(recs)} works={sorted({r['work'] for r in recs})}")
    print("\n=== per (work,recipe) worst proxies + truly_bad ===")
    for r in sorted(recs, key=lambda x: (x["work"], x["recipe"])):
        flags = "".join(d[0].upper() if truly_bad(r, d) else "." for d in DEFECTS)
        print(f"  {r['work']:20s} {r['recipe']:28s} theft={r['theft_mean']} min_rc_si={r['min_rc_si']} "
              f"eh={round(worst_proxy(r,'event_hole'),1)} vl={round(worst_proxy(r,'vocal_leakage'),3)} "
              f"st={round(worst_proxy(r,'stereo'),3)} bad[{'/'.join(d[:2] for d in DEFECTS)}]={flags}")

    # ---- MAIN calibration_v2: fit on all 5 non-verdi ----
    fit_recs = [r for r in recs if r["work"] in NONVERDI]
    doc2, _ = fit_doc(fit_recs, recs, NONVERDI, ["verdi"], "calibration_v2_all_nonverdi")
    Path(os.path.join(WORKDIR, "calibration_v2.json")).write_text(json.dumps(doc2, indent=1, default=str))
    print("\n=== calibration_v2 per-defect strongest certifiable target (work_model) ===")
    for d in DEFECTS:
        dd = doc2["defects"][d]
        wm = {t: dd["learn_then_test_unit=work_model"][t] for t in ("0.1", "0.15", "0.2")}
        w = {t: dd["learn_then_test_unit=work"][t] for t in ("0.1", "0.15", "0.2")}
        print(f"  {d:18s} bad={dd['fit_truly_bad']}/{dd['fit_units_work_model']} "
              f"strongest_wm={dd['strongest_certifiable_target']} "
              f"| wm cert@[.10/.15/.20]={[wm[t]['certifiable'] for t in ('0.1','0.15','0.2')]} "
              f"tau={[round(wm[t]['tau'],3) for t in ('0.1','0.15','0.2')]} "
              f"| work cert={[w[t]['certifiable'] for t in ('0.1','0.15','0.2')]}")

    # ---- LOWO frozen-selector ----
    print("\n=== LEAVE-ONE-WORK-OUT frozen-selector ===")
    lowo = {}
    for W in ALL_WORKS:
        fit_works = [w for w in NONVERDI if w != W]      # verdi always excluded
        fr = [r for r in recs if r["work"] in fit_works]
        docW, mapsW = fit_doc(fr, recs, fit_works, [W], f"lowo_holdout_{W}")
        fp = os.path.join(WORKDIR, f"calib_lowo_{W}.json")
        Path(fp).write_text(json.dumps(docW, indent=1, default=str))
        lay = TrackLayout(LIB, W)
        import sqlite3 as _sq
        _c = _sq.connect(str(lay.manifest_sqlite))
        for _t in ("selection_decision", "job"):
            try: _c.execute(f"DELETE FROM {_t}")
            except _sq.OperationalError: pass
        _c.commit(); _c.close()   # fresh so the as-shipped call isn't an immutable re-run conflict
        id2name = {r["recipe_id"]: r["recipe"] for r in recs}
        cands = build_candidates_correct(W, mapsW, recs)
        outcomes = {}
        for t in TARGETS:
            # (a) as-shipped certified path (documents the keying bug)
            try:
                shipped = cla.select_autonomous(lay, calibration_path=fp, target_risk=str(t),
                                                task="soloist_vs_rest", domain="opera_exact_pairs",
                                                distribution_flag="in_calibration_domain",
                                                probe_budget_left=False)
            except Exception:
                import traceback
                shipped = {"status": "ERROR", "error": traceback.format_exc()}
            # (b) corrected select_v3 with properly-keyed candidates (intended behavior)
            taus_t = {d: docW["defects"][d]["learn_then_test_unit=work_model"][str(t)]
                      for d in mapsW if d in docW["defects"]
                      and "learn_then_test_unit=work_model" in docW["defects"][d]}
            dec = select_v3(cands, taus_t, distribution_flag="in_calibration_domain",
                            probe_budget_left=False)
            cand = dec.get("candidate_id") or dec.get("best_available")
            cand_name = id2name.get(cand)
            noncert = [d for d in CRITICAL if not taus_t.get(d, {}).get("certifiable", False)]
            infeas = {id2name.get(cid, cid): cr.get("infeasible_defects")
                      for cid, cr in (dec.get("candidates") or {}).items()
                      if cr.get("infeasible_defects") or cr.get("hard_failed")}
            outcomes[str(t)] = {"status": dec.get("status"), "candidate_id": cand,
                                "candidate_name": cand_name, "reason": dec.get("reason"),
                                "noncertifiable_critical_taus": noncert,
                                "separation": dec.get("separation"),
                                "per_candidate_infeasible": infeas,
                                "shipped_status": shipped.get("status"),
                                "shipped_note": "select_autonomous mis-keys cells (challenge_id vs "
                                                "candidate_recipe_id) -> spurious no_acceptable"}
            print(f"  holdout={W:20s} fit={len(fit_works)}w t={t} -> CORRECTED:{dec.get('status')} "
                  f"cand={cand_name} noncert={noncert} | shipped:{shipped.get('status')} "
                  f"| {str(dec.get('reason'))[:48]}")
        lowo[W] = {"fit_works": fit_works, "frozen_artifact": os.path.basename(fp),
                   "held_out_is_frozen_test_v1": W == "verdi", "outcomes": outcomes}
    doc2["lowo_frozen_selector"] = lowo
    doc2["known_bug"] = {
        "where": "cli_autonomous.severity_cells_from_store",
        "bug": "per-case cells are keyed by challenge_id (positional: the tuple "
               "(case['challenge_id'], r['candidate_recipe_id'], result) is unpacked as "
               "(recipe_id, model, result) and key=recipe_id). So every recipe loses its "
               "event_hole/vocal_leakage/fullness/stereo cells (theft is keyed correctly by "
               "candidate_recipe_id), and a phantom challenge-keyed candidate collects them. "
               "select_autonomous therefore returns no_acceptable_candidate for ALL works at ALL "
               "targets for a spurious reason (missing cells -> ucb=1.0).",
        "impact": "Blocks the certified recipe-spec select path. LOWO outcomes below use a "
                  "correctly-keyed select_v3 (build_candidates_correct) to show the TRUE risk-gate "
                  "behavior; the as-shipped status is recorded per fold as 'shipped_status'.",
        "fix": "key by candidate_recipe_id: `key = model if str(model).startswith('sha256:') else model` "
               "(i.e. use r['candidate_recipe_id'], not case['challenge_id'])."}
    # aggregate coverage / abstention (primary target = 0.20, loosest)
    for t in TARGETS:
        tk = str(t)
        n_final = sum(1 for W in NONVERDI if lowo[W]["outcomes"][tk]["status"] == "final")
        n_abst = sum(1 for W in NONVERDI if lowo[W]["outcomes"][tk]["status"] in
                     ("no_acceptable_candidate", "needs_probe"))
        doc2.setdefault("lowo_aggregate", {})[tk] = {
            "n_works_nonverdi": len(NONVERDI), "n_final": n_final, "n_abstain": n_abst,
            "verdi_status": lowo["verdi"]["outcomes"][tk]["status"]}
    Path(os.path.join(WORKDIR, "calibration_v2.json")).write_text(json.dumps(doc2, indent=1, default=str))
    print("\n=== LOWO aggregate ===")
    for t in TARGETS:
        a = doc2["lowo_aggregate"][str(t)]
        print(f"  target={t}: nonverdi final={a['n_final']}/{a['n_works_nonverdi']} "
              f"abstain={a['n_abstain']} | verdi(test-v1)={a['verdi_status']}")
    print(f"\nWROTE {os.path.join(WORKDIR,'calibration_v2.json')}")


if __name__ == "__main__":
    main()
