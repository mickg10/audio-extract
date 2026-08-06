#!/usr/bin/env python3
"""Seed a synthetic ``calibration/calibration_v1.json`` under a GUI ``--lib`` root
that is **shape-identical to the real artifact** (schema
``audio-extract/calibration/v1``) so the **/calibration** page renders locally,
and identically, without a full Learn-Then-Test campaign.

The machinery is REAL, only the inputs are synthetic: every severity curve comes
from ``audio_extract.selector.SeverityMap.fit`` (PAV isotonic regression) and
every τ comes from ``audio_extract.selector.learn_then_test`` (fixed-sequence
scan with an exact one-sided Clopper–Pearson upper bound), grouped by
``work × model`` exactly as production does. Held-out violation rates are then
*measured* on a disjoint held-out split, so "claimed vs observed" is a genuine
out-of-sample check.

It deliberately reproduces the real artifact's three defects and their quirks so
the page's honest-refusal path is exercised:

* ``orchestral_theft`` — proxy (``theft_mean``) tracks truth → **certifiable at
  all three targets** (10 / 15 / 20%);
* ``event_hole``       — near-flat, high severity map (a structurally
  uninformative proxy) → **certification refused**;
* ``fullness``         — sloped map but the proxy does not track truth
  out-of-sample → **certification refused**.

Usage:  uv run python web/dev_seed_calibration.py [--lib /tmp/guilib4] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from audio_extract.selector import SeverityMap, learn_then_test  # noqa: E402

TARGETS = (0.10, 0.15, 0.20)
DELTA = 0.05
UNCERTAINTY_U = 0.05

# The real executed panel (mirrors configs + model-lock aliases). model_rank is a
# quality proxy: lower rank = cleaner separation (steals less, holes less).
MODELS = [
    "kim_vocal_2_onnx",
    "uvr_mdx_net_voc_ft_onnx",
    "mdx23c_8kfft_instvoc_hq_ckpt",
    "vocals_mel_band_roformer_ckpt",
    "model_bs_roformer_ep_317_sdr_12_9755_ckpt",
]

# 18 works across the four splits (fit = calibration+regression+dev; heldout is
# never touched during fitting). fit_units = 14×5 = 70; held-out = 4×5 = 20.
WORKS_SPLITS = (
    [("cal_%02d" % i, "calibration") for i in range(1, 9)] +     # 8
    [("reg_%02d" % i, "regression") for i in range(1, 4)] +      # 3
    [("dev_%02d" % i, "dev") for i in range(1, 4)] +             # 3
    [("held_%02d" % i, "heldout") for i in range(1, 5)]          # 4
)

TRUTH = {
    "event_hole": "truth_sev = clip(1 - min_case_si_sdr/30, 0, 1) over exact-reference remix "
                  "cases; truly_bad iff min case SI-SDR < 6.0 dB. proxy_raw = clip(1 - "
                  "min_rc_si_sdr/30, 0, 1) on genuine controls (deployable without synthetic truth).",
    "orchestral_theft": "truth_sev = clip(1 - min_rc_si_sdr/40, 0, 1); truly_bad iff min over "
                        "controls of actual-construction SI-SDR < 20.0 dB (A is exact truth for a "
                        "control). proxy_raw = theft_mean (mean broadband stem-energy theft on controls).",
    "fullness": "truth_sev = clip(mean_case_band_envelope_err/30, 0, 1) over exact cases; truly_bad "
                "iff mean case band_envelope_err > 6.0 dB. proxy_raw = clip(mean_rc_band_envelope_err/"
                "30, 0, 1) on controls. NOTE: proxy is a structurally weak stand-in for the exact truth.",
}


def _rank(model: str) -> int:
    return MODELS.index(model)


def gen_defect_units(name: str, rng: np.random.Generator):
    """Per (work, model) unit: (work, model, split, raw_proxy, true_sev, truly_bad).

    Informativeness is defect-specific and drives whether learn_then_test can
    certify — exactly the real behaviour, not a hand-set flag."""
    units = []
    for work, split in WORKS_SPLITS:
        for model in MODELS:
            r = _rank(model)
            if name == "orchestral_theft":
                # theft_mean grows with rank; clean models sit near 0. Truth
                # tracks the proxy tightly -> certifiable.
                base = 0.010 + 0.028 * r
                raw = float(max(0.0, base + rng.normal(0, 0.010) + (0.22 if rng.random() < 0.10 else 0.0)))
                true_sev = float(np.clip(raw / 0.55, 0, 1) ** 0.9)
                p_bad = float(np.clip((raw - 0.055) / 0.05, 0, 1))
                bad = bool(rng.random() < p_bad)
            elif name == "event_hole":
                # near-constant, high proxy (uninformative); truth ~ coin flip
                # independent of the proxy -> never certifiable, map is flat-high.
                raw = float(np.clip(0.90 + rng.normal(0, 0.02), 0, 1))
                true_sev = float(np.clip(0.90 + rng.normal(0, 0.03), 0, 1))
                bad = bool(rng.random() < 0.48)
            else:  # fullness — sloped map, but proxy does not predict truth
                raw = float(np.clip(rng.beta(1.6, 2.2), 0, 1))
                true_sev = float(np.clip(raw * 0.75 + rng.normal(0, 0.05), 0, 1))
                bad = bool(rng.random() < 0.42)   # independent of raw
            units.append((work, model, split, round(raw, 5), true_sev, bad))
    return units


def per_target_ltt(units, *, group_key):
    """learn_then_test at each target, grouped by ``group_key(work, model)``."""
    smap_rows = [(group_key(w, m), pred, bad) for (w, m, sp, raw, ts, bad), pred in units]
    out = {}
    for tr in TARGETS:
        out["%g" % tr] = learn_then_test(smap_rows, target_risk=tr, delta=DELTA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", default="/tmp/guilib4", help="target GUI lib root")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    splits = {w: s for w, s in WORKS_SPLITS}
    fit_works = [w for w, s in WORKS_SPLITS if s != "heldout"]
    held_out_works = [w for w, s in WORKS_SPLITS if s == "heldout"]

    defects: dict[str, dict] = {}
    held_out_violation: dict[str, dict] = {}
    # accumulate per (work, model) metrics for the top-level units table
    unit_metrics: dict[tuple, dict] = {}

    for name in ("event_hole", "orchestral_theft", "fullness"):
        raw_units = gen_defect_units(name, rng)
        fit = [(w, m, sp, raw, ts, bad) for (w, m, sp, raw, ts, bad) in raw_units if sp != "heldout"]
        held = [(w, m, sp, raw, ts, bad) for (w, m, sp, raw, ts, bad) in raw_units if sp == "heldout"]

        # 1) fit the severity map on the fit split (raw -> true severity)
        smap = SeverityMap.fit([u[3] for u in fit], [u[4] for u in fit])
        map_knots = {"xs": [round(float(x), 5) for x in smap.xs],
                     "ys": [round(float(y), 5) for y in smap.ys]}

        # 2) learn_then_test at 3 targets, grouped work×model (and work-only)
        fit_pred = [((w, m, sp, raw, ts, bad), smap.apply(raw)) for (w, m, sp, raw, ts, bad) in fit]
        wm = per_target_ltt(fit_pred, group_key=lambda w, m: "%s::%s" % (w, m))
        wonly = per_target_ltt(fit_pred, group_key=lambda w, m: w)

        strongest = None
        for tr in TARGETS:
            if wm["%g" % tr]["certifiable"]:
                strongest = tr
                break

        defects[name] = {
            "map_knots": map_knots,
            "fit_units": len(fit),
            "fit_truly_bad": int(sum(1 for u in fit if u[5])),
            "learn_then_test_unit=work_model": wm,
            "learn_then_test_unit=work (sensitivity)": wonly,
            "strongest_certifiable_target": strongest,
        }

        # 3) out-of-sample violation check on the held-out split
        held_pred = [((w, m), smap.apply(raw), bad) for (w, m, sp, raw, ts, bad) in held]
        per_target = {}
        for tr in TARGETS:
            entry = wm["%g" % tr]
            if not entry["certifiable"]:
                per_target["%g" % tr] = {"certifiable": False}
                continue
            tau = entry["tau"]
            certified = [(wmk, pred, bad) for wmk, pred, bad in held_pred if pred <= tau]
            viol = [wmk for wmk, pred, bad in certified if bad]
            n = len(certified)
            per_target["%g" % tr] = {
                "tau": tau, "n_certified": n, "n_violations": len(viol),
                "observed_rate": round(len(viol) / n, 4) if n else 0.0,
                "claimed_bound": tr,
                "violating": ["%s::%s" % (w, m) for (w, m) in viol],
            }
        held_out_violation[name] = {
            "n_units": len(held),
            "per_target": per_target,
            "rows": [{"work": w, "model": m, "pred": round(smap.apply(raw), 4), "bad": bool(bad)}
                     for (w, m, sp, raw, ts, bad) in held],
        }

        # stash proxy values for the units table
        for (w, m, sp, raw, ts, bad) in raw_units:
            unit_metrics.setdefault((w, m, sp), {})[name] = raw

    # top-level per-unit measurement table (schema completeness; decorative)
    units = []
    for (w, m, sp), mv in sorted(unit_metrics.items()):
        theft = mv.get("orchestral_theft", 0.0)
        hole = mv.get("event_hole", 0.0)
        full = mv.get("fullness", 0.0)
        units.append({
            "work": w, "model": m, "n_cases": 8, "n_controls": 3,
            "min_case_si": round(30.0 * (1 - hole), 3),
            "mean_case_band": round(30.0 * full, 4),
            "mean_case_stft": round(1.0 + 0.6 * full, 4),
            "min_rc_si": round(40.0 * (1 - min(1.0, theft * 6)), 3),
            "mean_rc_band": round(2.0 + 4.0 * full, 4),
            "theft_mean": round(theft, 5),
            "split": sp,
        })

    # a small in-domain reselection block (reselect_v3), same shape as real
    ot_tau = defects["orchestral_theft"]["learn_then_test_unit=work_model"]["0.1"]
    reselect = {
        "taus_passed": {"orchestral_theft": {"tau": ot_tau["tau"], "certifiable": ot_tau["certifiable"]}},
        "secondary_def": "mean exact-case stft_distance (lower=better)",
        "note": "synthetic dev seed — all reselection tracks are in-domain (FIT split)",
        "decisions": {},
    }
    for track in ("dectest", "dectest-safe"):
        cand = {}
        for m in MODELS:
            r = _rank(m)
            ot_ucb = round(0.05 + 0.16 * r, 4)
            eh_ucb = round(1.0285, 4)   # event_hole never feasible (uncertified proxy)
            infeasible = (["orchestral_theft"] if ot_ucb > ot_tau["tau"] else []) + ["event_hole"]
            cand[m] = {"hard_failed": [], "ucbs": {"orchestral_theft": ot_ucb, "event_hole": eh_ucb},
                       "infeasible_defects": infeasible, "secondary": round(1.5 + 0.02 * r, 4)}
        reselect["decisions"][track] = {
            "selector_version": "autonomous-selector/v3",
            "certification": "autonomous_proxy_certified",
            "distribution": "in_calibration_domain",
            "taus": {"orchestral_theft": {"tau": ot_tau["tau"], "certifiable": True}},
            "candidates": cand,
            "status": "final",
            "best_available": MODELS[0],
            "failed_gates": [],
            "reason": "clear feasible winner on the certified gate",
        }

    doc = {
        "schema": "audio-extract/calibration/v1",
        "delta": DELTA,
        "uncertainty_u": UNCERTAINTY_U,
        "splits": splits,
        "fit_works": fit_works,
        "held_out_works": held_out_works,
        "truth_definitions": TRUTH,
        "defects": defects,
        "units": units,
        "held_out_violation": held_out_violation,
        "reselect_v3": reselect,
        "generated_by": "web/dev_seed_calibration.py (synthetic; real SeverityMap.fit + "
                        "learn_then_test; shape-identical to audio-extract/calibration/v1)",
    }

    out_dir = os.path.join(args.lib, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "calibration_v1.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    print("wrote %s" % out)
    for name, d in defects.items():
        flags = "/".join(("Y" if d["learn_then_test_unit=work_model"]["%g" % tr]["certifiable"] else "n")
                         for tr in TARGETS)
        strong = d["strongest_certifiable_target"]
        print("  %-18s certifiable@10/15/20%%: %s  strongest=%s  knots=%d"
              % (name, flags, strong, len(d["map_knots"]["xs"])))
    print("serve with:\n  uv run python web/server_v2.py --lib %s" % args.lib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
