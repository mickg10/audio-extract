"""Autonomous-run CLI core (v2.1 WP12 / §17) — challenges build/run/report,
``select autonomous``, and ``run report``.

The heavy logic is factored into plain functions taking an *estimator factory*
(name → vocal-estimator callable), so tests drive the full flow with synthetic
estimators and the CLI wires the real ``Separator``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import challenges as ch, dsp
from .manifest_v2 import ManifestV2
from .storage import TrackLayout

# Seed severity mapping (recalibrated by SeverityMap once ladders accumulate):
# exact-target SI-SDR >= 30 dB is "no damage"; theft/bleed ratios map directly.
def severity_from_si_sdr(si_sdr_db: float) -> float:
    return float(np.clip(1.0 - si_sdr_db / 30.0, 0.0, 1.0))


def severity_from_band_err(band_err_db: float) -> float:
    return float(np.clip(band_err_db / 30.0, 0.0, 1.0))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_decision(layout: TrackLayout, decision: dict) -> str:
    """Persist a terminal decision as a selection_decision row; returns its id."""
    body = {k: v for k, v in decision.items()}
    # deterministic JSON (sorted keys) — decision reports carry measured floats,
    # which the strict recipe canonicalizer rejects by design; JCS is for recipes.
    payload = json.dumps({"decision": body, "track": layout.track_id},
                         sort_keys=True, default=str).encode()
    did = "sd_" + hashlib.sha256(payload).hexdigest()[:20]
    with ManifestV2(layout.manifest_sqlite) as m2:
        m2.add_selection_decision(
            decision_id=did, status=decision.get("status", "?"),
            selector_version=decision.get("selector_version", "conductor/v1"),
            report=body, created_at=_now(),
            candidate_recipe_id=decision.get("candidate_id"))
    return did


# ---------------------------------------------------------------------------
# challenges build
# ---------------------------------------------------------------------------
def genuine_controls(layout: TrackLayout, sr: int, *, max_ratio: float = 0.15
                     ) -> list[tuple[str, np.ndarray]]:
    """No-vocal control passages that PASSED the absolute vocal-absence gate."""
    import soundfile as sf

    pj = layout.passages_dir / "passages.v1.json"
    if not pj.exists():
        return []
    doc = json.loads(pj.read_text())
    audio, _ = sf.read(str(layout.source_dir / "canonical.f32.wav"),
                       dtype="float64", always_2d=True)
    out = []
    for p in doc["passages"]:
        if "no_vocal_control" not in p["tags"]:
            continue
        ratio = p.get("features", {}).get("vocal_energy_ratio")
        if ratio is None or ratio > max_ratio:
            continue
        out.append((p["passage_id"], audio[p["start_sample"]:p["end_sample"]]))
    return out


def vocal_segments(layout: TrackLayout, sr: int, *, n: int = 2,
                   seg_s: float = 4.0) -> list[tuple[str, np.ndarray]]:
    """Highest-RMS clean-vocal segments from the provisional vocal (supplemental
    same-track source; a licensed library takes precedence when configured)."""
    import soundfile as sf

    pv = layout.source_dir / "provisional_vocal.f32.wav"
    if not pv.exists():
        return []
    v, _ = sf.read(str(pv), dtype="float64", always_2d=True)
    seg = int(seg_s * sr)
    if len(v) < seg:
        return [("voc_full", v)]
    hop = seg // 2
    rms = [(float(np.sqrt(np.mean(v[s:s + seg] ** 2))), s)
           for s in range(0, len(v) - seg, hop)]
    rms.sort(reverse=True)
    picked, out = [], []
    for r, s in rms:
        if len(out) >= n:
            break
        if any(abs(s - q) < seg for q in picked):
            continue
        picked.append(s)
        out.append((f"voc_{s}", v[s:s + seg]))
    return out


def build_challenges(layout: TrackLayout, sr: int, *, count: int = 8,
                     seed: int = 0) -> dict:
    """Build remix cases from genuine controls + vocal segments; write each case's
    mixture/target under ``challenges/<id>/`` and record challenge_case rows."""
    import soundfile as sf

    controls = genuine_controls(layout, sr)
    vocals = vocal_segments(layout, sr)
    if not controls:
        return {"built": 0, "reason": "no genuine no-vocal controls (gate declined); "
                                      "challenges require vocal-free passages"}
    if not vocals:
        return {"built": 0, "reason": "no vocal source available"}
    rt60 = 1.2
    tail = controls[0][1][-int(1.0 * sr):]
    if len(tail) > sr // 2:
        rt60 = ch.estimate_rt60_from_tail(tail, sr)
    cases = ch.build_track_remix_cases(controls, vocals, sr, rt60_s=rt60,
                                       seed=seed, max_cases=count)
    chdir = layout.root / "challenges"
    ids = []
    with ManifestV2(layout.manifest_sqlite) as m2:
        for case in cases:
            cdir = chdir / case.challenge_id.replace(":", "_")
            cdir.mkdir(parents=True, exist_ok=True)
            if not (cdir / "mixture.f32.wav").exists():
                sf.write(str(cdir / "mixture.f32.wav"),
                         case.mixture.astype("float32"), sr, subtype="FLOAT")
                sf.write(str(cdir / "target.f32.wav"),
                         case.target.astype("float32"), sr, subtype="FLOAT")
                (cdir / "recipe.json").write_text(json.dumps(case.recipe, indent=2))
                (cdir / "class.json").write_text(json.dumps(case.klass, indent=2))
            m2.add_challenge_case(
                challenge_id=case.challenge_id, challenge_type="track_remix",
                mixture_node_id=case.challenge_id,
                target_node_id=case.challenge_id + "#target",
                recipe=case.recipe, klass=case.klass)
            ids.append(case.challenge_id)
    return {"built": len(ids), "rt60_s": round(rt60, 2), "challenge_ids": ids,
            "controls": [c[0] for c in controls], "vocal_sources": [v[0] for v in vocals]}


# ---------------------------------------------------------------------------
# challenges run  (estimator_factory: model_name -> fn(audio)->{"vocals": arr,...})
# ---------------------------------------------------------------------------
def run_challenges(layout: TrackLayout, sr: int, models: list[str],
                   estimator_factory) -> dict:
    import soundfile as sf

    chdir = layout.root / "challenges"
    case_dirs = sorted(d for d in chdir.glob("sha256_*") if (d / "mixture.f32.wav").exists())
    if not case_dirs:
        return {"ran": 0, "reason": "no built challenges (run `challenges build` first)"}
    controls = genuine_controls(layout, sr)
    matrix: dict[str, dict] = {}
    with ManifestV2(layout.manifest_sqlite) as m2:
        for model in models:
            estimate = estimator_factory(model)
            per_case = {}
            for cdir in case_dirs:
                mix, _ = sf.read(str(cdir / "mixture.f32.wav"), dtype="float64", always_2d=True)
                tgt, _ = sf.read(str(cdir / "target.f32.wav"), dtype="float64", always_2d=True)
                stems = estimate(mix)
                v_hat = dsp.as2d(stems.get("vocals", np.zeros_like(mix)))
                n = min(len(mix), len(v_hat))
                residual = mix[:n] - v_hat[:n]
                err = ch.exact_reference_error(residual, tgt, sr)
                cid = cdir.name.replace("sha256_", "sha256:")
                per_case[cid] = err
                m2.add_challenge_result(challenge_id=cid, candidate_recipe_id=model,
                                        result={"exact_reference": err})
            theft = None
            if controls:
                theft_rows = ch.run_no_vocal_theft_assay(
                    lambda a: dsp.as2d(estimator_factory(model)(dsp.as2d(a)).get(
                        "vocals", np.zeros_like(dsp.as2d(a)))), controls, sr)
                theft = float(np.mean([r["theft_broadband"] for r in theft_rows]))
                # v3 review §10: also evaluate the ACTUAL residual construction on
                # the control — Â_res(A) = A − V̂(A) compared directly with A. This
                # catches phase/compensation/timing errors invisible in stem energy.
                res_errs = []
                for ctl_id, a_ctl in controls:
                    a2 = dsp.as2d(a_ctl)
                    v_hat = dsp.as2d(estimate(a2).get("vocals", np.zeros_like(a2)))
                    nn = min(len(a2), len(v_hat))
                    res_errs.append({"control_id": ctl_id,
                                     **ch.exact_reference_error(a2[:nn] - v_hat[:nn], a2[:nn], sr)})
                m2.add_challenge_result(challenge_id="theft_assay",
                                        candidate_recipe_id=model,
                                        result={"theft_rows": theft_rows,
                                                "theft_mean": theft,
                                                "residual_construction": res_errs})
            matrix[model] = {"cases": per_case, "theft_mean": theft}
    return {"ran": len(case_dirs), "models": list(matrix), "matrix": matrix}


# ---------------------------------------------------------------------------
# select autonomous  (from stored challenge results)
# ---------------------------------------------------------------------------
def severity_cells_from_store(layout: TrackLayout, *, u: float = 0.05) -> dict[str, dict]:
    """Assemble selector input from stored challenge_result rows."""
    with ManifestV2(layout.manifest_sqlite) as m2:
        rows = []
        for case in m2.challenge_cases():
            rows.extend(
                (case["challenge_id"], r["candidate_recipe_id"], json.loads(r["result_json"]))
                for r in m2._conn.execute(
                    "SELECT * FROM challenge_result WHERE challenge_id=?",
                    (case["challenge_id"],)))
        theft_rows = list(m2._conn.execute(
            "SELECT * FROM challenge_result WHERE challenge_id='theft_assay'"))
    cands: dict[str, dict] = {}
    for _cid, model, result in rows:
        er = result.get("exact_reference")
        if not er:
            continue
        c = cands.setdefault(model, {"cells": {}, "gates": {}})
        c["cells"].setdefault("event_hole", []).append(
            (severity_from_si_sdr(er["si_sdr_db"]), u))
        c["cells"].setdefault("fullness", []).append(
            (severity_from_band_err(er["band_envelope_err_db"]), u))
        c["cells"].setdefault("stereo", []).append(
            (float(np.clip(er["stereo_width_err"], 0, 1)), u))
    for r in theft_rows:
        model = r["candidate_recipe_id"]
        theft = float(np.clip(json.loads(r["result_json"]).get("theft_mean", 0.0), 0, 1))
        c = cands.setdefault(model, {"cells": {}, "gates": {}})
        c["cells"].setdefault("orchestral_theft", []).append((theft, u))
        c["gates"]["orchestral_theft"] = theft
    for c in cands.values():
        if "event_hole" in c["cells"]:
            c["gates"]["event_hole"] = max(s for s, _ in c["cells"]["event_hole"])
    return cands


def select_autonomous(layout: TrackLayout, **selector_kwargs) -> dict:
    from . import selector as sel
    from .manifest import Manifest

    cands = severity_cells_from_store(layout)
    if not cands:
        return {"status": "no_acceptable_candidate", "best_available": None,
                "reason": "no challenge results stored (run `challenges run` first)"}
    decision = sel.select(cands, **selector_kwargs)
    decision_id = record_decision(layout, decision)
    decision["decision_id"] = decision_id
    if decision["status"] == "final":
        with Manifest(layout.manifest_sqlite) as man:
            man.set_state(layout.track_id, "FINALIST_SELECTED")
    return decision


def run_report(layout: TrackLayout) -> dict:
    """§17.3 consolidated report from stored records."""
    with ManifestV2(layout.manifest_sqlite) as m2:
        dec_row = m2._conn.execute(
            "SELECT * FROM selection_decision ORDER BY created_at DESC LIMIT 1").fetchone()
        n_cases = len(m2.challenge_cases())
    if dec_row is None:
        return {"status": "none", "reason": "no selection decision recorded"}
    report = json.loads(dec_row["report_json"])
    winner = dec_row["candidate_recipe_id"]
    cand_rep = report.get("candidates", {})
    runner = None
    if winner and len(cand_rep) > 1:
        others = sorted((c for c in cand_rep if c != winner),
                        key=lambda c: cand_rep[c].get("risk_mean", 1.0))
        runner = others[0] if others else None
    return {
        "status": dec_row["status"],
        "decision_id": dec_row["decision_id"],
        "candidate": {"id": winner, **cand_rep.get(winner, {})} if winner else {},
        "hard_gates": {c: r.get("failed_gates", []) for c, r in cand_rep.items()},
        "risk": {k: cand_rep.get(winner, {}).get(f"risk_{k}") for k in ("lower", "mean", "upper")} if winner else {},
        "runner_up": {"id": runner, **cand_rep.get(runner, {})} if runner else {},
        "challenge_summary": {"cases": n_cases},
        "selector_version": dec_row["selector_version"],
        "reason": report.get("reason"),
    }
