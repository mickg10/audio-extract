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
def _exact_case_labels(residual: np.ndarray, target: np.ndarray,
                       injected_vocal: np.ndarray, sr: int) -> dict:
    """Oracle §3: exact event-conditioned labels from the KNOWN A and injected V.
    Event-hole depth (multiband deficit during the injected-vocal events) is the
    PRIMARY event-hole truth; vocal interference = the error's projection onto the
    injected-vocal direction. SI-SDR stays a secondary broad metric."""
    from . import metrics_v2 as m2x

    n = min(len(residual), len(target), len(injected_vocal))
    res, tgt, vin = residual[:n], target[:n], dsp.as2d(injected_vocal)[:n]
    # events = frames where the injected vocal is active (known by construction)
    v_rms = dsp.frame_rms_db(vin, sr)
    thr = float(np.max(v_rms)) - 30.0
    active = v_rms > thr
    events, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= 10:
                events.append((start, i))
            start = None
    if start is not None:
        events.append((start, len(active)))
    ce = dsp.band_envelope_db(res, sr, dsp.PUMP_BANDS)
    te = dsp.band_envelope_db(tgt, sr, dsp.PUMP_BANDS)
    ve = dsp.band_envelope_db(vin, sr, dsp.PUMP_BANDS)
    holes = m2x.event_holes(ce, te, ve, events)
    hole = {o["metric"].split("/")[0]: o["value"] for o in holes if o["available"]}
    # vocal interference: projection of e = residual − target onto the V direction
    e = dsp.mono(res) - dsp.mono(tgt)
    v = dsp.mono(vin)
    v_energy = float(np.dot(v, v)) + 1e-12
    beta = float(np.dot(e, v)) / v_energy
    interf = beta * v
    si_sir = 10 * np.log10((float(np.dot(interf, interf)) + 1e-12)
                           / (float(np.dot(e - interf, e - interf)) + 1e-12))
    return {"event_hole_depth_db": hole.get("event_hole_depth"),
            "event_hole_area": hole.get("event_hole_area"),
            "masked_hole_uncertainty": hole.get("masked_hole_uncertainty"),
            "vocal_interference_ratio": round(abs(beta), 5),
            "vocal_si_sir_db": round(si_sir, 2),
            "n_events": len(events)}


def run_challenges(layout: TrackLayout, sr: int, models: list[str],
                   estimator_factory, persist_outputs: bool = True) -> dict:
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
                # oracle §9.1: persist the per-case output audio (append-only)
                if persist_outputs:
                    odir = cdir / "outputs"
                    odir.mkdir(exist_ok=True)
                    slug = "".join(c if c.isalnum() else "_" for c in model)[:64]
                    opath = odir / f"{slug}.f32.wav"
                    if not opath.exists():
                        sf.write(str(opath), residual.astype("float32"), sr, subtype="FLOAT")
                # oracle §9.2: exact event-hole + interference labels (V known)
                labels = {}
                inj = mix[:n] - tgt[: n]  # injected vocal = mixture − exact target
                try:
                    labels = _exact_case_labels(residual, tgt[:n], inj, sr)
                except Exception as exc:
                    labels = {"label_error": f"{type(exc).__name__}: {exc}"}
                cid = cdir.name.replace("sha256_", "sha256:")
                per_case[cid] = {**err, **labels}
                m2.add_challenge_result(challenge_id=cid, candidate_recipe_id=model,
                                        result={"exact_reference": err,
                                                "exact_labels": labels})
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
        labels = result.get("exact_labels") or {}
        # oracle §3: exact event-conditioned deficit is the PRIMARY event-hole
        # evidence; SI-SDR is only the fallback broad signal when labels absent
        depth = labels.get("event_hole_depth_db")
        if depth is not None:
            c["cells"].setdefault("event_hole", []).append(
                (float(np.clip(depth / 24.0, 0, 1)),
                 u + float(labels.get("masked_hole_uncertainty") or 0.0)))
        else:
            c["cells"].setdefault("event_hole", []).append(
                (severity_from_si_sdr(er["si_sdr_db"]), u))
        interf = labels.get("vocal_interference_ratio")
        if interf is not None:
            c["cells"].setdefault("vocal_leakage", []).append(
                (float(np.clip(interf, 0, 1)), u))
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


def revalidate_finalist(layout: TrackLayout, sr: int, candidate_recipe_id: str,
                        *, source_peak: float | None = None) -> dict:
    """v3 review §11: before delivery, extract the ORIGINAL mined passage intervals
    from the finalist's full-track audio and rerun the hard gates + core damage
    metrics there. A full render that regresses on its own decisive passages is
    rejected, not shipped."""
    import soundfile as sf

    from . import metrics as mx
    from . import metrics_v2 as m2

    cand_wav = layout.candidate_dir(candidate_recipe_id) / "output.f32.wav"
    if not cand_wav.exists():
        return {"ok": False, "reason": f"candidate {candidate_recipe_id} not in store"}
    audio, csr = sf.read(str(cand_wav), dtype="float64", always_2d=True)
    if int(csr) != sr:
        return {"ok": False, "reason": f"candidate sr {csr} != canonical {sr}"}
    pj = layout.passages_dir / "passages.v1.json"
    passages = json.loads(pj.read_text())["passages"] if pj.exists() else []
    if not passages:
        return {"ok": False, "reason": "no mined passages to revalidate against"}

    src = layout.source_dir / "canonical.f32.wav"
    reference = None
    if src.exists():
        reference, _ = sf.read(str(src), dtype="float64", always_2d=True)
        if source_peak is None and reference.size:
            source_peak = float(np.max(np.abs(reference)))

    per_passage, failures = [], []
    for p in passages:
        s, e = p["start_sample"], min(p["end_sample"], audio.shape[0])
        if e - s < sr // 4:
            continue
        seg = audio[s:e]
        hc = mx.hard_checks(seg, sr, source_peak=source_peak)
        row = {"passage_id": p["passage_id"], "tags": p["tags"],
               "hard_ok": hc["ok"], "problems": hc["problems"]}
        if reference is not None and "no_vocal_control" in p["tags"]:
            # on a genuine control the finalist should preserve the original mix
            err = m2.fullness_v2(seg, reference[s:e], sr)
            row["control_band_deficit_db"] = err[0]["value"]
            if err[0]["value"] is not None and err[0]["value"] > 6.0:
                failures.append(f"{p['passage_id']}: control deficit {err[0]['value']:.1f} dB")
        if not hc["ok"]:
            failures.append(f"{p['passage_id']}: {hc['problems']}")
        per_passage.append(row)

    return {"ok": not failures, "passages_checked": len(per_passage),
            "failures": failures, "per_passage": per_passage}


def finalize_and_deliver(layout: TrackLayout, sr: int, candidate_recipe_id: str,
                         *, target_dbfs: float = -5.0, bitrate: str = "256k",
                         code_commit: str = "") -> dict:
    """The last mile: revalidate the finalist on its mined passages, then run the
    delivery DAG. A revalidation failure leaves the run NOT complete."""
    from .delivery import deliver
    from .manifest import Manifest

    reval = revalidate_finalist(layout, sr, candidate_recipe_id)
    if not reval["ok"]:
        with Manifest(layout.manifest_sqlite) as man:
            man.set_state(layout.track_id, "FINAL_QC")
        return {"status": "revalidation_failed", "revalidation": reval}
    src_record = json.loads((layout.source_dir / "source.json").read_text())
    report = deliver(layout, src_record, candidate_recipe_id,
                     target_dbfs=target_dbfs, bitrate=bitrate, code_commit=code_commit)
    return {"status": "delivered", "revalidation": reval, "delivery": report}


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
