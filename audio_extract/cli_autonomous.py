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


class UnsupportedConstructionError(RuntimeError):
    """A recipe kind cannot be materialized (e.g. a 'native' construction on a
    checkpoint with no native instrumental stem). Raised instead of silently
    running a different construction under the requested identity (oracle §2)."""

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
    # §8.2: JOINT source regression Â ≈ α·A + β·V + e_perp (mono). Projecting the
    # error onto V alone is biased when A and V correlate; the 2-var least-squares
    # fit decomposes cleanly into accompaniment scaling (α), vocal interference (β),
    # and orthogonal artifact (e_perp).
    a = dsp.mono(tgt)
    v = dsp.mono(vin)
    ahat = dsp.mono(res)
    G = np.array([[np.dot(a, a), np.dot(a, v)], [np.dot(a, v), np.dot(v, v)]]) + 1e-9 * np.eye(2)
    rhs = np.array([np.dot(a, ahat), np.dot(v, ahat)])
    alpha, beta = np.linalg.solve(G, rhs)
    interf = beta * v
    e_perp = ahat - alpha * a - interf
    e_target = alpha * a
    si_sir = 10 * np.log10((float(np.dot(e_target, e_target)) + 1e-12)
                           / (float(np.dot(interf, interf)) + 1e-12))
    artifact_ratio = float(np.sqrt(np.dot(e_perp, e_perp) / (np.dot(a, a) + 1e-12)))
    return {"event_hole_depth_db": hole.get("event_hole_depth"),
            "event_hole_area": hole.get("event_hole_area"),
            "masked_hole_uncertainty": hole.get("masked_hole_uncertainty"),
            "accompaniment_scaling": round(float(alpha), 5),
            "vocal_interference_ratio": round(abs(float(beta)), 5),
            "orthogonal_artifact_ratio": round(artifact_ratio, 5),
            "vocal_si_sir_db": round(float(si_sir), 2),
            "n_events": len(events)}


def recipe_spec_id(spec: dict) -> str:
    """Stable recipe id for a bake-off spec (oracle §2 — evaluate recipes, not model
    names). Each construction is a distinct artifact with its own identity.

    §2 identity fix: canonicalize (member, weight) PAIRS together so a weight can
    never be misassociated with the wrong member, and so [MDX,Mel]+[.8,.2] and
    [Mel,MDX]+[.8,.2] get distinct ids (they are semantically different weighted
    means). The median is weight-symmetric, so its members canonicalize alone."""
    models = list(spec.get("models", []))
    weights = list(spec.get("weights", []))
    if weights and len(weights) == len(models):
        pairs = sorted((m, int(round(w * 1000))) for m, w in zip(models, weights))
        members = [{"model": m, "weight_milli": w} for m, w in pairs]
    else:
        members = [{"model": m} for m in sorted(models)]
    canon = {"kind": spec["kind"], "members": members,
             "algo": spec.get("algo", ""), "overlap": spec.get("overlap", 0)}
    return "sha256:" + hashlib.sha256(
        b"audio-extract-recipe-spec-v1\x00" + json.dumps(canon, sort_keys=True).encode()
    ).hexdigest()


def compose_accompaniment(spec: dict, estimator_factory):
    """Return ``estimate_acc(mix) -> accompaniment`` for a recipe spec.

    ``residual``: A = M − V̂(model);  ``native``: A = model's instrumental stem
    (falls back to residual if the checkpoint has no native output);
    ``ensemble_residual``: A = M − aggregate(V̂ over models) with median/weighted-mean.
    The theft/label layer derives the REMOVED material as M − A for ANY recipe, so
    every construction is scored exactly as it would be delivered (§10)."""
    kind = spec["kind"]
    models = spec["models"]

    def estimate_acc(mix):
        mix = dsp.as2d(mix)
        if kind == "native":
            stems = estimator_factory(models[0])(mix)
            if "instrumental" not in stems:
                # §2: never silently run residual under a 'native' identity
                raise UnsupportedConstructionError(
                    f"model {models[0]!r} has no native instrumental stem "
                    f"(emits {sorted(stems)}); a native recipe cannot be materialized")
            inst = dsp.as2d(stems["instrumental"])
            n = min(len(mix), len(inst))
            return inst[:n]
        if kind == "ensemble_residual":
            vs = []
            for m in models:
                v = dsp.as2d(estimator_factory(m)(mix).get("vocals", np.zeros_like(mix)))
                vs.append(v)
            n = min(len(mix), *(len(v) for v in vs))
            stack = np.stack([v[:n] for v in vs], axis=0)
            if spec.get("algo", "median") == "median":
                v_agg = np.median(stack, axis=0)
            else:
                w = np.asarray(spec.get("weights") or [1.0 / len(vs)] * len(vs))
                w = w / w.sum()
                v_agg = np.tensordot(w, stack, axes=(0, 0))
            return mix[:n] - v_agg
        # residual (default)
        v = dsp.as2d(estimator_factory(models[0])(mix).get("vocals", np.zeros_like(mix)))
        n = min(len(mix), len(v))
        return mix[:n] - v[:n]

    return estimate_acc


def run_challenges(layout: TrackLayout, sr: int, models: list[str] | None = None,
                   estimator_factory=None, persist_outputs: bool = True,
                   recipe_specs: list[dict] | None = None) -> dict:
    """Score candidates over the built challenges. Pass ``recipe_specs`` (the
    bake-off path — each keyed by its recipe id) or ``models`` (legacy residual,
    keyed by model name)."""
    import soundfile as sf

    chdir = layout.root / "challenges"
    case_dirs = sorted(d for d in chdir.glob("sha256_*") if (d / "mixture.f32.wav").exists())
    if not case_dirs:
        return {"ran": 0, "reason": "no built challenges (run `challenges build` first)"}
    controls = genuine_controls(layout, sr)

    # unify the two entry points into (key, estimate_acc) candidates
    candidates: list[tuple[str, object]] = []
    if recipe_specs:
        for spec in recipe_specs:
            rid = spec.get("recipe_id") or recipe_spec_id(spec)
            candidates.append((rid, compose_accompaniment(spec, estimator_factory)))
    else:
        for model in (models or []):
            def _resid(mix, _m=model):
                v = dsp.as2d(estimator_factory(_m)(mix).get("vocals", np.zeros_like(dsp.as2d(mix))))
                mx = dsp.as2d(mix)
                n = min(len(mx), len(v))
                return mx[:n] - v[:n]
            candidates.append((model, _resid))

    matrix: dict[str, dict] = {}
    with ManifestV2(layout.manifest_sqlite) as m2:
        for key, estimate_acc in candidates:
            model = key
            per_case = {}
            for cdir in case_dirs:
                mix, _ = sf.read(str(cdir / "mixture.f32.wav"), dtype="float64", always_2d=True)
                tgt, _ = sf.read(str(cdir / "target.f32.wav"), dtype="float64", always_2d=True)
                residual = dsp.as2d(estimate_acc(mix))
                n = min(len(mix), len(residual))
                residual = residual[:n]
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
                # the REMOVED material for ANY recipe is M − A (§10): score the
                # actual construction, not just a model's vocal stem.
                def _removed(a, _acc=estimate_acc):
                    a2 = dsp.as2d(a)
                    acc = dsp.as2d(_acc(a2))
                    nn = min(len(a2), len(acc))
                    return a2[:nn] - acc[:nn]
                theft_rows = ch.run_no_vocal_theft_assay(_removed, controls, sr)
                theft = float(np.mean([r["theft_broadband"] for r in theft_rows]))
                # also compare the actual accompaniment construction on the control
                # directly with A (catches phase/compensation/timing errors).
                res_errs = []
                for ctl_id, a_ctl in controls:
                    a2 = dsp.as2d(a_ctl)
                    acc = dsp.as2d(estimate_acc(a2))
                    nn = min(len(a2), len(acc))
                    res_errs.append({"control_id": ctl_id,
                                     **ch.exact_reference_error(acc[:nn], a2[:nn], sr)})
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
CERTIFIED_SELECTOR = "autonomous-selector/v3"


def load_calibration(path: str | Path, target_risk: str = "0.1") -> dict:
    """Load a FROZEN calibration artifact (oracle P0 §1): the only certified path
    consumes this, never the dev seed scalings. Returns per-defect Learn-Then-Test
    thresholds at ``target_risk`` + frozen SeverityMaps + the artifact hash that a
    decision must record and a delivery must re-verify."""
    from . import selector as sel

    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw)
    sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    taus: dict[str, dict] = {}
    maps: dict[str, sel.SeverityMap] = {}
    for defect, d in doc.get("defects", {}).items():
        ltt = d.get("learn_then_test_unit=work_model", {})
        row = ltt.get(target_risk) or ltt.get(str(float(target_risk)))
        taus[defect] = row or {"tau": 0.0, "risk_ucb": 1.0, "n_groups": 0, "certifiable": False}
        knots = d.get("map_knots")
        if knots and knots.get("xs"):
            import numpy as _np
            maps[defect] = sel.SeverityMap(xs=_np.asarray(knots["xs"], float),
                                           ys=_np.asarray(knots["ys"], float))
    return {"taus": taus, "severity_maps": maps, "calibration_sha256": sha,
            "target_risk": target_risk, "schema": doc.get("schema"),
            "path": str(p)}


# raw proxy per (defect): what the frozen SeverityMap was/should be fit on.
def _raw_proxies(er: dict, labels: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    depth = labels.get("event_hole_depth_db")
    if depth is not None:
        out["event_hole"] = float(depth)               # exact label (oracle §3 primary)
    elif er.get("si_sdr_db") is not None:
        out["event_hole"] = max(0.0, 30.0 - float(er["si_sdr_db"]))  # broad fallback
    if labels.get("vocal_interference_ratio") is not None:
        out["vocal_leakage"] = float(labels["vocal_interference_ratio"])
    if er.get("band_envelope_err_db") is not None:
        out["fullness"] = float(er["band_envelope_err_db"])
    if er.get("stereo_width_err") is not None:
        out["stereo"] = float(er["stereo_width_err"])
    return out


def severity_cells_from_store(layout: TrackLayout, *, u: float = 0.05,
                              calibration: dict | None = None) -> dict[str, dict]:
    """Assemble selector input from stored challenge_result rows.

    ``calibration`` present  → severities come from the FROZEN SeverityMaps
    (the certified path). Absent → dev seed scalings (uncertified preview only,
    per oracle §1 — these must never emit a certified ``final``)."""
    maps = (calibration or {}).get("severity_maps") or {}

    def _sev(defect: str, raw: float) -> float:
        if defect in maps:
            return maps[defect].apply(raw)
        # dev seed scalings (uncertified)
        if defect == "event_hole":
            return float(np.clip(raw / 24.0, 0, 1))
        if defect == "fullness":
            return float(np.clip(raw / 30.0, 0, 1))
        return float(np.clip(raw, 0, 1))
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
    for challenge_id, candidate_recipe_id, result in rows:
        er = result.get("exact_reference")
        if not er:
            continue
        labels = result.get("exact_labels") or {}
        # oracle P0 §1: aggregate by the CANDIDATE RECIPE across all challenges —
        # NOT the challenge id. (Both ids are sha256:… so the old
        # "starts-with-sha256" guard silently picked the challenge id, giving one
        # pseudo-candidate per challenge.)
        key = candidate_recipe_id
        c = cands.setdefault(key, {"cells": {}, "gates": {}, "recipe_id": candidate_recipe_id})
        raws = _raw_proxies(er, labels)
        for defect, raw in raws.items():
            uu = u + (float(labels.get("masked_hole_uncertainty") or 0.0)
                      if defect == "event_hole" else 0.0)
            c["cells"].setdefault(defect, []).append((_sev(defect, raw), uu))
        # §4 second vocal-leakage gate: |β| (calibrated cell above) measures how
        # much voice is retained; SI-SIR measures whether it's AUDIBLE vs the
        # orchestra. A quiet retained voice and a dominant one aren't equivalent.
        sir = labels.get("vocal_si_sir_db")
        if sir is not None:
            # audibility severity: SI-SIR >= +6 dB (voice well below orchestra) -> 0;
            # <= -6 dB (voice at/above orchestra) -> 1. A fixed physical bound.
            aud = float(np.clip((6.0 - float(sir)) / 12.0, 0.0, 1.0))
            c.setdefault("_aud", []).append(aud)
    for r in theft_rows:
        key = r["candidate_recipe_id"]
        theft_mean = float(json.loads(r["result_json"]).get("theft_mean", 0.0))
        theft_sev = _sev("orchestral_theft", theft_mean)
        c = cands.setdefault(key, {"cells": {}, "gates": {}, "recipe_id": key})
        c["cells"].setdefault("orchestral_theft", []).append((theft_sev, u))
        c["gates"]["orchestral_theft"] = theft_sev
    for c in cands.values():
        if "event_hole" in c["cells"]:
            c["gates"]["event_hole"] = max(s for s, _ in c["cells"]["event_hole"])
        # §4 audibility hard gate = worst-case retained-voice audibility
        aud = c.pop("_aud", None)
        if aud:
            c["gates"]["vocal_audibility"] = max(aud)
        # secondary = mean fullness severity (spectral fidelity tie-breaker)
        full = c["cells"].get("fullness")
        if full:
            c["secondary"] = float(np.mean([s for s, _ in full]))
    return cands


def _rollout_level(decision: dict, calibration: dict | None) -> str:
    """oracle §5 rollout ladder. Certified paths require the frozen artifact AND a
    real work-level evaluation (not present for a single-track run) — so the most a
    single-track certified selection can claim here is exact_benchmark_qualified;
    everything else is an explicitly uncertified engineering_preview."""
    if calibration is None or decision.get("selector_version") != CERTIFIED_SELECTOR:
        return "engineering_preview"
    if decision.get("status") != "final":
        return "engineering_preview"
    return "exact_benchmark_qualified"


# Critical axes we have no certified evidence for yet — scope-limited (declared,
# never silently passed) so a single-track run tops out at exact_benchmark_qualified.
DEFAULT_SCOPE_LIMITED = frozenset({"severe_artifact"})


def select_autonomous(layout: TrackLayout, *, calibration_path: str | Path | None = None,
                      target_risk: str = "0.1", task: str | None = None,
                      domain: str | None = None,
                      scope_limited: frozenset[str] = DEFAULT_SCOPE_LIMITED,
                      **selector_kwargs) -> dict:
    """Certified path when ``calibration_path`` is given: frozen artifact →
    select_v3 → immutable decision. Without it, a dev PREVIEW that can never emit a
    certified final (oracle P0 §1)."""
    from . import selector as sel
    from .manifest import Manifest

    calibration = load_calibration(calibration_path, target_risk) if calibration_path else None
    cands = severity_cells_from_store(layout, calibration=calibration)
    if not cands:
        return {"status": "no_acceptable_candidate", "best_available": None,
                "reason": "no challenge results stored (run `challenges run` first)"}

    if calibration is not None:
        decision = sel.select_v3(cands, calibration["taus"],
                                 scope_limited=scope_limited, **selector_kwargs)
        decision["certification_inputs"] = {
            "calibration_sha256": calibration["calibration_sha256"],
            "target_risk": target_risk, "task": task, "domain": domain,
            "selector_version": CERTIFIED_SELECTOR,
            "candidate_recipe_ids": {k: v.get("recipe_id") for k, v in cands.items()},
        }
    else:
        decision = sel.select(cands, **selector_kwargs)
        decision["certification"] = "none"          # a preview may never certify

    decision["rollout_level"] = _rollout_level(decision, calibration)
    # §8: explicit certification SCOPE, not a bare boolean. A single-track exact run
    # never claims transfer/stereo/hall/production — those need the work-level path.
    lvl = decision["rollout_level"]
    decision["certification_scope"] = {
        "level": lvl,
        "risk_claim": None,                          # no work-level risk claim here
        "transfer_supported": False,
        "stereo_supported": False,
        "hall_supported": False,
        "scope_limited_defects": sorted(scope_limited) if calibration is not None else [],
    }
    decision_id = record_decision(layout, decision)
    decision["decision_id"] = decision_id
    if decision.get("status") == "final":
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

    # excerpt-screen outputs for scope/chunk-context regression comparison (§10.2):
    # the same recipe rendered as an independent excerpt vs cut from the full track.
    excerpts = {}
    exc_dir = layout.candidate_dir(candidate_recipe_id) / "excerpts"
    if exc_dir.exists():
        for w in exc_dir.glob("*.f32.wav"):
            try:
                excerpts[w.stem], _ = sf.read(str(w), dtype="float64", always_2d=True)
            except Exception:
                pass

    per_passage, failures = [], []
    for p in passages:
        s, e = p["start_sample"], min(p["end_sample"], audio.shape[0])
        if e - s < sr // 4:
            continue
        seg = audio[s:e]
        tags = p["tags"]
        hc = mx.hard_checks(seg, sr, source_peak=source_peak)
        row = {"passage_id": p["passage_id"], "tags": tags,
               "hard_ok": hc["ok"], "problems": hc["problems"]}
        if not hc["ok"]:
            failures.append(f"{p['passage_id']}: {hc['problems']}")
        if reference is not None:
            ref_seg = reference[s:e]
            # control retention (§10.3): a genuine no-vocal control must be preserved
            if "no_vocal_control" in tags:
                err = m2.fullness_v2(seg, ref_seg, sr)
                row["control_band_deficit_db"] = err[0]["value"]
                if err[0]["value"] is not None and err[0]["value"] > 6.0:
                    failures.append(f"{p['passage_id']}: control deficit {err[0]['value']:.1f} dB")
            # scope/chunk-context regression (§10.2): full-track cut vs excerpt render
            if p["passage_id"] in excerpts:
                ex = excerpts[p["passage_id"]]
                nn = min(len(seg), len(ex))
                drift = float(np.sqrt(np.mean((seg[:nn] - ex[:nn]) ** 2)) /
                              (np.sqrt(np.mean(ex[:nn] ** 2)) + 1e-9))
                row["excerpt_scope_drift"] = round(drift, 4)
                if drift > 0.15:
                    failures.append(f"{p['passage_id']}: full-track render drifts "
                                    f"{drift:.2f} from the screened excerpt")
        per_passage.append(row)

    return {"ok": not failures, "passages_checked": len(per_passage),
            "failures": failures, "per_passage": per_passage,
            "excerpt_comparisons": sum(1 for r in per_passage if "excerpt_scope_drift" in r)}


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


def _load_decision(layout: TrackLayout, decision_id: str) -> dict | None:
    with ManifestV2(layout.manifest_sqlite) as m2:
        row = m2._conn.execute(
            "SELECT * FROM selection_decision WHERE decision_id=?", (decision_id,)).fetchone()
    return dict(row) if row is not None else None


def finalize_from_decision(layout: TrackLayout, sr: int, decision_id: str, *,
                           calibration_path: str | Path | None = None,
                           target_dbfs: float = -5.0, bitrate: str = "256k",
                           code_commit: str = "") -> dict:
    """oracle P0 §9: a CERTIFIED render is bound to an immutable selection decision,
    never a free-floating candidate id. Verify status/selector/calibration/recipe
    freshness/gates BEFORE rendering; refuse otherwise."""
    row = _load_decision(layout, decision_id)
    if row is None:
        return {"status": "refused", "reason": f"decision {decision_id} not found"}
    report = json.loads(row["report_json"])
    checks: list[str] = []
    if row["status"] != "final":
        checks.append(f"decision status is {row['status']!r}, not 'final'")
    if report.get("selector_version") != CERTIFIED_SELECTOR:
        checks.append(f"decision from {report.get('selector_version')!r}, not the certified selector")
    ci = report.get("certification_inputs", {})
    # §8: the frozen calibration recheck is MANDATORY for a certified decision —
    # the artifact must be resolvable and rehash to the recorded value; a missing
    # or mismatched artifact refuses the render (never a silent skip).
    if calibration_path is None:
        checks.append("no calibration artifact supplied; a certified render must "
                      "re-verify the frozen calibration hash (§8)")
    else:
        want = load_calibration(calibration_path).get("calibration_sha256")
        if ci.get("calibration_sha256") != want:
            checks.append("calibration hash mismatch (decision is stale vs the supplied artifact)")
    recipe_id = row["candidate_recipe_id"]
    if not recipe_id:
        checks.append("decision names no candidate recipe id")
    elif not (layout.candidate_dir(recipe_id) / "output.f32.wav").exists():
        checks.append(f"decision's candidate {recipe_id} is not in the store")
    if checks:
        return {"status": "refused", "decision_id": decision_id, "checks_failed": checks}
    res = finalize_and_deliver(layout, sr, recipe_id, target_dbfs=target_dbfs,
                               bitrate=bitrate, code_commit=code_commit)
    res["decision_id"] = decision_id
    # §8: carry the explicit certification scope, not a bare boolean
    res["certification_scope"] = report.get("certification_scope",
                                            {"level": report.get("rollout_level")})
    return res


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
