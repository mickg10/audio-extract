"""Judge training pipeline: assemble (features, exact-label targets) grouped by work,
train a boosted-tree baseline, evaluate leave-one-work-out.

Ties the two halves of the workbench together:
* ``judge_features_v2.source_aware_features`` — the reference-free inputs (M, Y, D).
* ``judge_labels.local_source_coordinate_labels`` — the exact α/β/R/κ tile labels
  (needs the known accompaniment A and voice V).

The baseline is a per-head HistGradientBoosting regressor with LeaveOneGroupOut CV
over WORKS — the honest test (never a random clip split, per both reviewers). A
neural frozen-encoder student is a later step; this establishes the interpretable
floor a student must beat on held-out works.
"""

from __future__ import annotations

import numpy as np

from . import judge_features_v2 as jf2
from . import judge_labels as jl

# Regression targets the judge predicts, aggregated over a work's available tiles.
# Each maps to (label field, aggregator) — worst-case aggregators, never a mean.
TARGETS = {
    "event_hole_db_p90": ("accompaniment_hole_db", lambda v: _q(v, 90)),
    "event_hole_db_max": ("accompaniment_hole_db", lambda v: _q(v, 100)),
    "retained_voice_db_p90": ("retained_voice_energy_db", lambda v: _q(v, 90)),
    "retained_voice_coef_p90": ("retained_voice_coefficient", lambda v: _q(v, 90)),
    "artifact_ratio_p90": ("orthogonal_artifact_ratio", lambda v: _q(v, 90)),
    "alpha_error_p90": ("alpha_error_abs", lambda v: _q(v, 90)),
}


# Heads that are physically nonnegative (a hole depth, an artifact ratio, an |α−1|, a
# retained-voice coefficient). ``retained_voice_db_p90`` is a dB ratio and may be negative.
NONNEG_HEADS = frozenset({
    "event_hole_db_p90", "event_hole_db_max",
    "retained_voice_coef_p90", "artifact_ratio_p90", "alpha_error_p90",
})


def project_coherent(pred: dict) -> dict:
    """Project a {head: value} prediction (scalars or arrays) onto the physically feasible
    set, so six independently-fit regressors emit a coherent measurement (oracle P0):
    nonnegative heads clipped at 0; ``event_hole_db_max`` never below ``event_hole_db_p90``."""
    out = dict(pred)
    for h in NONNEG_HEADS:
        if h in out:
            out[h] = np.clip(out[h], 0.0, None)
    if "event_hole_db_max" in out and "event_hole_db_p90" in out:
        out["event_hole_db_max"] = np.maximum(out["event_hole_db_max"], out["event_hole_db_p90"])
    return out


def predict_coherent(models: dict, X: np.ndarray) -> dict:
    """Inference path: predict every fitted head and project onto the feasible set.
    The selection harness MUST use this, never raw per-head ``.predict`` (which can emit
    negative holes / ``hole_max < hole_p90``)."""
    return project_coherent({h: m.predict(X) for h, m in models.items()})


def _q(vals: list[float], pct: float) -> float:
    a = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    return float(np.percentile(a, pct)) if a.size else 0.0


def label_targets(labels) -> dict:
    """Aggregate available tile labels into the regression targets + a coverage note.
    Ill-conditioned/unavailable tiles are excluded (never treated as clean zeros)."""
    avail = [lab for lab in labels if getattr(lab, "available", False)]
    by_field: dict[str, list] = {}
    for lab in avail:
        d = lab.to_dict()
        for field in {f for f, _ in TARGETS.values()}:
            by_field.setdefault(field, []).append(d.get(field))
    out = {name: agg(by_field.get(field, [])) for name, (field, agg) in TARGETS.items()}
    out["_available_tiles"] = len(avail)
    out["_total_tiles"] = len(list(labels))
    return out


def build_example(mixture, accompaniment, vocal, candidate, sr, *, task="soloist_vs_rest",
                  tile_seconds: float = 0.5) -> tuple[np.ndarray, dict]:
    """One (feature vector, label targets) example from a known-truth case.

    ``mixture`` M, ``accompaniment`` A, ``vocal`` V are the exact signals; ``candidate``
    Y is a recipe's accompaniment estimate on M. Shapes are aligned to the shortest."""
    from . import dsp
    M, A, V, Y = (dsp.as2d(x) for x in (mixture, accompaniment, vocal, candidate))
    n = min(len(M), len(A), len(V), len(Y))
    M, A, V, Y = M[:n], A[:n], V[:n], Y[:n]
    feats, avail = jf2.source_aware_features(M, Y, sr, removed=M - Y, task=task)
    x = jf2.feature_vector(feats, avail)   # task one-hot + availability bits now IN the vector
    tile = max(256, int(tile_seconds * sr))
    labels = jl.local_source_coordinate_labels(Y, A, V, tile_frames=tile, hop_frames=tile // 2)
    return x, label_targets(labels)


def train_baseline(X: np.ndarray, targets: dict[str, np.ndarray], groups: np.ndarray) -> dict:
    """Per-head HistGradientBoosting with LeaveOneGroupOut over works. Predictions are
    **projected coherently** each fold (nonnegative heads, ``hole_max ≥ hole_p90``) BEFORE
    scoring, so the reported LOWO error is the error of the deployable output. Returns fitted
    models + per-head LOWO MAE + a calibrated p90 absolute-error interval (the honest
    generalization numbers)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import LeaveOneGroupOut

    def _mk():
        return HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                             learning_rate=0.08, l2_regularization=1.0)

    heads = list(targets)
    Y = {h: np.asarray(targets[h], dtype=np.float64) for h in heads}
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    logo = LeaveOneGroupOut()

    # Accumulate PROJECTED per-row absolute errors across LOWO folds (all heads jointly,
    # so the hole_max ≥ hole_p90 projection sees both heads' predictions on the same rows).
    abs_err: dict[str, list] = {h: [] for h in heads}
    base_err: dict[str, list] = {h: [] for h in heads}
    if n_groups >= 2:
        for tr, te in logo.split(X, groups=groups):
            raw: dict = {}
            for h in heads:
                if len(np.unique(Y[h][tr])) < 2:
                    continue
                m = _mk(); m.fit(X[tr], Y[h][tr]); raw[h] = m.predict(X[te])
            proj = project_coherent(raw)
            for h in heads:
                if h in proj:
                    abs_err[h].append(np.abs(proj[h] - Y[h][te]))
                    base_err[h].append(np.abs(np.median(Y[h][tr]) - Y[h][te]))

    report: dict = {"n_examples": int(len(X)), "n_groups": int(n_groups), "heads": {}}
    for h in heads:
        ae = np.concatenate(abs_err[h]) if abs_err[h] else None
        be = np.concatenate(base_err[h]) if base_err[h] else None
        lowo = float(np.mean(ae)) if ae is not None else None
        base = float(np.mean(be)) if be is not None else None
        report["heads"][h] = {
            "lowo_mae": lowo, "median_baseline_mae": base,
            "beats_baseline": (lowo is not None and base is not None and lowo < base),
            "calibrated_p90_abs_error": (float(np.percentile(ae, 90)) if ae is not None else None),
            "target_mean": float(np.mean(Y[h])), "target_std": float(np.std(Y[h])),
            "nonnegative": h in NONNEG_HEADS}

    # Full-data models per head (inference MUST go through predict_coherent).
    models: dict = {}
    for h in heads:
        if len(np.unique(Y[h])) >= 2:
            m = _mk(); m.fit(X, Y[h]); models[h] = m
    return {"report": report, "models": models}


def assemble_and_train(examples: list[tuple[np.ndarray, dict, str]]) -> dict:
    """``examples``: list of (feature_vec, targets, work_group). Drops examples whose
    labels had no available tiles."""
    X, groups = [], []
    tcols: dict[str, list] = {k: [] for k in TARGETS}
    for x, tgt, grp in examples:
        if tgt.get("_available_tiles", 0) == 0:
            continue
        X.append(x); groups.append(grp)
        for k in TARGETS:
            tcols[k].append(tgt[k])
    if not X:
        return {"report": {"n_examples": 0, "reason": "no examples with available label tiles"}}
    X = np.vstack(X)
    return train_baseline(X, {k: np.asarray(v) for k, v in tcols.items()}, np.asarray(groups))
