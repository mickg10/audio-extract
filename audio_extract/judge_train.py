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
    feats, _ = jf2.source_aware_features(M, Y, sr, removed=M - Y, task=task)
    x = jf2.feature_vector(feats)
    tile = max(256, int(tile_seconds * sr))
    labels = jl.local_source_coordinate_labels(Y, A, V, tile_frames=tile, hop_frames=tile // 2)
    return x, label_targets(labels)


def train_baseline(X: np.ndarray, targets: dict[str, np.ndarray], groups: np.ndarray) -> dict:
    """Per-head HistGradientBoosting with LeaveOneGroupOut over works. Returns fitted
    models + per-head LOWO mean-absolute-error (the honest generalization number)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import LeaveOneGroupOut

    logo = LeaveOneGroupOut()
    n_groups = len(np.unique(groups))
    report: dict = {"n_examples": len(X), "n_groups": n_groups, "heads": {}}
    models: dict = {}
    for head, y in targets.items():
        y = np.asarray(y, dtype=np.float64)
        # LOWO error
        errs, baseline_errs = [], []
        if n_groups >= 2:
            for tr, te in logo.split(X, y, groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                m = HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                                  learning_rate=0.08, l2_regularization=1.0)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                errs.append(float(np.mean(np.abs(pred - y[te]))))
                baseline_errs.append(float(np.mean(np.abs(np.median(y[tr]) - y[te]))))
        full = HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                             learning_rate=0.08, l2_regularization=1.0)
        if len(np.unique(y)) >= 2:
            full.fit(X, y)
            models[head] = full
        lowo = float(np.mean(errs)) if errs else None
        base = float(np.mean(baseline_errs)) if baseline_errs else None
        report["heads"][head] = {
            "lowo_mae": lowo, "median_baseline_mae": base,
            "beats_baseline": (lowo is not None and base is not None and lowo < base),
            "target_mean": float(np.mean(y)), "target_std": float(np.std(y))}
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
