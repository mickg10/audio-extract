"""Panel scoring, Pareto frontier, and successive halving (docs/v2 §3.5, §4.5).

Deterministic orchestration over rendered candidates. For real audio there is no
ground-truth accompaniment, so the *reference* for reference-dependent metrics is
a robust candidate consensus (median across the architecture-diverse set); the
vocal estimate supplies the leakage reference. Ranking is multi-axis: a candidate
survives if nothing dominates it on every damage axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp, metrics

# Damage axes — lower is better on every one.
AXES = ["leakage", "holes_db", "pump_depth_db", "brightness_deviation", "hall_damage_db", "stereo_deviation"]


@dataclass
class Scored:
    recipe_id: str
    costs: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


def consensus_reference(cands: list[np.ndarray]) -> np.ndarray:
    """Per-sample robust median across the candidate set (docs/v2 §4.3)."""
    n = min(len(c) for c in cands)
    stack = np.stack([dsp.as2d(c)[:n] for c in cands], axis=0)
    return np.median(stack, axis=0)


def _windows_or_whole(arr: np.ndarray, windows):
    if not windows:
        yield arr
        return
    for (s, e) in windows:
        s, e = int(s), int(e)
        if e - s > 1:
            yield arr[s:e]


def score_candidate(cand: np.ndarray, sr: int, *, reference: np.ndarray,
                    vocal_ref: np.ndarray | None = None, activity: np.ndarray | None = None,
                    windows=None) -> dict:
    """Aggregate the metric bank over passage windows into a damage cost vector.

    Reference-dependent axes are OMITTED when their evidence is absent — missing
    evidence is unknown, never scored as 0 (best). Stereo width is compared
    per-passage, not against a single whole-file width.
    """
    axes = ["holes_db", "pump_depth_db", "brightness_deviation", "stereo_deviation"]
    if vocal_ref is not None:
        axes.append("leakage")
    if activity is not None:
        axes.append("hall_damage_db")
    acc: dict[str, list[float]] = {a: [] for a in axes}
    ref_windows = list(_windows_or_whole(reference, windows))
    for w_i, seg in enumerate(_windows_or_whole(cand, windows)):
        ref_seg = ref_windows[w_i]
        acc["holes_db"].append(metrics.fullness(seg, ref_seg, sr)["holes_db"])
        acc["pump_depth_db"].append(metrics.pumping(seg, sr)["pump_depth_db"])
        acc["brightness_deviation"].append(metrics.brightness_deviation(seg, ref_seg, sr)["brightness_deviation"])
        acc["stereo_deviation"].append(
            abs(metrics.stereo(seg)["stereo_width"] - metrics.stereo(ref_seg)["stereo_width"]))
        if vocal_ref is not None:
            vseg = list(_windows_or_whole(dsp.as2d(vocal_ref), windows))[w_i]
            acc["leakage"].append(metrics.leakage(seg, vseg, sr)["leakage"])
        if activity is not None:
            aseg = list(_windows_or_whole(np.asarray(activity).reshape(-1, 1), windows))[w_i].ravel()
            acc["hall_damage_db"].append(metrics.hall(seg, ref_seg, sr, aseg)["hall_damage_db"])
    return {a: float(np.mean(v)) for a, v in acc.items() if v}


def _axes_for(scored: list["Scored"]) -> list[str]:
    """Axes present in EVERY candidate's cost vector, in canonical order — so a run
    with omitted (unavailable) axes ranks on the axes it actually measured."""
    if not scored:
        return list(AXES)
    common = set(scored[0].costs)
    for s in scored[1:]:
        common &= set(s.costs)
    ordered = [a for a in AXES if a in common]
    return ordered or sorted(common)


def dominates(a: dict, b: dict, axes=AXES, eps: float = 1e-9) -> bool:
    """True if ``a`` is no worse on every axis and strictly better on at least one."""
    no_worse = all(a[x] <= b[x] + eps for x in axes)
    strictly = any(a[x] < b[x] - eps for x in axes)
    return no_worse and strictly


def pareto_frontier(scored: list[Scored], axes=None) -> list[Scored]:
    axes = axes or _axes_for(scored)
    front = []
    for s in scored:
        if not any(dominates(o.costs, s.costs, axes) for o in scored if o is not s):
            front.append(s)
    return front


def rank_by_scalarized(scored: list[Scored], weights: dict | None = None, axes=None) -> list[Scored]:
    """Tie-break the frontier with a z-scored weighted sum (lower total = better)."""
    if not scored:
        return []
    axes = axes or _axes_for(scored)
    w = weights or {a: 1.0 for a in axes}
    mat = np.array([[s.costs[a] for a in axes] for s in scored], dtype=np.float64)
    mu, sd = mat.mean(0), mat.std(0) + 1e-9
    z = (mat - mu) / sd
    totals = z @ np.array([w[a] for a in axes])
    order = np.argsort(totals)
    return [scored[i] for i in order]


def successive_halving(scored: list[Scored], keep: int, axes=None) -> list[Scored]:
    """Keep the Pareto frontier; if it exceeds ``keep``, trim by scalarized rank
    (docs/v2 §3.5). If the frontier is smaller than ``keep``, backfill with the
    next-best dominated candidates so a round always advances enough work."""
    axes = axes or _axes_for(scored)
    front = pareto_frontier(scored, axes)
    ranked_front = rank_by_scalarized(front, axes=axes)
    if len(ranked_front) >= keep:
        return ranked_front[:keep]
    remaining = [s for s in scored if s not in front]
    backfill = rank_by_scalarized(remaining, axes=axes)
    return ranked_front + backfill[: keep - len(ranked_front)]
