"""Evidence dependence + fusion (v3 Phases G/H, §11–12).

The committee is **not independent** (adversarial review §4): SAJ, aesthetics
models, and our metrics share lineages and inputs. So:

* every critical defect has a **primary/secondary source mapping** (§11.2) —
  secondaries refine uncertainty, they never outvote a primary;
* every source carries a **reliability profile**; an uncalibrated or unreliable
  source is *diagnostic only* and excluded from fusion;
* a **dependence profile** detects redundant sources — a redundant pair
  contributes once (the more reliable member), never twice;
* **no majority voting** (§12.3): fusion is defect-specific and
  reliability-weighted, with disagreement inflating uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# §11.2 — primary/secondary evidence sources per critical defect axis.
AXIS_SOURCES: dict[str, dict[str, tuple[str, ...]]] = {
    "orchestral_theft": {
        "primary": ("exact_construction_error",),
        "secondary": ("orchestra_response",),
    },
    "vocal_leakage": {
        "primary": ("exact_target_leakage",),
        "secondary": ("saj_precision", "harmonic_f0"),
    },
    "event_hole": {
        "primary": ("exact_event_deficit",),
        "secondary": ("local_envelope_deficit", "masked_hole_uncertainty"),
    },
    "severe_artifact": {
        "primary": ("perceptual_intrusive",),
        "secondary": ("generic_quality", "transient_noise"),
    },
}


@dataclass
class ReliabilityProfile:
    """§12.1 per-source, per-defect reliability facts (from held-out calibration)."""
    pairwise_accuracy: float = 0.0
    calibration_error: float = 1.0
    repeatability: float = 0.0
    order_sensitivity: float = 1.0
    ood_failure_rate: float = 1.0
    calibrated: bool = False

    def usable(self) -> bool:
        return (self.calibrated
                and self.pairwise_accuracy >= 0.75
                and self.repeatability >= 0.90
                and self.order_sensitivity <= 0.10
                and self.ood_failure_rate <= 0.25)

    def weight(self) -> float:
        if not self.usable():
            return 0.0
        return max(0.05, self.pairwise_accuracy * self.repeatability
                   * (1.0 - self.calibration_error))


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(a) != len(b):
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def dependence_profile(scores_a: list[float], scores_b: list[float]) -> dict:
    """§12.2 pairwise dependence on shared held-out cases."""
    rank = _spearman(scores_a, scores_b)
    resid = 0.0
    if len(scores_a) == len(scores_b) and len(scores_a) >= 3:
        a, b = np.asarray(scores_a), np.asarray(scores_b)
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            resid = float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])
    return {"rank_correlation": round(rank, 4),
            "residual_correlation": round(resid, 4),
            "redundant": rank > 0.90}


def fuse(defect: str,
         evidence: dict[str, list[tuple[float, float]]],
         reliability: dict[str, ReliabilityProfile],
         dependence: dict[tuple[str, str], dict] | None = None) -> dict:
    """Fuse per-source severity cells for ONE defect into selector cells.

    Primaries drive the fused severity (reliability-weighted mean); secondaries
    never overturn it — they only widen uncertainty when they disagree. Redundant
    pairs contribute once. Unusable sources are returned as diagnostic_only.
    """
    mapping = AXIS_SOURCES.get(defect, {"primary": (), "secondary": ()})
    dependence = dependence or {}

    usable: dict[str, list[tuple[float, float]]] = {}
    diagnostic_only: list[str] = []
    for src, cells in evidence.items():
        prof = reliability.get(src)
        if prof is None or not prof.usable() or not cells:
            diagnostic_only.append(src)
        else:
            usable[src] = cells

    # collapse redundant pairs: keep the more reliable member (§12.3)
    dropped_redundant: list[str] = []
    for (a, b), dep in dependence.items():
        if not dep.get("redundant") or a not in usable or b not in usable:
            continue
        loser = a if reliability[a].weight() < reliability[b].weight() else b
        usable.pop(loser, None)
        dropped_redundant.append(loser)

    primaries = [s for s in mapping["primary"] if s in usable]
    secondaries = [s for s in mapping["secondary"] if s in usable]
    if not primaries:
        # no calibrated primary evidence -> the defect is UNMEASURED (selector
        # treats missing critical evidence as infeasible; secondaries alone
        # cannot certify a critical axis)
        return {"cells": [], "primaries": [], "secondaries": secondaries,
                "diagnostic_only": sorted(diagnostic_only),
                "dropped_redundant": dropped_redundant, "measured": False}

    def _summary(src: str) -> tuple[float, float]:
        vals = usable[src]
        s = float(np.mean([v for v, _ in vals]))
        u = float(np.mean([u_ for _, u_ in vals]))
        return s, u

    w = np.array([reliability[s].weight() for s in primaries])
    sv = np.array([_summary(s)[0] for s in primaries])
    uv = np.array([_summary(s)[1] for s in primaries])
    fused_s = float(np.average(sv, weights=w))
    fused_u = float(np.average(uv, weights=w))
    # primary disagreement widens uncertainty
    if len(primaries) > 1:
        fused_u += float(np.std(sv))
    # secondaries: never move severity; disagreement with the primary inflates u
    for s in secondaries:
        ss, su = _summary(s)
        fused_u += 0.5 * abs(ss - fused_s) * (1.0 - reliability[s].weight())
        del su
    n = max(len(usable[p]) for p in primaries)
    return {"cells": [(fused_s, fused_u)] * n,
            "primaries": primaries, "secondaries": secondaries,
            "diagnostic_only": sorted(diagnostic_only),
            "dropped_redundant": dropped_redundant, "measured": True}
