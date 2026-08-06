"""Robust autonomous selector (v2.1 WP8 / §13) — the deterministic terminal authority.

DeepSeek/Pi may choose the next experiment; **this module owns the final decision**.
Pipeline: severity calibration (isotonic, [0,1]) → hard gates → per-defect
CVaR-with-uncertainty risk → bootstrap bounds → clear-winner / best-safe rules.
A candidate failing a hard gate cannot win on a good aesthetics score, and the
selector never invents confidence: outcomes are ``final`` (mode ``clear_winner`` or
``best_safe``) or ``no_acceptable_candidate`` with ``best_available``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SELECTOR_VERSION = "autonomous-selector/v1"

# Lexicographic defect priorities (§13.4): theft/holes first, leakage ceiling,
# then preservation axes, generic quality last.
DEFAULT_PRIORITIES: tuple[tuple[str, ...], ...] = (
    ("orchestral_theft", "event_hole"),
    ("vocal_leakage",),
    ("fullness", "brightness", "hall", "transients", "stereo"),
    ("generic_quality",),
)

DEFAULT_GATE_LIMITS = {
    "technical": 0.0,           # boolean-ish: any technical failure blocks
    "orchestral_theft": 0.6,
    "vocal_leakage": 0.7,
    "event_hole": 0.7,
    "severe_artifact": 0.8,
}


# ---------------------------------------------------------------------------
# severity calibration (§13.2): isotonic map raw metric -> [0, 1]
# ---------------------------------------------------------------------------
def pav_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: monotone non-decreasing fit of y over sorted x.
    Returns (sorted_x, fitted_y)."""
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    fitted = ys.copy()
    w = np.ones_like(fitted)
    i = 0
    while i < len(fitted) - 1:
        if fitted[i] > fitted[i + 1] + 1e-15:
            merged = (fitted[i] * w[i] + fitted[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            fitted[i] = fitted[i + 1] = merged
            w[i] = w[i + 1] = w[i] + w[i + 1]
            j = i
            while j > 0 and fitted[j - 1] > fitted[j] + 1e-15:
                merged = (fitted[j - 1] * w[j - 1] + fitted[j] * w[j]) / (w[j - 1] + w[j])
                fitted[j - 1] = fitted[j] = merged
                w[j - 1] = w[j] = w[j - 1] + w[j]
                j -= 1
            i = max(j, 0)
        else:
            i += 1
    return xs, fitted


@dataclass
class SeverityMap:
    """Monotonic raw-metric → calibrated-severity map, fit from injected-defect
    ladders / exact-target errors (x = raw values, y = known severities in [0,1])."""
    xs: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    ys: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    @classmethod
    def fit(cls, raw: list[float], severity: list[float]) -> "SeverityMap":
        if len(raw) < 2:
            raise ValueError("severity calibration needs >=2 anchor points")
        xs, ys = pav_isotonic(np.asarray(raw), np.clip(np.asarray(severity, float), 0, 1))
        return cls(xs=xs, ys=ys)

    def apply(self, value: float) -> float:
        return float(np.clip(np.interp(value, self.xs, self.ys), 0.0, 1.0))


# ---------------------------------------------------------------------------
# risk (§13.5): per-defect CVaR over passages/challenges, worst defect wins
# ---------------------------------------------------------------------------
def cvar(values: list[float], alpha: float = 0.90) -> float:
    """Mean of the worst (1-alpha) tail; at small n this approaches minimax (the
    tail is at least one element), which is the intended behavior."""
    if not values:
        return 0.0
    v = np.sort(np.asarray(values, float))[::-1]
    k = max(1, math.ceil((1.0 - alpha) * len(v)))
    return float(v[:k].mean())


def candidate_risk(cells: dict[str, list[tuple[float, float]]], *, lam: float = 0.5,
                   alpha: float = 0.90) -> tuple[float, dict[str, float]]:
    """``cells``: defect -> [(severity, uncertainty), ...] over passages/challenges.
    Risk = max over defects of CVaR(s + λu). Returns (risk, per_defect)."""
    per: dict[str, float] = {}
    for defect, pairs in cells.items():
        per[defect] = cvar([s + lam * u for s, u in pairs], alpha)
    return (max(per.values()) if per else 1.0), per


def bootstrap_risk(cells: dict[str, list[tuple[float, float]]], *, lam: float = 0.5,
                   alpha: float = 0.90, n_boot: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    risks = []
    for _ in range(n_boot):
        resampled = {}
        for defect, pairs in cells.items():
            if not pairs:
                continue
            idx = rng.integers(0, len(pairs), size=len(pairs))
            resampled[defect] = [pairs[i] for i in idx]
        r, _ = candidate_risk(resampled, lam=lam, alpha=alpha)
        risks.append(r)
    r_mean, _ = candidate_risk(cells, lam=lam, alpha=alpha)
    if not risks:
        return {"risk_lower": 1.0, "risk_mean": 1.0, "risk_upper": 1.0}
    return {"risk_lower": float(np.percentile(risks, 5)),
            "risk_mean": float(r_mean),
            "risk_upper": float(np.percentile(risks, 95))}


# ---------------------------------------------------------------------------
# hard gates + lexicographic tie-break (§13.3–13.4)
# ---------------------------------------------------------------------------
def hard_gates(gate_severities: dict[str, float],
               limits: dict[str, float] = DEFAULT_GATE_LIMITS) -> tuple[bool, list[str]]:
    failed = [g for g, lim in limits.items()
              if gate_severities.get(g) is not None and gate_severities[g] > lim]
    return (not failed), sorted(failed)


def lexicographic_key(per_defect: dict[str, float],
                      priorities=DEFAULT_PRIORITIES) -> tuple[float, ...]:
    """Lower tuple = better; each priority tier contributes its worst member."""
    return tuple(max((per_defect.get(d, 0.0) for d in tier), default=0.0)
                 for tier in priorities)


# ---------------------------------------------------------------------------
# the terminal decision (§13.7–13.8)
# ---------------------------------------------------------------------------
def select(candidates: dict[str, dict], *, lam: float = 0.5, alpha: float = 0.90,
           delta: float = 0.05, max_acceptable_risk: float = 0.65,
           gate_limits: dict[str, float] = DEFAULT_GATE_LIMITS,
           priorities=DEFAULT_PRIORITIES, n_boot: int = 300, seed: int = 0) -> dict:
    """``candidates``: id -> {"cells": {defect: [(s, u), ...]}, "gates": {gate: sev}}.
    Returns the terminal decision dict; the selector is the only component allowed
    to emit ``final``."""
    report: dict[str, dict] = {}
    eligible: list[str] = []
    for cid, c in candidates.items():
        ok, failed = hard_gates(c.get("gates", {}), gate_limits)
        _, per_defect = candidate_risk(c["cells"], lam=lam, alpha=alpha)
        bounds = bootstrap_risk(c["cells"], lam=lam, alpha=alpha, n_boot=n_boot, seed=seed)
        report[cid] = {"gates_passed": ok, "failed_gates": failed,
                       "per_defect": {k: round(v, 4) for k, v in per_defect.items()},
                       **{k: round(v, 4) for k, v in bounds.items()}}
        if ok:
            eligible.append(cid)

    base = {"selector_version": SELECTOR_VERSION, "candidates": report,
            "params": {"lambda": lam, "alpha": alpha, "delta": delta,
                       "max_acceptable_risk": max_acceptable_risk}}

    if not eligible:
        best = min(report, key=lambda c: report[c]["risk_mean"]) if report else None
        best_failed = report[best]["failed_gates"] if best is not None else []
        return {**base, "status": "no_acceptable_candidate", "best_available": best,
                "failed_gates": best_failed,
                "reason": "no candidate passes the hard gates"}

    # clear winner (§13.7): U(c*) + delta < min L(others)
    by_mean = sorted(eligible, key=lambda c: report[c]["risk_mean"])
    star = by_mean[0]
    others = [c for c in eligible if c != star]
    if not others:
        if report[star]["risk_upper"] <= max_acceptable_risk:
            return {**base, "status": "final", "mode": "best_safe", "candidate_id": star,
                    "risk_upper": report[star]["risk_upper"],
                    "reason": "single eligible candidate under the acceptable-risk ceiling"}
        return {**base, "status": "no_acceptable_candidate", "best_available": star,
                "failed_gates": [], "reason": "single candidate exceeds acceptable risk"}
    if report[star]["risk_upper"] + delta < min(report[c]["risk_lower"] for c in others):
        return {**base, "status": "final", "mode": "clear_winner", "candidate_id": star,
                "risk_upper": report[star]["risk_upper"],
                "reason": "upper bound + margin below every competitor's lower bound"}

    # best-safe (§13.8): bounds overlap after the probe budget
    def _lex(cid: str):
        _, per = candidate_risk(candidates[cid]["cells"], lam=lam, alpha=alpha)
        return lexicographic_key(per, priorities)

    safe = [c for c in eligible if report[c]["risk_upper"] <= max_acceptable_risk]
    if safe:
        winner = min(safe, key=_lex)
        return {**base, "status": "final", "mode": "best_safe", "candidate_id": winner,
                "risk_upper": report[winner]["risk_upper"],
                "reason": "bounds overlap; lexicographic best among gate-passing, "
                          "risk-acceptable candidates (not 'certain')"}

    best = min(eligible, key=lambda c: report[c]["risk_mean"])
    return {**base, "status": "no_acceptable_candidate", "best_available": best,
            "failed_gates": [],
            "reason": "all candidates exceed the acceptable-risk ceiling"}
