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
    "vocal_audibility": 0.5,    # §4: retained voice must be inaudible vs orchestra
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
# ===========================================================================
# v3 (Phases K/L): Learn-Then-Test calibration + calibrated constrained
# feasibility. CVaR (above) is demoted to a stress summary; feasibility gates +
# robustness + regret own the terminal. No LLM can override this state machine.
# ===========================================================================
SELECTOR_V3_VERSION = "autonomous-selector/v3"
CRITICAL_DEFECTS = ("orchestral_theft", "vocal_leakage", "event_hole", "severe_artifact")


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(X <= k) for X ~ Binomial(n, p), via log-sum-exp over exact terms."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 0.0 if k >= n else -math.inf
    logs = []
    lp, lq = math.log(p), math.log(1.0 - p)
    for i in range(0, k + 1):
        logs.append(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * lp + (n - i) * lq)
    m = max(logs)
    return m + math.log(sum(math.exp(x - m) for x in logs))


def _binomial_ucb(k: int, n: int, delta: float = 0.05) -> float:
    """Exact one-sided Clopper–Pearson upper bound: the smallest p with
    P(X <= k; n, p) <= delta. Small n ⇒ wide bound ⇒ strict thresholds (by design);
    at k=0 this is 1 - delta^(1/n) — the standard Learn-Then-Test choice."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    log_delta = math.log(delta)
    for _ in range(60):  # binary search to ~1e-18 precision
        mid = 0.5 * (lo + hi)
        if _log_binom_cdf(k, n, mid) > log_delta:
            lo = mid
        else:
            hi = mid
    return min(1.0, hi)


def learn_then_test(calibration: list[tuple[str, float, bool]], *, target_risk: float = 0.10,
                    delta: float = 0.05) -> dict:
    """Choose a certification threshold τ for ONE defect from grouped calibration
    data (§15.3): ``(group_id, predicted_severity, truly_bad)`` triples, where
    truth comes from exact-target labels. Group-atomic: each group contributes its
    worst case once. Scans τ from strict→loose (fixed-sequence testing) and keeps
    the loosest τ whose *upper-bounded* certified-risk stays ≤ target_risk.
    Data choose the threshold — never an attractive constant (§15.5)."""
    per_group: dict[str, tuple[float, bool]] = {}
    for gid, sev, bad in calibration:
        prev = per_group.get(gid)
        if prev is None or sev > prev[0] or (bad and not prev[1]):
            per_group[gid] = (max(sev, prev[0]) if prev else sev, bad or (prev[1] if prev else False))
    units = sorted(per_group.values())
    if not units:
        return {"tau": 0.0, "risk_ucb": 1.0, "n_groups": 0, "certifiable": False}
    grid = sorted({s for s, _ in units})   # exact severities: tau lands ON a unit,
    # so the first bad case fails its own grid point rather than being excluded
    # by a rounding epsilon (tau settles on the largest certifiably-good unit)
    best = None
    for tau in grid:  # strict -> loose
        certified = [(s, b) for s, b in units if s <= tau]
        n = len(certified)
        k = sum(1 for _, b in certified if b)
        ucb = _binomial_ucb(k, n, delta)
        if n > 0 and ucb <= target_risk:
            best = {"tau": float(tau), "risk_ucb": round(ucb, 4), "n_groups": n,
                    "certifiable": True}
        elif best is not None:
            break  # fixed-sequence: stop at the first failure after a success
    return best or {"tau": 0.0, "risk_ucb": 1.0, "n_groups": len(units), "certifiable": False}


def calibrate_taus(calibration_by_defect: dict[str, list[tuple[str, float, bool]]],
                   *, target_risk: float = 0.10, delta: float = 0.05) -> dict[str, dict]:
    return {d: learn_then_test(rows, target_risk=target_risk, delta=delta)
            for d, rows in calibration_by_defect.items()}


def _defect_ucb(cells: dict[str, list[tuple[float, float]]], defect: str, *,
                n_boot: int = 200, seed: int = 0) -> float:
    pairs = cells.get(defect)
    if not pairs:
        return 1.0  # missing evidence on a critical defect is NOT feasibility
    vals = [s + u for s, u in pairs]
    rng = np.random.default_rng(seed)
    boots = [max(float(np.max(rng.choice(vals, size=len(vals), replace=True))), 0.0)
             for _ in range(n_boot)]
    return float(np.percentile(boots, 95))


def select_v3(candidates: dict[str, dict], taus: dict[str, dict], *,
              epsilon_regret: float = 0.10, probe_budget_left: bool = False,
              distribution_flag: str = "in_calibration_domain",
              scope_limited: frozenset[str] = frozenset(),
              n_boot: int = 200, seed: int = 0) -> dict:
    """§16 terminal logic. ``candidates``: id -> {"cells": {defect: [(s,u)...]},
    "secondary": float, "evidence": {family: cells-dict}, "gates": {...}}.
    Outcomes: final | needs_probe (budget permitting) | no_acceptable_candidate.
    Certification: autonomous_proxy_certified (+ the distribution flag)."""
    report: dict[str, dict] = {}
    feasible: list[str] = []
    for cid, c in candidates.items():
        ok_hard, failed_hard = hard_gates(c.get("gates", {}))
        # §6: iterate EVERY critical defect unconditionally. A missing observation
        # or a non-certifiable threshold is infeasible — UNLESS the axis is
        # explicitly scope_limited (then it's a declared limitation, not a pass,
        # and it caps the achievable certification level).
        ucbs, infeasible, scoped_out = {}, [], []
        for d in CRITICAL_DEFECTS:
            if d in scope_limited:
                scoped_out.append(d)
                continue
            u = round(_defect_ucb(c["cells"], d, n_boot=n_boot, seed=seed), 4)
            ucbs[d] = u
            if not taus.get(d, {}).get("certifiable", False) or u > taus[d]["tau"]:
                infeasible.append(d)
        report[cid] = {"hard_failed": failed_hard, "ucbs": ucbs,
                       "infeasible_defects": infeasible, "scope_limited": scoped_out,
                       "secondary": round(float(c.get("secondary", 0.0)), 4)}
        if "secondary_lower" in c:
            report[cid]["secondary_lower"] = round(float(c["secondary_lower"]), 4)
        if "secondary_upper" in c:
            report[cid]["secondary_upper"] = round(float(c["secondary_upper"]), 4)
        if ok_hard and not infeasible:
            feasible.append(cid)

    base = {"selector_version": SELECTOR_V3_VERSION,
            "certification": "autonomous_proxy_certified",
            "distribution": distribution_flag,
            "scope_limited": sorted(scope_limited),
            "taus": taus, "candidates": report}
    if distribution_flag == "out_of_domain":
        return {**base, "status": "no_acceptable_candidate", "best_available": None,
                "reason": "out of calibration domain: bounds are not valid here (§15.4)"}
    if not feasible:
        if probe_budget_left:
            return {**base, "status": "needs_probe",
                    "reason": "no candidate currently feasible; a probe may separate"}
        best = min(report, key=lambda c: (len(report[c]["infeasible_defects"]),
                                          report[c]["secondary"])) if report else None
        return {**base, "status": "no_acceptable_candidate", "best_available": best,
                "failed_gates": report[best]["infeasible_defects"] if best else [],
                "reason": "no candidate passes the calibrated risk gates"}

    ranked = sorted(feasible, key=lambda c: report[c]["secondary"])
    star = ranked[0]

    # robustness (§16.2): still feasible with any one evidence family removed
    loeo_ok = True
    families = candidates[star].get("evidence") or {}
    for fam in families:
        reduced: dict[str, list] = {}
        for other, cells in families.items():
            if other == fam:
                continue
            for d, pairs in cells.items():
                reduced.setdefault(d, []).extend(pairs)
        if not reduced:
            continue
        u2 = {d: _defect_ucb(reduced, d, n_boot=n_boot, seed=seed)
              for d in CRITICAL_DEFECTS if d in reduced or d in taus}
        if any(not taus.get(d, {}).get("certifiable", False) or u > taus[d]["tau"]
               for d, u in u2.items()):
            loeo_ok = False
            break

    # separation (§16.4, oracle P0 §3.1): the winner must be provably SEPARATED
    # from the runner-up, not merely have a lower point estimate. The old check
    # (star.secondary − runner.secondary ≤ ε) was always true because star already
    # sorts lowest — it never tested anything. Correct margin test (lower=better):
    #   runner.secondary_lower − winner.secondary_upper ≥ margin
    # When secondary bounds are unavailable, fall back to a point-gap margin and
    # flag the decision as unseparated so it cannot certify on a coincidental tie.
    regret_ok = True
    separation = None
    if len(ranked) > 1:
        runner = ranked[1]
        s_star = report[star]["secondary"]
        s_run = report[runner]["secondary"]
        star_hi = report[star].get("secondary_upper", s_star)
        run_lo = report[runner].get("secondary_lower", s_run)
        separation = round(run_lo - star_hi, 4)
        regret_ok = separation >= epsilon_regret
        base["separation"] = {"winner": star, "runner_up": runner,
                              "margin": separation, "required": epsilon_regret}

    if loeo_ok and regret_ok:
        # §8: a scope-limited critical axis caps the achievable level — a full
        # risk-certified production claim requires ALL critical defects certified.
        level = ("risk_certified_production" if not scope_limited
                 else "exact_benchmark_qualified")
        return {**base, "status": "final", "candidate_id": star,
                "mode": "feasible_certified",
                "certification_level": level,
                "reason": "passes hard + calibrated gates; leave-one-evidence-out "
                          "stable; separated from runner-up by >= margin"
                          + (f"; scope-limited on {sorted(scope_limited)}" if scope_limited else "")}
    if probe_budget_left:
        return {**base, "status": "needs_probe", "candidate_id": star,
                "reason": ("evidence-removal instability" if not loeo_ok
                           else "regret bound not met")}
    return {**base, "status": "no_acceptable_candidate", "best_available": star,
            "failed_gates": [],
            "reason": ("not robust to evidence removal and probe budget exhausted"
                       if not loeo_ok else
                       "regret bound unmet and probe budget exhausted")}


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
