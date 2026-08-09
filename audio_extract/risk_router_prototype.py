"""Synthetic counterfactual-risk student and deterministic Potts decoder.

This module is deliberately not wired into production.  Exact reference risks
are training/calibration labels only.  Inference accepts features, frozen
calibration offsets, and fixed policy thresholds; the decoder never receives a
clean accompaniment or vocal reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import numpy as np


class RiskRouterError(ValueError):
    """The synthetic risk-router contract is invalid."""


def _finite_array(value, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise RiskRouterError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RiskRouterError(f"{name} must be non-empty and finite")
    return array


@dataclass(frozen=True)
class LinearRiskStudent:
    """Independent linear heads for all ``candidate x defect`` outcomes."""

    coefficients: np.ndarray  # (features + intercept, candidates, defects)
    upper_offsets: np.ndarray  # (candidates, defects)
    quantile: float

    def predict_upper(self, features) -> np.ndarray:
        """Predict calibrated upper risks without accepting exact truth."""
        x = _finite_array(features, "features")
        coefficients = _finite_array(self.coefficients, "coefficients", ndim=3)
        offsets = _finite_array(self.upper_offsets, "upper offsets", ndim=2)
        if coefficients.shape[1:] != offsets.shape:
            raise RiskRouterError("coefficient and calibration head shapes disagree")
        if x.ndim < 2:
            raise RiskRouterError("features must end in one feature dimension")
        feature_count = coefficients.shape[0] - 1
        if x.shape[-1] != feature_count:
            raise RiskRouterError(
                f"feature count {x.shape[-1]} != trained count {feature_count}"
            )
        design = np.concatenate((x, np.ones((*x.shape[:-1], 1))), axis=-1)
        point = np.einsum("...f,fkd->...kd", design, coefficients)
        return np.maximum(point + offsets, 0.0)


def fit_linear_risk_student(train_features, train_risks, calibration_features,
                            calibration_risks, *, quantile: float = 0.9,
                            ridge: float = 1e-6) -> LinearRiskStudent:
    """Fit all counterfactual heads, then conformally calibrate upper offsets.

    Risks use ``(cells, candidates, defects)``.  One cell therefore contributes
    ``K x D`` supervised outcomes rather than one hard route label.
    """
    x = _finite_array(train_features, "train_features", ndim=2)
    y = _finite_array(train_risks, "train_risks", ndim=3)
    cx = _finite_array(calibration_features, "calibration_features", ndim=2)
    cy = _finite_array(calibration_risks, "calibration_risks", ndim=3)
    if x.shape[0] != y.shape[0] or cx.shape[0] != cy.shape[0]:
        raise RiskRouterError("feature/risk cell counts disagree")
    if x.shape[1] != cx.shape[1] or y.shape[1:] != cy.shape[1:]:
        raise RiskRouterError("training/calibration head shapes disagree")
    if not 0.5 < quantile < 1.0:
        raise RiskRouterError("upper quantile must be between 0.5 and 1")
    if not math.isfinite(ridge) or ridge <= 0:
        raise RiskRouterError("ridge must be finite and positive")

    design = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ y.reshape(len(y), -1)).reshape(
        design.shape[1], *y.shape[1:]
    )
    cal_design = np.concatenate((cx, np.ones((len(cx), 1))), axis=1)
    point = np.einsum("nf,fkd->nkd", cal_design, coefficients)
    residual = cy - point
    # Split-conformal upper order statistic, finite-sample corrected.
    rank = min(len(cx) - 1, max(0, math.ceil((len(cx) + 1) * quantile) - 1))
    offsets = np.sort(residual, axis=0)[rank]
    return LinearRiskStudent(coefficients, offsets, quantile)


def confident_cell_mask(exact_counterfactual_risks, critical_thresholds,
                        secondary_weights, *, minimum_margin: float) -> np.ndarray:
    """Offline mask for near-tied exact labels.

    This consumes exact counterfactual labels only while constructing training
    data.  A cell is confident when its two best feasible candidates differ by
    at least ``minimum_margin`` in normalized secondary cost.  One feasible
    candidate is unambiguous; zero feasible candidates is masked.
    """
    risks = _finite_array(exact_counterfactual_risks, "exact risks", ndim=4)
    thresholds = _finite_array(critical_thresholds, "critical thresholds", ndim=1)
    weights = _finite_array(secondary_weights, "secondary weights", ndim=1)
    if minimum_margin < 0 or not math.isfinite(minimum_margin):
        raise RiskRouterError("minimum_margin must be finite and non-negative")
    if len(thresholds) + len(weights) != risks.shape[-1]:
        raise RiskRouterError("critical + secondary dimensions do not match risks")
    if np.any(thresholds <= 0) or np.any(weights < 0):
        raise RiskRouterError("thresholds must be positive and weights non-negative")
    feasible = np.all(risks[..., :len(thresholds)] <= thresholds, axis=-1)
    secondary = risks[..., len(thresholds):] @ weights
    result = np.zeros(risks.shape[:2], dtype=bool)
    for t in range(risks.shape[0]):
        for band in range(risks.shape[1]):
            costs = np.sort(secondary[t, band, feasible[t, band]])
            if len(costs) == 1:
                result[t, band] = True
            elif len(costs) >= 2:
                result[t, band] = costs[1] - costs[0] >= minimum_margin
    return result


@dataclass(frozen=True)
class PottsRouteResult:
    decision: str
    route: np.ndarray | None
    fallback_route: np.ndarray
    used_fallback: bool
    objective: float | None
    plan_id: str
    reason: str | None


def _plan_id(upper: np.ndarray, policy: dict, outcome: dict) -> str:
    h = hashlib.sha256()
    h.update(b"audio-extract/risk-router-plan/v0\0")
    h.update(np.ascontiguousarray(upper, dtype="<f8").tobytes())
    h.update(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode())
    h.update(json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode())
    return "sha256:" + h.hexdigest()


def decode_potts(predicted_upper_risks, critical_thresholds, secondary_weights,
                 *, time_switch_cost: float, frequency_switch_cost: float,
                 fallback_candidate: int = 0, confidence_mask=None,
                 maximum_cells: int = 12) -> PottsRouteResult:
    """Solve a small deterministic exact Potts route by exhaustive enumeration.

    Feasibility under calibrated upper voice/hole/artifact bounds is hard and
    lexicographically precedes secondary distortion.  This intentionally small
    solver is an executable specification, not the production MILP.
    """
    upper = _finite_array(predicted_upper_risks, "predicted upper risks", ndim=4)
    thresholds = _finite_array(critical_thresholds, "critical thresholds", ndim=1)
    weights = _finite_array(secondary_weights, "secondary weights", ndim=1)
    time_count, band_count, candidate_count, defect_count = upper.shape
    if len(thresholds) + len(weights) != defect_count:
        raise RiskRouterError("critical + secondary dimensions do not match risks")
    if np.any(thresholds <= 0) or np.any(weights < 0):
        raise RiskRouterError("thresholds must be positive and weights non-negative")
    if any(not math.isfinite(x) or x < 0 for x in (time_switch_cost, frequency_switch_cost)):
        raise RiskRouterError("switching costs must be finite and non-negative")
    if not 0 <= fallback_candidate < candidate_count:
        raise RiskRouterError("fallback candidate is outside the candidate bank")
    if time_count * band_count > maximum_cells:
        raise RiskRouterError("prototype refuses a grid larger than maximum_cells")

    fallback = np.full((time_count, band_count), fallback_candidate, dtype=np.int64)
    policy = {
        "critical_thresholds": thresholds.tolist(),
        "secondary_weights": weights.tolist(),
        "time_switch_cost": time_switch_cost,
        "frequency_switch_cost": frequency_switch_cost,
        "fallback_candidate": fallback_candidate,
    }
    if confidence_mask is not None:
        confidence = np.asarray(confidence_mask, dtype=bool)
        if confidence.shape != (time_count, band_count):
            raise RiskRouterError("confidence mask shape disagrees with risk grid")
        policy["confidence_mask"] = confidence.tolist()
        if not np.all(confidence):
            plan = fallback.tolist()
            outcome = {"decision": "abstain", "reason": "near_tie_or_low_confidence",
                       "route": plan}
            return PottsRouteResult(
                "abstain", None, fallback, True, None,
                _plan_id(upper, policy, outcome), "near_tie_or_low_confidence",
            )

    feasible = np.all(upper[..., :len(thresholds)] <= thresholds, axis=-1)
    if np.any(~np.any(feasible, axis=-1)):
        plan = fallback.tolist()
        outcome = {"decision": "abstain", "reason": "no_feasible_candidate",
                   "route": plan}
        return PottsRouteResult(
            "abstain", None, fallback, True, None,
            _plan_id(upper, policy, outcome), "no_feasible_candidate",
        )
    secondary = upper[..., len(thresholds):] @ weights
    best_cost = math.inf
    best_route = None
    for assignment in itertools.product(range(candidate_count), repeat=time_count * band_count):
        route = np.asarray(assignment, dtype=np.int64).reshape(time_count, band_count)
        if any(not feasible[t, b, route[t, b]]
               for t in range(time_count) for b in range(band_count)):
            continue
        cost = sum(secondary[t, b, route[t, b]]
                   for t in range(time_count) for b in range(band_count))
        cost += time_switch_cost * np.count_nonzero(route[1:] != route[:-1])
        cost += frequency_switch_cost * np.count_nonzero(route[:, 1:] != route[:, :-1])
        cost = float(cost)
        if cost < best_cost:  # product order supplies deterministic tie-breaking
            best_cost, best_route = cost, route.copy()
    if best_route is None:  # guarded by per-cell feasibility, retained defensively
        raise RuntimeError("feasible cells produced no global route")
    plan = best_route.tolist()
    outcome = {"decision": "route", "reason": None, "route": plan}
    return PottsRouteResult(
        "route", best_route, fallback, False, best_cost,
        _plan_id(upper, policy, outcome), None,
    )


def gather_shared_stereo(candidate_cells, route) -> np.ndarray:
    """Apply one candidate index per time/band cell to both stereo channels."""
    candidates = _finite_array(candidate_cells, "candidate cells", ndim=4)
    plan = np.asarray(route, dtype=np.int64)
    if plan.shape != candidates.shape[:2]:
        raise RiskRouterError("route shape disagrees with candidate grid")
    if candidates.shape[-1] != 2:
        raise RiskRouterError("prototype requires exactly two shared-route channels")
    if np.any(plan < 0) or np.any(plan >= candidates.shape[2]):
        raise RiskRouterError("route contains an invalid candidate index")
    return np.take_along_axis(candidates, plan[..., None, None], axis=2)[..., 0, :]
