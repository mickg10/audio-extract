"""Identity-complete counterfactual-risk router v2.

This versioned successor keeps the v1 prototype for review history and closes
its remaining safety ambiguities:

* candidate recipe and decoded-PCM identities are bound separately;
* predictor, query encoder, and calibration are verified bundle/certificate
  manifests rather than unqualified weight hashes;
* the conservative whole-track fallback is a separately certified raw parent;
* unavailable risk values cannot enter arithmetic;
* zero-weight secondary metrics are rejected;
* the MILP result is accepted only when optimal, integral, objective-recomputed,
  and unique within the frozen numerical tolerance.

The inference API accepts only predicted upper risks. Exact clean references are
not arguments to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json
import math
import re

import numpy as np

PANEL_SCHEMA = "audio-extract/counterfactual-risk-panel/v2"
ROUTER_SCHEMA = "audio-extract/counterfactual-risk-router/v2"
ROUTE_SCHEMA = "audio-extract/counterfactual-risk-route/v2"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RiskRouterV2Error(RuntimeError):
    """The identity, prediction panel, or structured certificate is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be canonical sha256:<64 lowercase hex>")
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class CandidateIdentity:
    recipe_id: str
    artifact_pcm_sha256: str

    def validate(self) -> None:
        _sha(self.recipe_id, "candidate recipe_id")
        _sha(self.artifact_pcm_sha256, "candidate artifact_pcm_sha256")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "recipe_id": self.recipe_id,
            "artifact_pcm_sha256": self.artifact_pcm_sha256,
        }


@dataclass(frozen=True)
class RiskPanelIdentityV2:
    source_pcm_sha256: str
    candidates: tuple[CandidateIdentity, ...]
    metric_names: tuple[str, ...]
    feature_contract_sha256: str
    risk_model_bundle_sha256: str
    calibration_policy_sha256: str
    calibration_certificate_sha256: str
    conservative_parent: CandidateIdentity
    conservative_parent_certificate_sha256: str
    query_encoder_bundle_sha256: str | None = None
    query_condition_sha256: str | None = None

    def validate(self) -> None:
        _sha(self.source_pcm_sha256, "source_pcm_sha256")
        if len(self.candidates) < 2:
            raise ValueError("at least two candidate identities are required")
        for candidate in self.candidates:
            candidate.validate()
        pairs = tuple(
            (candidate.recipe_id, candidate.artifact_pcm_sha256)
            for candidate in self.candidates
        )
        if len(set(pairs)) != len(pairs):
            raise ValueError("candidate recipe/PCM pairs must be unique and ordered")
        if not self.metric_names or any(
            not isinstance(name, str) or not name for name in self.metric_names
        ):
            raise ValueError("metric names must be non-empty strings")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric names must be unique and ordered")
        for name, value in (
            ("feature_contract_sha256", self.feature_contract_sha256),
            ("risk_model_bundle_sha256", self.risk_model_bundle_sha256),
            ("calibration_policy_sha256", self.calibration_policy_sha256),
            (
                "calibration_certificate_sha256",
                self.calibration_certificate_sha256,
            ),
            (
                "conservative_parent_certificate_sha256",
                self.conservative_parent_certificate_sha256,
            ),
        ):
            _sha(value, name)
        self.conservative_parent.validate()
        if self.conservative_parent not in self.candidates:
            raise ValueError("conservative parent is not in the frozen candidate bank")
        if (self.query_encoder_bundle_sha256 is None) != (
            self.query_condition_sha256 is None
        ):
            raise ValueError(
                "query encoder bundle and query condition must be both present or absent"
            )
        if self.query_encoder_bundle_sha256 is not None:
            _sha(
                self.query_encoder_bundle_sha256,
                "query_encoder_bundle_sha256",
            )
            _sha(self.query_condition_sha256, "query_condition_sha256")

    @property
    def conservative_index(self) -> int:
        self.validate()
        return self.candidates.index(self.conservative_parent)

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PANEL_SCHEMA,
            "source_pcm_sha256": self.source_pcm_sha256,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metric_names": list(self.metric_names),
            "feature_contract_sha256": self.feature_contract_sha256,
            "risk_model_bundle_sha256": self.risk_model_bundle_sha256,
            "calibration_policy_sha256": self.calibration_policy_sha256,
            "calibration_certificate_sha256": (
                self.calibration_certificate_sha256
            ),
            "conservative_parent": self.conservative_parent.to_dict(),
            "conservative_parent_certificate_sha256": (
                self.conservative_parent_certificate_sha256
            ),
            "query_encoder_bundle_sha256": self.query_encoder_bundle_sha256,
            "query_condition_sha256": self.query_condition_sha256,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class RiskRouterConfigV2:
    critical_thresholds: tuple[tuple[str, float], ...]
    secondary_weights: tuple[tuple[str, float], ...] = ()
    critical_slack_weight: float = 1.0
    temporal_switch_penalty: float = 0.05
    frequency_switch_penalty: float = 0.05
    feasibility_tolerance: float = 0.0
    milp_time_limit_seconds: float = 180.0
    milp_relative_gap: float = 0.0
    integrality_tolerance: float = 1e-7
    objective_tolerance: float = 1e-8

    def validate(self, panel: RiskPanelIdentityV2) -> None:
        panel.validate()
        critical = dict(self.critical_thresholds)
        secondary = dict(self.secondary_weights)
        if (
            not critical
            or len(critical) != len(self.critical_thresholds)
            or any(name not in panel.metric_names for name in critical)
        ):
            raise ValueError("critical thresholds must be unique known metrics")
        if (
            len(secondary) != len(self.secondary_weights)
            or any(name not in panel.metric_names for name in secondary)
        ):
            raise ValueError("secondary weights must be unique known metrics")
        if set(critical) & set(secondary):
            raise ValueError("a metric cannot be both critical and secondary")
        for name, value in self.critical_thresholds:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"critical threshold {name} must be positive")
        for name, value in self.secondary_weights:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(
                    f"secondary weight {name} must be finite and strictly positive"
                )
        for name in (
            "critical_slack_weight",
            "temporal_switch_penalty",
            "frequency_switch_penalty",
            "feasibility_tolerance",
            "milp_time_limit_seconds",
            "milp_relative_gap",
            "integrality_tolerance",
            "objective_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.critical_slack_weight <= 0:
            raise ValueError("critical_slack_weight must be positive")
        if self.milp_time_limit_seconds <= 0:
            raise ValueError("milp_time_limit_seconds must be positive")
        if self.milp_relative_gap > 1:
            raise ValueError("milp_relative_gap must not exceed one")
        if self.integrality_tolerance <= 0 or self.objective_tolerance <= 0:
            raise ValueError("solver validation tolerances must be positive")

    def identity_dict(self, panel: RiskPanelIdentityV2) -> dict[str, Any]:
        self.validate(panel)
        return {
            "schema": ROUTER_SCHEMA,
            "panel_identity_sha256": panel.sha256,
            "critical_thresholds": [
                [name, float(value)] for name, value in self.critical_thresholds
            ],
            "secondary_weights": [
                [name, float(value)] for name, value in self.secondary_weights
            ],
            "critical_slack_weight": float(self.critical_slack_weight),
            "temporal_switch_penalty": float(self.temporal_switch_penalty),
            "frequency_switch_penalty": float(self.frequency_switch_penalty),
            "feasibility_tolerance": float(self.feasibility_tolerance),
            "milp_time_limit_seconds": float(self.milp_time_limit_seconds),
            "milp_relative_gap": float(self.milp_relative_gap),
            "integrality_tolerance": float(self.integrality_tolerance),
            "objective_tolerance": float(self.objective_tolerance),
        }

    def sha256(self, panel: RiskPanelIdentityV2) -> str:
        return _mapping_sha(self.identity_dict(panel))


@dataclass(frozen=True)
class RiskPanelV2:
    identity: RiskPanelIdentityV2
    upper: np.ndarray
    available: np.ndarray

    def validate(self) -> tuple[int, int, int, int]:
        self.identity.validate()
        upper = np.asarray(self.upper, dtype=np.float64)
        available = np.asarray(self.available, dtype=bool)
        if upper.ndim != 4 or upper.shape != available.shape:
            raise ValueError(
                "upper/available must share (time,band,candidate,metric)"
            )
        time, bands, candidates, metrics = upper.shape
        if min(time, bands, candidates, metrics) < 1:
            raise ValueError("risk panel dimensions must be positive")
        if candidates != len(self.identity.candidates):
            raise ValueError("risk candidate axis differs from frozen identity")
        if metrics != len(self.identity.metric_names):
            raise ValueError("risk metric axis differs from frozen identity")
        if np.any(~np.isfinite(upper[available])):
            raise ValueError("available upper risks must be finite")
        if np.any(upper[available] < 0):
            raise ValueError("available upper risks must be non-negative")
        return upper.shape

    @property
    def sha256(self) -> str:
        self.validate()
        mask = np.asarray(self.available, dtype=bool)
        upper = np.ascontiguousarray(
            np.where(mask, np.asarray(self.upper, dtype=np.float64), 0.0),
            dtype="<f8",
        )
        available = np.ascontiguousarray(mask, dtype=np.uint8)
        header = {
            "schema": PANEL_SCHEMA,
            "identity_sha256": self.identity.sha256,
            "shape": list(upper.shape),
            "risk_dtype": upper.dtype.str,
            "availability_dtype": available.dtype.str,
        }
        return "sha256:" + hashlib.sha256(
            _canonical(header) + upper.tobytes() + available.tobytes()
        ).hexdigest()


@dataclass(frozen=True)
class RouteDecisionV2:
    status: str
    labels: np.ndarray
    raw_parent_bypass: bool
    objective: float | None
    data_objective: float | None
    temporal_switches: int
    frequency_switches: int
    selection_counts: tuple[int, ...]
    infeasible_cells: tuple[tuple[int, int], ...]
    routing_plan_sha256: str
    conservative_parent: CandidateIdentity
    conservative_parent_certificate_sha256: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_SCHEMA,
            "status": self.status,
            "labels_shape": list(np.asarray(self.labels).shape),
            "raw_parent_bypass": self.raw_parent_bypass,
            "objective": self.objective,
            "data_objective": self.data_objective,
            "temporal_switches": self.temporal_switches,
            "frequency_switches": self.frequency_switches,
            "selection_counts": list(self.selection_counts),
            "infeasible_cells": [list(value) for value in self.infeasible_cells],
            "routing_plan_sha256": self.routing_plan_sha256,
            "conservative_parent": self.conservative_parent.to_dict(),
            "conservative_parent_certificate_sha256": (
                self.conservative_parent_certificate_sha256
            ),
            "reason": self.reason,
        }


def _metric_indices(panel: RiskPanelIdentityV2) -> dict[str, int]:
    return {name: index for index, name in enumerate(panel.metric_names)}


def candidate_costs_v2(
    panel: RiskPanelV2,
    config: RiskRouterConfigV2,
) -> tuple[np.ndarray, np.ndarray]:
    time, bands, candidates, _ = panel.validate()
    config.validate(panel.identity)
    upper = np.asarray(panel.upper, dtype=np.float64)
    available = np.asarray(panel.available, dtype=bool)
    indices = _metric_indices(panel.identity)
    normalized = np.zeros(
        (time, bands, candidates, len(config.critical_thresholds)),
        dtype=np.float64,
    )
    feasible = np.ones((time, bands, candidates), dtype=bool)
    for axis, (name, threshold) in enumerate(config.critical_thresholds):
        metric = indices[name]
        present = available[..., metric]
        value = np.where(present, upper[..., metric], 0.0)
        feasible &= present
        feasible &= value <= (
            float(threshold) + float(config.feasibility_tolerance)
        )
        normalized[..., axis] = value / float(threshold)
    cost = float(config.critical_slack_weight) * np.max(
        normalized, axis=-1
    )
    for name, weight in config.secondary_weights:
        metric = indices[name]
        present = available[..., metric]
        value = np.where(present, upper[..., metric], 0.0)
        feasible &= present
        cost += float(weight) * value
    if np.any(~np.isfinite(cost)):
        raise ValueError("risk unary costs are non-finite")
    return cost, feasible


def _plan_sha(
    labels: np.ndarray,
    panel: RiskPanelV2,
    config: RiskRouterConfigV2,
    *,
    raw_parent_bypass: bool,
) -> str:
    array = np.ascontiguousarray(labels, dtype="<i4")
    header = {
        "schema": ROUTE_SCHEMA,
        "risk_panel_sha256": panel.sha256,
        "router_sha256": config.sha256(panel.identity),
        "raw_parent_bypass": bool(raw_parent_bypass),
        "conservative_parent": panel.identity.conservative_parent.to_dict(),
        "conservative_parent_certificate_sha256": (
            panel.identity.conservative_parent_certificate_sha256
        ),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


def _fallback(
    panel: RiskPanelV2,
    config: RiskRouterConfigV2,
    infeasible: np.ndarray,
    status: str,
    reason: str,
) -> RouteDecisionV2:
    time, bands, candidates, _ = panel.validate()
    index = panel.identity.conservative_index
    labels = np.full((time, bands), index, dtype=np.int32)
    counts = np.bincount(labels.ravel(), minlength=candidates)
    cells = tuple(
        tuple(int(value) for value in row)
        for row in np.argwhere(infeasible)
    )
    return RouteDecisionV2(
        status=status,
        labels=labels,
        raw_parent_bypass=True,
        objective=None,
        data_objective=None,
        temporal_switches=0,
        frequency_switches=0,
        selection_counts=tuple(int(value) for value in counts),
        infeasible_cells=cells,
        routing_plan_sha256=_plan_sha(
            labels, panel, config, raw_parent_bypass=True
        ),
        conservative_parent=panel.identity.conservative_parent,
        conservative_parent_certificate_sha256=(
            panel.identity.conservative_parent_certificate_sha256
        ),
        reason=reason,
    )


def _problem(
    cost: np.ndarray,
    feasible: np.ndarray,
    config: RiskRouterConfigV2,
    *,
    excluded_labels: np.ndarray | None = None,
):
    from scipy.optimize import Bounds, LinearConstraint
    from scipy.sparse import coo_matrix

    time, bands, candidates = cost.shape
    cells = time * bands
    temporal_edges = [
        (t * bands + b, (t + 1) * bands + b)
        for t in range(time - 1) for b in range(bands)
    ]
    frequency_edges = [
        (t * bands + b, t * bands + b + 1)
        for t in range(time) for b in range(bands - 1)
    ]
    edges = (
        [(u, v, float(config.temporal_switch_penalty))
         for u, v in temporal_edges]
        + [(u, v, float(config.frequency_switch_penalty))
           for u, v in frequency_edges]
    )
    x_count = cells * candidates
    variable_count = x_count + len(edges) * candidates
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:x_count] = cost.reshape(-1) / float(cells)
    for edge_index, (_, _, penalty) in enumerate(edges):
        objective[
            x_count + edge_index * candidates:
            x_count + (edge_index + 1) * candidates
        ] = penalty / (2.0 * float(cells))
    lower_bound = np.zeros(variable_count, dtype=np.float64)
    upper_bound = np.ones(variable_count, dtype=np.float64)
    upper_bound[:x_count] = feasible.reshape(-1).astype(np.float64)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for cell in range(cells):
        for candidate in range(candidates):
            rows.append(row); columns.append(cell * candidates + candidate)
            values.append(1.0)
        lower.append(1.0); upper.append(1.0); row += 1
    for edge_index, (left, right, _) in enumerate(edges):
        for candidate in range(candidates):
            d_index = x_count + edge_index * candidates + candidate
            for first, second in ((left, right), (right, left)):
                rows.extend((row, row, row))
                columns.extend((
                    first * candidates + candidate,
                    second * candidates + candidate,
                    d_index,
                ))
                values.extend((1.0, -1.0, -1.0))
                lower.append(-np.inf); upper.append(0.0); row += 1
    if excluded_labels is not None:
        labels = np.asarray(excluded_labels, dtype=np.int64).reshape(-1)
        if labels.size != cells:
            raise ValueError("excluded route shape differs")
        for cell, candidate in enumerate(labels):
            rows.append(row)
            columns.append(cell * candidates + int(candidate))
            values.append(1.0)
        lower.append(-np.inf)
        upper.append(float(cells - 1))
        row += 1
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(row, variable_count)
    ).tocsr()
    return (
        objective,
        np.r_[np.ones(x_count, dtype=np.int8),
              np.zeros(variable_count - x_count, dtype=np.int8)],
        Bounds(lower_bound, upper_bound),
        LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        x_count,
    )


def _solve(problem, config: RiskRouterConfigV2):
    from scipy.optimize import milp

    objective, integrality, bounds, constraints, _ = problem
    return milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(config.milp_time_limit_seconds),
            "mip_rel_gap": float(config.milp_relative_gap),
        },
    )


def _certified_optimal(result: Any, config: RiskRouterConfigV2) -> bool:
    if not result.success or result.x is None or int(result.status) != 0:
        return False
    gap = getattr(result, "mip_gap", None)
    return gap is None or (
        math.isfinite(float(gap))
        and float(gap) <= float(config.milp_relative_gap) + 1e-12
    )


def solve_risk_route_v2(
    panel: RiskPanelV2,
    config: RiskRouterConfigV2,
) -> RouteDecisionV2:
    """Return a unique certified Potts route or the raw verified parent."""

    cost, feasible = candidate_costs_v2(panel, config)
    time, bands, candidates = cost.shape
    no_candidate = feasible.sum(axis=-1) == 0
    if np.any(no_candidate):
        return _fallback(
            panel, config, no_candidate,
            "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE",
            "one or more cells have no confidently feasible candidate",
        )
    problem = _problem(cost, feasible, config)
    result = _solve(problem, config)
    if not _certified_optimal(result, config):
        return _fallback(
            panel, config, np.zeros((time, bands), dtype=bool),
            "ABSTAIN_SOLVER_UNCERTIFIED",
            "primary structured solve lacks an optimality certificate",
        )
    x_count = problem[-1]
    x = result.x[:x_count].reshape(time, bands, candidates)
    if (
        np.max(np.abs(x.sum(axis=-1) - 1.0))
        > config.integrality_tolerance
        or np.max(np.minimum(np.abs(x), np.abs(x - 1.0)))
        > config.integrality_tolerance
    ):
        raise RiskRouterV2Error("MILP returned a nonintegral route")
    labels = np.argmax(x, axis=-1).astype(np.int32)
    selected_feasible = np.take_along_axis(
        feasible, labels[..., None], axis=-1
    )[..., 0]
    if not np.all(selected_feasible):
        raise RiskRouterV2Error("MILP selected an infeasible candidate")

    selected_cost = np.take_along_axis(
        cost, labels[..., None], axis=-1
    )[..., 0]
    cells = float(time * bands)
    data = float(selected_cost.mean())
    temporal = int(np.count_nonzero(labels[1:] != labels[:-1]))
    frequency = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    total = (
        data
        + config.temporal_switch_penalty * temporal / cells
        + config.frequency_switch_penalty * frequency / cells
    )
    if not math.isclose(
        float(result.fun), total,
        rel_tol=config.objective_tolerance,
        abs_tol=config.objective_tolerance,
    ):
        raise RiskRouterV2Error(
            f"MILP objective mismatch: {result.fun} != {total}"
        )

    alternate = _solve(
        _problem(cost, feasible, config, excluded_labels=labels), config
    )
    if _certified_optimal(alternate, config):
        if float(alternate.fun) <= total + config.objective_tolerance:
            return _fallback(
                panel, config, np.zeros((time, bands), dtype=bool),
                "ABSTAIN_NONUNIQUE_OPTIMUM",
                "more than one route is optimal within the frozen tolerance",
            )
    elif int(getattr(alternate, "status", -1)) != 2:
        return _fallback(
            panel, config, np.zeros((time, bands), dtype=bool),
            "ABSTAIN_UNIQUENESS_UNCERTIFIED",
            "alternate-route solve did not certify infeasibility or a worse optimum",
        )

    counts = np.bincount(labels.ravel(), minlength=candidates)
    return RouteDecisionV2(
        status="ROUTE",
        labels=labels,
        raw_parent_bypass=False,
        objective=total,
        data_objective=data,
        temporal_switches=temporal,
        frequency_switches=frequency,
        selection_counts=tuple(int(value) for value in counts),
        infeasible_cells=(),
        routing_plan_sha256=_plan_sha(
            labels, panel, config, raw_parent_bypass=False
        ),
        conservative_parent=panel.identity.conservative_parent,
        conservative_parent_certificate_sha256=(
            panel.identity.conservative_parent_certificate_sha256
        ),
        reason="unique optimum; every selected upper-risk cell is feasible",
    )
