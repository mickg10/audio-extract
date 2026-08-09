"""Counterfactual-risk routing over a frozen separator candidate bank.

This dormant research component implements the structured half of a proposed
opera separator student:

    inference-available features
        -> predicted per-candidate upper defect risks
        -> deterministic constrained Potts decoder
        -> one shared-stereo candidate label per time/frequency cell

The module never consumes clean accompaniment or vocal references. Exact truth is
used only by ``build_exact_teacher_targets`` to construct training labels outside
the inference path.

Compared with direct imitation of one hard oracle route, a risk student can use
all ``K x D`` counterfactual candidate/defect labels, preserve calibrated
uncertainty, change hard thresholds without retraining the encoder, and abstain
when no candidate is confidently feasible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import math

import numpy as np


PANEL_SCHEMA = "audio-extract/counterfactual-risk-panel/v1"
ROUTER_SCHEMA = "audio-extract/counterfactual-risk-router/v1"
TEACHER_SCHEMA = "audio-extract/counterfactual-risk-teacher/v1"
ROUTE_SCHEMA = "audio-extract/counterfactual-risk-route/v1"


class RiskRouterError(RuntimeError):
    """The risk panel, teacher, or structured decoder is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "").lower()
    if not result.startswith("sha256:") or len(result) != 71:
        raise ValueError(f"{name} must be sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be sha256:<64 hex>") from exc
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_mapping(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class RiskPanelIdentity:
    """Identity of the frozen inference feature/risk prediction contract."""

    candidate_ids: tuple[str, ...]
    metric_names: tuple[str, ...]
    feature_contract_sha256: str
    risk_model_sha256: str
    query_encoder_sha256: str | None = None

    def validate(self) -> None:
        if len(self.candidate_ids) < 2:
            raise ValueError("at least two candidate IDs are required")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be unique and ordered")
        for index, value in enumerate(self.candidate_ids):
            _sha(value, f"candidate_ids[{index}]")
        if not self.metric_names or any(not str(name) for name in self.metric_names):
            raise ValueError("metric names must be non-empty")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric names must be unique and ordered")
        _sha(self.feature_contract_sha256, "feature_contract_sha256")
        _sha(self.risk_model_sha256, "risk_model_sha256")
        if self.query_encoder_sha256 is not None:
            _sha(self.query_encoder_sha256, "query_encoder_sha256")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PANEL_SCHEMA,
            "candidate_ids": list(self.candidate_ids),
            "metric_names": list(self.metric_names),
            "feature_contract_sha256": self.feature_contract_sha256,
            "risk_model_sha256": self.risk_model_sha256,
            "query_encoder_sha256": self.query_encoder_sha256,
        }

    @property
    def sha256(self) -> str:
        return _hash_mapping(self.identity_dict())


@dataclass(frozen=True)
class RiskRouterConfig:
    """Frozen constrained decoder policy.

    ``critical_thresholds`` and ``secondary_weights`` are ordered pairs to make
    their canonical identity explicit.
    """

    critical_thresholds: tuple[tuple[str, float], ...]
    secondary_weights: tuple[tuple[str, float], ...] = ()
    critical_slack_weight: float = 1.0
    temporal_switch_penalty: float = 0.05
    frequency_switch_penalty: float = 0.05
    conservative_index: int = 0
    teacher_near_tie_margin: float = 0.05
    milp_time_limit_seconds: float = 180.0
    milp_relative_gap: float = 0.0

    def validate(self, panel: RiskPanelIdentity) -> None:
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
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"secondary weight {name} must be non-negative")
        for name in (
            "critical_slack_weight",
            "temporal_switch_penalty",
            "frequency_switch_penalty",
            "teacher_near_tie_margin",
            "milp_time_limit_seconds",
            "milp_relative_gap",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.milp_time_limit_seconds <= 0:
            raise ValueError("milp_time_limit_seconds must be positive")
        if not 0 <= int(self.conservative_index) < len(panel.candidate_ids):
            raise ValueError("conservative index is outside candidate bank")

    def identity_dict(self, panel: RiskPanelIdentity) -> dict[str, Any]:
        self.validate(panel)
        return {
            "schema": ROUTER_SCHEMA,
            "panel_sha256": panel.sha256,
            "critical_thresholds": [
                [name, float(value)] for name, value in self.critical_thresholds
            ],
            "secondary_weights": [
                [name, float(value)] for name, value in self.secondary_weights
            ],
            "critical_slack_weight": float(self.critical_slack_weight),
            "temporal_switch_penalty": float(self.temporal_switch_penalty),
            "frequency_switch_penalty": float(self.frequency_switch_penalty),
            "conservative_index": int(self.conservative_index),
            "teacher_near_tie_margin": float(self.teacher_near_tie_margin),
            "milp_time_limit_seconds": float(self.milp_time_limit_seconds),
            "milp_relative_gap": float(self.milp_relative_gap),
        }

    def sha256(self, panel: RiskPanelIdentity) -> str:
        return _hash_mapping(self.identity_dict(panel))


@dataclass(frozen=True)
class RiskPanel:
    """Predicted upper risk and availability arrays.

    Shapes:
        upper:     (time, band, candidate, metric)
        available: same shape

    There is intentionally no audio-channel axis. One selected label owns both
    stereo channels for a cell.
    """

    identity: RiskPanelIdentity
    upper: np.ndarray
    available: np.ndarray

    def validate(self) -> tuple[int, int, int, int]:
        self.identity.validate()
        upper = np.asarray(self.upper, dtype=np.float64)
        available = np.asarray(self.available, dtype=bool)
        if upper.ndim != 4 or upper.shape != available.shape:
            raise ValueError(
                "upper and available must share (time,band,candidate,metric)"
            )
        time, bands, candidates, metrics = upper.shape
        if min(time, bands, candidates, metrics) < 1:
            raise ValueError("risk panel dimensions must be positive")
        if candidates != len(self.identity.candidate_ids):
            raise ValueError("risk panel candidate axis differs from identity")
        if metrics != len(self.identity.metric_names):
            raise ValueError("risk panel metric axis differs from identity")
        if np.any(~np.isfinite(upper[available])):
            raise ValueError("available predicted risks must be finite")
        if np.any(upper[available] < 0):
            raise ValueError("available predicted risks must be non-negative")
        return upper.shape


@dataclass(frozen=True)
class TeacherTargets:
    schema: str
    labels: np.ndarray
    available: np.ndarray
    margins: np.ndarray
    feasible_counts: np.ndarray
    panel_sha256: str
    router_sha256: str

    def validate(self) -> tuple[int, int]:
        labels = np.asarray(self.labels)
        available = np.asarray(self.available, dtype=bool)
        margins = np.asarray(self.margins, dtype=np.float64)
        counts = np.asarray(self.feasible_counts)
        if labels.ndim != 2:
            raise ValueError("teacher labels must be (time,band)")
        if available.shape != labels.shape or margins.shape != labels.shape:
            raise ValueError("teacher masks/margins must match labels")
        if counts.shape != labels.shape:
            raise ValueError("teacher feasible counts must match labels")
        if np.any(counts < 0):
            raise ValueError("teacher feasible counts must be non-negative")
        if np.any(~np.isfinite(margins[available])) or np.any(
            margins[available] < 0
        ):
            raise ValueError("available teacher margins must be finite/nonnegative")
        _sha(self.panel_sha256, "teacher panel SHA")
        _sha(self.router_sha256, "teacher router SHA")
        return labels.shape


@dataclass(frozen=True)
class RouteDecision:
    status: str
    labels: np.ndarray
    objective: float | None
    data_objective: float | None
    temporal_switches: int
    frequency_switches: int
    selection_counts: tuple[int, ...]
    infeasible_cells: tuple[tuple[int, int], ...]
    routing_plan_sha256: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_SCHEMA,
            "status": self.status,
            "labels_shape": list(np.asarray(self.labels).shape),
            "objective": self.objective,
            "data_objective": self.data_objective,
            "temporal_switches": self.temporal_switches,
            "frequency_switches": self.frequency_switches,
            "selection_counts": list(self.selection_counts),
            "infeasible_cells": [list(value) for value in self.infeasible_cells],
            "routing_plan_sha256": self.routing_plan_sha256,
            "reason": self.reason,
        }


def _metric_indices(panel: RiskPanelIdentity) -> dict[str, int]:
    return {name: index for index, name in enumerate(panel.metric_names)}


def _candidate_costs(
    panel: RiskPanel,
    config: RiskRouterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scalar unaries and hard feasibility from upper defect risks."""

    time, bands, candidates, _ = panel.validate()
    config.validate(panel.identity)
    upper = np.asarray(panel.upper, dtype=np.float64)
    available = np.asarray(panel.available, dtype=bool)
    indices = _metric_indices(panel.identity)
    critical = tuple(config.critical_thresholds)
    secondary = tuple(config.secondary_weights)

    feasible = np.ones((time, bands, candidates), dtype=bool)
    normalized = np.zeros((time, bands, candidates, len(critical)))
    for defect_index, (name, threshold) in enumerate(critical):
        metric = indices[name]
        feasible &= available[..., metric]
        value = upper[..., metric]
        feasible &= value <= float(threshold)
        normalized[..., defect_index] = value / float(threshold)

    if critical:
        slack = np.max(normalized, axis=-1)
    else:  # pragma: no cover - config requires critical metrics
        slack = np.zeros((time, bands, candidates))
    cost = float(config.critical_slack_weight) * slack
    for name, weight in secondary:
        metric = indices[name]
        feasible &= available[..., metric]
        cost = cost + float(weight) * upper[..., metric]
    if np.any(~np.isfinite(cost)):
        raise ValueError("risk unary costs are non-finite")
    return cost, feasible


def build_exact_teacher_targets(
    exact_risk: RiskPanel,
    config: RiskRouterConfig,
) -> TeacherTargets:
    """Create near-tie-aware direct-imitation targets from exact risks.

    This function is training-only. Exact references are represented solely by
    the already computed risk tensor. A cell is teachable only when at least one
    candidate is feasible and the best/second-best normalized cost margin is at
    least the frozen threshold. Third or unavailable candidates are never
    relabelled as the conservative parent.
    """

    cost, feasible = _candidate_costs(exact_risk, config)
    time, bands, candidates = cost.shape
    labels = np.full((time, bands), -1, dtype=np.int32)
    available = np.zeros((time, bands), dtype=bool)
    margins = np.full((time, bands), np.nan, dtype=np.float64)
    counts = feasible.sum(axis=-1).astype(np.int32)

    for t in range(time):
        for b in range(bands):
            allowed = np.flatnonzero(feasible[t, b])
            if allowed.size == 0:
                continue
            order = allowed[np.argsort(cost[t, b, allowed], kind="stable")]
            best = int(order[0])
            margin = (
                math.inf
                if order.size == 1
                else float(cost[t, b, order[1]] - cost[t, b, best])
            )
            labels[t, b] = best
            margins[t, b] = margin
            available[t, b] = (
                margin >= float(config.teacher_near_tie_margin)
            )

    result = TeacherTargets(
        schema=TEACHER_SCHEMA,
        labels=labels,
        available=available,
        margins=margins,
        feasible_counts=counts,
        panel_sha256=exact_risk.identity.sha256,
        router_sha256=config.sha256(exact_risk.identity),
    )
    result.validate()
    return result


def _plan_sha(
    labels: np.ndarray,
    panel: RiskPanelIdentity,
    config: RiskRouterConfig,
) -> str:
    array = np.ascontiguousarray(labels, dtype="<i4")
    header = {
        "schema": ROUTE_SCHEMA,
        "panel_sha256": panel.sha256,
        "router_sha256": config.sha256(panel),
        "candidate_ids": list(panel.candidate_ids),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


def _fallback(
    panel: RiskPanel,
    config: RiskRouterConfig,
    infeasible: np.ndarray,
    reason: str,
) -> RouteDecision:
    time, bands, candidates, _ = panel.validate()
    labels = np.full(
        (time, bands), int(config.conservative_index), dtype=np.int32
    )
    counts = np.bincount(labels.ravel(), minlength=candidates)
    cells = tuple(
        tuple(int(value) for value in row) for row in np.argwhere(infeasible)
    )
    return RouteDecision(
        status="ABSTAIN_USE_CONSERVATIVE_WHOLE_TRACK",
        labels=labels,
        objective=None,
        data_objective=None,
        temporal_switches=0,
        frequency_switches=0,
        selection_counts=tuple(int(value) for value in counts),
        infeasible_cells=cells,
        routing_plan_sha256=_plan_sha(labels, panel.identity, config),
        reason=reason,
    )


def solve_risk_route(
    panel: RiskPanel,
    config: RiskRouterConfig,
) -> RouteDecision:
    """Solve a global discrete Potts route or abstain fail-closed.

    The route uses only predicted upper risks and frozen identities. Clean truth,
    oracle labels, and exact metrics are not accepted by this inference API.
    """

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    cost, feasible = _candidate_costs(panel, config)
    time, bands, candidates = cost.shape
    no_candidate = feasible.sum(axis=-1) == 0
    if np.any(no_candidate):
        return _fallback(
            panel,
            config,
            no_candidate,
            "one or more cells have no confidently feasible candidate",
        )

    cells = time * bands
    temporal_edges = [
        (t * bands + b, (t + 1) * bands + b)
        for t in range(time - 1)
        for b in range(bands)
    ]
    frequency_edges = [
        (t * bands + b, t * bands + b + 1)
        for t in range(time)
        for b in range(bands - 1)
    ]
    edges = (
        [
            (left, right, float(config.temporal_switch_penalty))
            for left, right in temporal_edges
        ]
        + [
            (left, right, float(config.frequency_switch_penalty))
            for left, right in frequency_edges
        ]
    )

    x_count = cells * candidates
    d_count = len(edges) * candidates
    variable_count = x_count + d_count
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
            rows.append(row)
            columns.append(cell * candidates + candidate)
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        row += 1

    for edge_index, (left_cell, right_cell, _) in enumerate(edges):
        for candidate in range(candidates):
            d_index = x_count + edge_index * candidates + candidate
            for left_cell_i, right_cell_i in (
                (left_cell, right_cell),
                (right_cell, left_cell),
            ):
                rows.extend((row, row, row))
                columns.extend((
                    left_cell_i * candidates + candidate,
                    right_cell_i * candidates + candidate,
                    d_index,
                ))
                values.extend((1.0, -1.0, -1.0))
                lower.append(-np.inf)
                upper.append(0.0)
                row += 1

    matrix = coo_matrix(
        (values, (rows, columns)), shape=(row, variable_count)
    ).tocsr()
    result = milp(
        objective,
        integrality=np.r_[
            np.ones(x_count, dtype=np.int8),
            np.zeros(d_count, dtype=np.int8),
        ],
        bounds=Bounds(lower_bound, upper_bound),
        constraints=LinearConstraint(
            matrix, np.asarray(lower), np.asarray(upper)
        ),
        options={
            "time_limit": float(config.milp_time_limit_seconds),
            "mip_rel_gap": float(config.milp_relative_gap),
        },
    )
    if not result.success or result.x is None:
        return _fallback(
            panel,
            config,
            np.zeros((time, bands), dtype=bool),
            f"structured decoder lacks an optimality certificate: {result.message}",
        )
    gap = getattr(result, "mip_gap", None)
    if gap is not None and float(gap) > float(config.milp_relative_gap) + 1e-12:
        return _fallback(
            panel,
            config,
            np.zeros((time, bands), dtype=bool),
            f"structured decoder MIP gap {gap} exceeds configured limit",
        )

    labels = np.argmax(
        result.x[:x_count].reshape(time, bands, candidates), axis=-1
    ).astype(np.int32)
    selected_feasible = np.take_along_axis(
        feasible, labels[..., None], axis=-1
    )[..., 0]
    if not np.all(selected_feasible):
        raise RiskRouterError("MILP selected an infeasible candidate")

    selected_cost = np.take_along_axis(
        cost, labels[..., None], axis=-1
    )[..., 0]
    data_objective = float(selected_cost.mean())
    temporal_switches = int(np.count_nonzero(labels[1:] != labels[:-1]))
    frequency_switches = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    total = (
        data_objective
        + float(config.temporal_switch_penalty)
        * temporal_switches / float(cells)
        + float(config.frequency_switch_penalty)
        * frequency_switches / float(cells)
    )
    counts = np.bincount(labels.ravel(), minlength=candidates)
    return RouteDecision(
        status="ROUTE",
        labels=labels,
        objective=total,
        data_objective=data_objective,
        temporal_switches=temporal_switches,
        frequency_switches=frequency_switches,
        selection_counts=tuple(int(value) for value in counts),
        infeasible_cells=(),
        routing_plan_sha256=_plan_sha(labels, panel.identity, config),
        reason="all selected upper-risk cells satisfy the frozen constraints",
    )
