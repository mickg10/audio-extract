"""Fail-closed hard-feasibility preflight for partitioned upper risks.

This module does not solve a route.  It freezes and verifies the metric policy,
computes finite unary costs without allowing missing evidence into arithmetic,
marks every unavailable required head infeasible, and declares whether the
complete grid is eligible for the separately certified structured decoder.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedUpperRiskPanelV1,
)

POLICY_SCHEMA = "audio-extract/counterfactual-risk-routing-policy/v1"
PREFLIGHT_SCHEMA = "audio-extract/counterfactual-risk-routing-preflight/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RoutingPreflightError(ValueError):
    """The routing policy or fail-closed preflight result is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise RoutingPreflightError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _array_sha(value: np.ndarray, *, component: str, dtype: str) -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    header = {
        "schema": PREFLIGHT_SCHEMA,
        "component": component,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class FrozenRoutingPolicyV1:
    critical_thresholds: tuple[tuple[str, float], ...]
    secondary_weights: tuple[tuple[str, float], ...] = ()
    critical_slack_weight: float = 1.0
    temporal_switch_penalty: float = 0.05
    frequency_switch_penalty: float = 0.05
    feasibility_tolerance: float = 0.0
    objective_tolerance: float = 1e-10
    whole_track_abstention: bool = True
    require_secondary_evidence: bool = True

    def validate(self, metric_names: tuple[str, ...] | None = None) -> None:
        critical_names = tuple(name for name, _ in self.critical_thresholds)
        secondary_names = tuple(name for name, _ in self.secondary_weights)
        if not critical_names or len(set(critical_names)) != len(critical_names):
            raise RoutingPreflightError(
                "critical thresholds must be non-empty and unique"
            )
        if len(set(secondary_names)) != len(secondary_names):
            raise RoutingPreflightError("secondary weights must be unique")
        if set(critical_names) & set(secondary_names):
            raise RoutingPreflightError(
                "a metric cannot be both critical and secondary"
            )
        if metric_names is not None:
            unknown = (set(critical_names) | set(secondary_names)) - set(
                metric_names
            )
            if unknown:
                raise RoutingPreflightError(
                    f"routing policy contains unknown metrics: {sorted(unknown)}"
                )
        for name, value in self.critical_thresholds:
            if not isinstance(name, str) or not name:
                raise RoutingPreflightError(
                    "critical metric names must be non-empty strings"
                )
            threshold = float(value)
            if not math.isfinite(threshold) or threshold <= 0:
                raise RoutingPreflightError(
                    f"critical threshold {name!r} must be finite and positive"
                )
        for name, value in self.secondary_weights:
            if not isinstance(name, str) or not name:
                raise RoutingPreflightError(
                    "secondary metric names must be non-empty strings"
                )
            weight = float(value)
            if not math.isfinite(weight) or weight <= 0:
                raise RoutingPreflightError(
                    f"secondary weight {name!r} must be finite and strictly positive"
                )
        for name in (
            "critical_slack_weight",
            "temporal_switch_penalty",
            "frequency_switch_penalty",
            "feasibility_tolerance",
            "objective_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise RoutingPreflightError(
                    f"{name} must be finite and non-negative"
                )
        if self.critical_slack_weight <= 0:
            raise RoutingPreflightError(
                "critical_slack_weight must be strictly positive"
            )
        if self.objective_tolerance <= 0:
            raise RoutingPreflightError(
                "objective_tolerance must be strictly positive"
            )
        if self.whole_track_abstention is not True:
            raise RoutingPreflightError(
                "the first D0/R0 policy requires whole-track abstention"
            )
        if not isinstance(self.require_secondary_evidence, bool):
            raise RoutingPreflightError(
                "require_secondary_evidence must be boolean"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": POLICY_SCHEMA,
            "critical_thresholds": [
                [name, float(value)]
                for name, value in self.critical_thresholds
            ],
            "secondary_weights": [
                [name, float(value)]
                for name, value in self.secondary_weights
            ],
            "critical_slack_weight": float(self.critical_slack_weight),
            "temporal_switch_penalty": float(
                self.temporal_switch_penalty
            ),
            "frequency_switch_penalty": float(
                self.frequency_switch_penalty
            ),
            "feasibility_tolerance": float(self.feasibility_tolerance),
            "objective_tolerance": float(self.objective_tolerance),
            "whole_track_abstention": self.whole_track_abstention,
            "require_secondary_evidence": self.require_secondary_evidence,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class RoutingPreflightV1:
    panel_sha256: str
    policy_sha256: str
    cost: np.ndarray
    feasible: np.ndarray
    no_feasible_candidate_cells: tuple[tuple[int, int], ...]
    status: str

    def validate(self) -> None:
        _sha(self.panel_sha256, "preflight panel_sha256")
        _sha(self.policy_sha256, "preflight policy_sha256")
        cost = np.asarray(self.cost, dtype=np.float64)
        feasible = np.asarray(self.feasible)
        if cost.ndim != 3 or min(cost.shape) < 1:
            raise RoutingPreflightError(
                "routing unary cost must be a non-empty (time,band,candidate) tensor"
            )
        if feasible.shape != cost.shape or feasible.dtype != np.bool_:
            raise RoutingPreflightError(
                "routing feasibility must be a boolean tensor matching cost"
            )
        if not np.all(np.isfinite(cost)) or np.any(cost < 0):
            raise RoutingPreflightError(
                "routing unary costs must be finite and non-negative"
            )
        observed = tuple(
            tuple(int(value) for value in row)
            for row in np.argwhere(~np.any(feasible, axis=-1))
        )
        if self.no_feasible_candidate_cells != observed:
            raise RoutingPreflightError(
                "no-feasible-candidate cell list differs from the mask"
            )
        expected_status = (
            "READY_FOR_CERTIFIED_DECODER"
            if not observed
            else "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
        )
        if self.status != expected_status:
            raise RoutingPreflightError(
                "routing preflight status differs from fail-closed feasibility"
            )

    @property
    def cost_sha256(self) -> str:
        self.validate()
        return _array_sha(
            np.asarray(self.cost, dtype=np.float64),
            component="cost",
            dtype="<f8",
        )

    @property
    def feasibility_sha256(self) -> str:
        self.validate()
        return _array_sha(
            np.asarray(self.feasible, dtype=np.uint8),
            component="feasible",
            dtype="u1",
        )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PREFLIGHT_SCHEMA,
            "panel_sha256": self.panel_sha256,
            "policy_sha256": self.policy_sha256,
            "status": self.status,
            "cost_shape": list(np.asarray(self.cost).shape),
            "cost_sha256": self.cost_sha256,
            "feasibility_sha256": self.feasibility_sha256,
            "no_feasible_candidate_cells": [
                list(value) for value in self.no_feasible_candidate_cells
            ],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


def preflight_partitioned_upper_risks(
    panel: PartitionedUpperRiskPanelV1,
    policy: FrozenRoutingPolicyV1,
) -> RoutingPreflightV1:
    """Compute hard feasibility and finite unary cost without solving a route."""

    panel.validate()
    policy.validate(panel.metric_names)
    if policy.sha256 != panel.route_policy_sha256:
        raise RoutingPreflightError(
            "routing policy identity differs from the partitioned panel"
        )
    metric_index = {
        name: index for index, name in enumerate(panel.metric_names)
    }
    upper = np.asarray(panel.upper, dtype=np.float64)
    available = np.asarray(panel.available, dtype=bool)
    time_count, band_count, candidate_count, _ = upper.shape
    feasible = np.ones(
        (time_count, band_count, candidate_count), dtype=bool
    )
    normalized = np.zeros(
        (
            time_count,
            band_count,
            candidate_count,
            len(policy.critical_thresholds),
        ),
        dtype=np.float64,
    )
    for axis, (name, threshold) in enumerate(policy.critical_thresholds):
        index = metric_index[name]
        present = available[..., index]
        value = np.where(present, upper[..., index], 0.0)
        feasible &= present
        feasible &= value <= (
            float(threshold) + float(policy.feasibility_tolerance)
        )
        normalized[..., axis] = value / float(threshold)
    cost = float(policy.critical_slack_weight) * np.max(
        normalized, axis=-1
    )
    for name, weight in policy.secondary_weights:
        index = metric_index[name]
        present = available[..., index]
        value = np.where(present, upper[..., index], 0.0)
        if policy.require_secondary_evidence:
            feasible &= present
        cost += float(weight) * value
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise RoutingPreflightError(
            "routing preflight produced invalid unary costs"
        )
    no_candidate = tuple(
        tuple(int(value) for value in row)
        for row in np.argwhere(~np.any(feasible, axis=-1))
    )
    status = (
        "READY_FOR_CERTIFIED_DECODER"
        if not no_candidate
        else "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    )
    cost.setflags(write=False)
    feasible.setflags(write=False)
    result = RoutingPreflightV1(
        panel_sha256=panel.sha256,
        policy_sha256=policy.sha256,
        cost=cost,
        feasible=feasible,
        no_feasible_candidate_cells=no_candidate,
        status=status,
    )
    result.validate()
    return result
