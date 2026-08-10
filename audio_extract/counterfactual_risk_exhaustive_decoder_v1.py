"""Solver-independent exhaustive structured decoder for small D0/R0 grids.

This is an executable reference, not the production large-grid solver.  It uses
the exact partition-normalized physical measure for unary risk, the frozen
switch penalties for adjacent route cells, hard preflight feasibility, explicit
non-unique-optimum abstention, and no implicit candidate fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import itertools
import json
import math
import re

import numpy as np

from .counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedUpperRiskPanelV1,
)
from .counterfactual_risk_routing_preflight_v1 import (
    FrozenRoutingPolicyV1,
    RoutingPreflightV1,
)

DECISION_SCHEMA = "audio-extract/counterfactual-risk-exhaustive-route/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ExhaustiveDecoderError(ValueError):
    """A small-grid structured route or certificate is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise ExhaustiveDecoderError(
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


def _labels_sha(labels: np.ndarray) -> str:
    array = np.ascontiguousarray(labels, dtype="<i4")
    header = {
        "schema": DECISION_SCHEMA,
        "component": "labels",
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


def _objective_equal(first: float, second: float, tolerance: float) -> bool:
    return abs(first - second) <= tolerance * max(
        1.0, abs(first), abs(second)
    )


def recompute_route_objective(
    labels: np.ndarray,
    panel: PartitionedUpperRiskPanelV1,
    preflight: RoutingPreflightV1,
    policy: FrozenRoutingPolicyV1,
) -> tuple[float, float, int, int]:
    """Recompute exact measured unary plus normalized switch penalties."""

    panel.validate()
    preflight.validate()
    policy.validate(panel.metric_names)
    route = np.asarray(labels)
    if route.shape != (panel.time_cell_count, panel.band_count):
        raise ExhaustiveDecoderError(
            "route label grid differs from the partition shape"
        )
    if not np.issubdtype(route.dtype, np.integer):
        raise ExhaustiveDecoderError("route labels must be integers")
    candidate_count = len(panel.candidate_slot_sha256s)
    if np.any(route < 0) or np.any(route >= candidate_count):
        raise ExhaustiveDecoderError(
            "route labels contain a candidate outside the frozen panel"
        )
    feasible = np.asarray(preflight.feasible, dtype=bool)
    selected_feasible = np.take_along_axis(
        feasible, route[..., None], axis=-1
    )[..., 0]
    if not np.all(selected_feasible):
        raise ExhaustiveDecoderError(
            "route selects a candidate that failed hard feasibility"
        )
    cost = np.asarray(preflight.cost, dtype=np.float64)
    selected_cost = np.take_along_axis(
        cost, route[..., None], axis=-1
    )[..., 0]
    measure = panel.normalized_measure_matrix()
    data_terms = [
        float(measure[time_index][band_index])
        * float(selected_cost[time_index, band_index])
        for time_index in range(panel.time_cell_count)
        for band_index in range(panel.band_count)
    ]
    data = math.fsum(data_terms)
    temporal = int(np.count_nonzero(route[1:] != route[:-1]))
    frequency = int(np.count_nonzero(route[:, 1:] != route[:, :-1]))
    cells = float(panel.time_cell_count * panel.band_count)
    total = (
        data
        + float(policy.temporal_switch_penalty) * temporal / cells
        + float(policy.frequency_switch_penalty) * frequency / cells
    )
    if not math.isfinite(total) or total < 0:
        raise ExhaustiveDecoderError(
            "recomputed route objective is invalid"
        )
    return total, data, temporal, frequency


@dataclass(frozen=True)
class ExhaustiveRouteDecisionV1:
    panel_sha256: str
    preflight_sha256: str
    policy_sha256: str
    status: str
    labels: np.ndarray | None
    objective: float | None
    data_objective: float | None
    temporal_switches: int
    frequency_switches: int
    selection_counts: tuple[int, ...]
    optimum_count: int
    reason: str

    def validate(self) -> None:
        for name, value in (
            ("panel_sha256", self.panel_sha256),
            ("preflight_sha256", self.preflight_sha256),
            ("policy_sha256", self.policy_sha256),
        ):
            _sha(value, name)
        allowed = {
            "ROUTE",
            "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE",
            "ABSTAIN_NONUNIQUE_OPTIMUM",
            "ABSTAIN_NO_FEASIBLE_GLOBAL_ROUTE",
        }
        if self.status not in allowed:
            raise ExhaustiveDecoderError(
                f"unknown exhaustive route status {self.status!r}"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise ExhaustiveDecoderError(
                "route decision reason must be non-empty"
            )
        for name, value in (
            ("temporal_switches", self.temporal_switches),
            ("frequency_switches", self.frequency_switches),
            ("optimum_count", self.optimum_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExhaustiveDecoderError(
                    f"{name} must be a non-negative integer"
                )
        if not self.selection_counts or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in self.selection_counts
        ):
            raise ExhaustiveDecoderError(
                "selection_counts must contain non-negative integers"
            )
        if self.status == "ROUTE":
            if self.labels is None:
                raise ExhaustiveDecoderError("ROUTE decision lacks labels")
            labels = np.asarray(self.labels)
            if labels.ndim != 2 or min(labels.shape) < 1:
                raise ExhaustiveDecoderError(
                    "route labels must be a non-empty two-dimensional grid"
                )
            if not np.issubdtype(labels.dtype, np.integer):
                raise ExhaustiveDecoderError("route labels must be integers")
            if self.objective is None or self.data_objective is None:
                raise ExhaustiveDecoderError(
                    "ROUTE decision lacks finite objectives"
                )
            if (
                not math.isfinite(float(self.objective))
                or not math.isfinite(float(self.data_objective))
                or self.objective < 0
                or self.data_objective < 0
            ):
                raise ExhaustiveDecoderError(
                    "route objectives must be finite and non-negative"
                )
            if self.optimum_count != 1:
                raise ExhaustiveDecoderError(
                    "ROUTE requires exactly one certified optimum"
                )
            counts = np.bincount(
                labels.ravel(), minlength=len(self.selection_counts)
            )
            if tuple(int(value) for value in counts) != self.selection_counts:
                raise ExhaustiveDecoderError(
                    "selection counts differ from route labels"
                )
        else:
            if self.labels is not None:
                raise ExhaustiveDecoderError(
                    "abstention decision must not return route labels"
                )
            if self.objective is not None or self.data_objective is not None:
                raise ExhaustiveDecoderError(
                    "abstention decision must not return route objectives"
                )
            if self.temporal_switches or self.frequency_switches:
                raise ExhaustiveDecoderError(
                    "abstention decision cannot report route switches"
                )
            if any(self.selection_counts):
                raise ExhaustiveDecoderError(
                    "abstention decision cannot report candidate selections"
                )
            if self.status == "ABSTAIN_NONUNIQUE_OPTIMUM":
                if self.optimum_count < 2:
                    raise ExhaustiveDecoderError(
                        "non-unique abstention requires at least two optima"
                    )
            elif self.optimum_count != 0:
                raise ExhaustiveDecoderError(
                    "non-route abstention requires zero certified optima"
                )

    def validate_against(
        self,
        panel: PartitionedUpperRiskPanelV1,
        preflight: RoutingPreflightV1,
        policy: FrozenRoutingPolicyV1,
    ) -> None:
        self.validate()
        panel.validate()
        preflight.validate()
        policy.validate(panel.metric_names)
        comparisons = {
            "panel": (self.panel_sha256, panel.sha256),
            "preflight": (self.preflight_sha256, preflight.sha256),
            "policy": (self.policy_sha256, policy.sha256),
        }
        different = sorted(
            name for name, (actual, expected) in comparisons.items()
            if actual != expected
        )
        if different:
            raise ExhaustiveDecoderError(
                f"route decision differs from decoder inputs: {different}"
            )
        if self.status == "ROUTE":
            assert self.labels is not None
            total, data, temporal, frequency = recompute_route_objective(
                self.labels, panel, preflight, policy
            )
            if not _objective_equal(
                float(self.objective), total, policy.objective_tolerance
            ):
                raise ExhaustiveDecoderError(
                    "route objective differs from independent recomputation"
                )
            if not _objective_equal(
                float(self.data_objective),
                data,
                policy.objective_tolerance,
            ):
                raise ExhaustiveDecoderError(
                    "route data objective differs from independent recomputation"
                )
            if (
                self.temporal_switches != temporal
                or self.frequency_switches != frequency
            ):
                raise ExhaustiveDecoderError(
                    "route switch counts differ from independent recomputation"
                )
        elif (
            self.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
            and preflight.status
            != "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
        ):
            raise ExhaustiveDecoderError(
                "no-candidate abstention contradicts the preflight"
            )

    @property
    def labels_sha256(self) -> str | None:
        self.validate()
        if self.labels is None:
            return None
        return _labels_sha(np.asarray(self.labels))

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": DECISION_SCHEMA,
            "panel_sha256": self.panel_sha256,
            "preflight_sha256": self.preflight_sha256,
            "policy_sha256": self.policy_sha256,
            "status": self.status,
            "labels_shape": (
                None if self.labels is None else list(np.asarray(self.labels).shape)
            ),
            "labels_sha256": self.labels_sha256,
            "objective": self.objective,
            "data_objective": self.data_objective,
            "temporal_switches": self.temporal_switches,
            "frequency_switches": self.frequency_switches,
            "selection_counts": list(self.selection_counts),
            "optimum_count": self.optimum_count,
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


def _abstain(
    panel: PartitionedUpperRiskPanelV1,
    preflight: RoutingPreflightV1,
    policy: FrozenRoutingPolicyV1,
    *,
    status: str,
    optimum_count: int,
    reason: str,
) -> ExhaustiveRouteDecisionV1:
    candidate_count = len(panel.candidate_slot_sha256s)
    result = ExhaustiveRouteDecisionV1(
        panel_sha256=panel.sha256,
        preflight_sha256=preflight.sha256,
        policy_sha256=policy.sha256,
        status=status,
        labels=None,
        objective=None,
        data_objective=None,
        temporal_switches=0,
        frequency_switches=0,
        selection_counts=tuple(0 for _ in range(candidate_count)),
        optimum_count=optimum_count,
        reason=reason,
    )
    result.validate_against(panel, preflight, policy)
    return result


def decode_exhaustive_small_grid(
    panel: PartitionedUpperRiskPanelV1,
    preflight: RoutingPreflightV1,
    policy: FrozenRoutingPolicyV1,
    *,
    maximum_cells: int = 12,
) -> ExhaustiveRouteDecisionV1:
    """Return one unique measured Potts route or explicit abstention."""

    panel.validate()
    preflight.validate()
    policy.validate(panel.metric_names)
    if isinstance(maximum_cells, bool) or not isinstance(maximum_cells, int):
        raise ExhaustiveDecoderError(
            "maximum_cells must be a positive integer, not boolean"
        )
    if maximum_cells < 1:
        raise ExhaustiveDecoderError("maximum_cells must be positive")
    comparisons = {
        "panel": (preflight.panel_sha256, panel.sha256),
        "policy": (preflight.policy_sha256, policy.sha256),
        "panel policy": (panel.route_policy_sha256, policy.sha256),
    }
    different = sorted(
        name for name, (actual, expected) in comparisons.items()
        if actual != expected
    )
    if different:
        raise ExhaustiveDecoderError(
            f"decoder inputs have inconsistent identities: {different}"
        )
    cells = panel.time_cell_count * panel.band_count
    if cells > maximum_cells:
        raise ExhaustiveDecoderError(
            f"exhaustive decoder refuses {cells} cells above maximum_cells={maximum_cells}"
        )
    if preflight.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE":
        return _abstain(
            panel,
            preflight,
            policy,
            status="ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE",
            optimum_count=0,
            reason="one or more cells lack a confidently feasible candidate",
        )
    if preflight.status != "READY_FOR_CERTIFIED_DECODER":
        raise ExhaustiveDecoderError(
            f"unknown preflight status {preflight.status!r}"
        )

    candidate_count = len(panel.candidate_slot_sha256s)
    feasible = np.asarray(preflight.feasible, dtype=bool)
    evaluated: list[tuple[float, np.ndarray, float, int, int]] = []
    for assignment in itertools.product(range(candidate_count), repeat=cells):
        labels = np.asarray(assignment, dtype=np.int32).reshape(
            panel.time_cell_count, panel.band_count
        )
        selected = np.take_along_axis(
            feasible, labels[..., None], axis=-1
        )[..., 0]
        if not np.all(selected):
            continue
        total, data, temporal, frequency = recompute_route_objective(
            labels, panel, preflight, policy
        )
        evaluated.append((total, labels.copy(), data, temporal, frequency))
    if not evaluated:
        return _abstain(
            panel,
            preflight,
            policy,
            status="ABSTAIN_NO_FEASIBLE_GLOBAL_ROUTE",
            optimum_count=0,
            reason="no globally feasible route exists despite local preflight",
        )
    best = min(value[0] for value in evaluated)
    optima = tuple(
        value for value in evaluated
        if _objective_equal(value[0], best, policy.objective_tolerance)
    )
    if len(optima) != 1:
        return _abstain(
            panel,
            preflight,
            policy,
            status="ABSTAIN_NONUNIQUE_OPTIMUM",
            optimum_count=len(optima),
            reason="more than one route is optimal within the frozen tolerance",
        )
    total, labels, data, temporal, frequency = optima[0]
    labels.setflags(write=False)
    counts = np.bincount(labels.ravel(), minlength=candidate_count)
    result = ExhaustiveRouteDecisionV1(
        panel_sha256=panel.sha256,
        preflight_sha256=preflight.sha256,
        policy_sha256=policy.sha256,
        status="ROUTE",
        labels=labels,
        objective=total,
        data_objective=data,
        temporal_switches=temporal,
        frequency_switches=frequency,
        selection_counts=tuple(int(value) for value in counts),
        optimum_count=1,
        reason="unique exhaustive optimum; every selected candidate is feasible",
    )
    result.validate_against(panel, preflight, policy)
    return result
