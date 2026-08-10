"""Offline complete-route exact evaluation for D0/R0 submissions.

Promotion is based on complete-route regret and catastrophic false-safe behavior,
not crop classification accuracy.  This module evaluates one submitted route
against an exact-reference partition using the same frozen policy and measured
structured objective.  Critical missing/above-threshold selections are reported
as catastrophic false safes; missing required secondary evidence is a separate
fail-closed result; abstention is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

import numpy as np

from .counterfactual_risk_exhaustive_decoder_v1 import (
    ExhaustiveRouteDecisionV1,
    decode_exhaustive_small_grid,
    recompute_route_objective,
)
from .counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedUpperRiskPanelV1,
)
from .counterfactual_risk_routing_preflight_v1 import (
    FrozenRoutingPolicyV1,
    RoutingPreflightV1,
)

SUBMISSION_SCHEMA = "audio-extract/counterfactual-route-submission/v1"
VIOLATION_SCHEMA = "audio-extract/counterfactual-route-violation/v1"
EVALUATION_SCHEMA = "audio-extract/counterfactual-route-evaluation/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class CompleteRouteEvaluationError(ValueError):
    """A submitted route or complete-route exact evaluation is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise CompleteRouteEvaluationError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CompleteRouteEvaluationError(
            f"{name} must be a non-empty, trim-stable string"
        )
    return value


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


def _labels_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<i4")
    header = {
        "schema": SUBMISSION_SCHEMA,
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


@dataclass(frozen=True)
class ArmRouteSubmissionV1:
    arm_id: str
    route_evidence_sha256: str
    source_status: str
    status: str
    labels: np.ndarray | None
    reason: str

    @classmethod
    def from_exhaustive_decision(
        cls,
        arm_id: str,
        decision: ExhaustiveRouteDecisionV1,
    ) -> "ArmRouteSubmissionV1":
        decision.validate()
        status = "ROUTE" if decision.status == "ROUTE" else "ABSTAIN"
        result = cls(
            arm_id=arm_id,
            route_evidence_sha256=decision.sha256,
            source_status=decision.status,
            status=status,
            labels=decision.labels,
            reason=decision.reason,
        )
        result.validate()
        return result

    def validate(self) -> None:
        _text(self.arm_id, "arm_id")
        _sha(self.route_evidence_sha256, "route_evidence_sha256")
        _text(self.source_status, "source_status")
        _text(self.reason, "reason")
        if self.status not in {"ROUTE", "ABSTAIN"}:
            raise CompleteRouteEvaluationError(
                "submission status must be ROUTE or ABSTAIN"
            )
        if self.status == "ROUTE":
            if self.labels is None:
                raise CompleteRouteEvaluationError(
                    "ROUTE submission lacks labels"
                )
            labels = np.asarray(self.labels)
            if labels.ndim != 2 or min(labels.shape) < 1:
                raise CompleteRouteEvaluationError(
                    "submitted labels must be a non-empty two-dimensional grid"
                )
            if not np.issubdtype(labels.dtype, np.integer):
                raise CompleteRouteEvaluationError(
                    "submitted labels must be integers"
                )
        elif self.labels is not None:
            raise CompleteRouteEvaluationError(
                "ABSTAIN submission must not include labels"
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
            "schema": SUBMISSION_SCHEMA,
            "arm_id": self.arm_id,
            "route_evidence_sha256": self.route_evidence_sha256,
            "source_status": self.source_status,
            "status": self.status,
            "labels_shape": (
                None if self.labels is None else list(np.asarray(self.labels).shape)
            ),
            "labels_sha256": self.labels_sha256,
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True, order=True)
class SelectedRiskViolationV1:
    time_index: int
    band_index: int
    candidate_index: int
    metric_name: str
    kind: str
    threshold: float | None
    value: float | None

    def validate(self) -> None:
        for name, value in (
            ("time_index", self.time_index),
            ("band_index", self.band_index),
            ("candidate_index", self.candidate_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CompleteRouteEvaluationError(
                    f"{name} must be a non-negative integer"
                )
        _text(self.metric_name, "metric_name")
        if self.kind not in {
            "critical_missing",
            "critical_above_threshold",
            "required_secondary_missing",
        }:
            raise CompleteRouteEvaluationError(
                f"unknown selected-risk violation kind {self.kind!r}"
            )
        if self.kind == "critical_above_threshold":
            if self.threshold is None or self.value is None:
                raise CompleteRouteEvaluationError(
                    "above-threshold violation lacks value/threshold"
                )
            if (
                not math.isfinite(float(self.threshold))
                or not math.isfinite(float(self.value))
                or self.threshold <= 0
                or self.value < 0
                or self.value <= self.threshold
            ):
                raise CompleteRouteEvaluationError(
                    "above-threshold violation has invalid numeric facts"
                )
        elif self.value is not None:
            raise CompleteRouteEvaluationError(
                "missing-evidence violation must not invent a value"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": VIOLATION_SCHEMA,
            "time_index": self.time_index,
            "band_index": self.band_index,
            "candidate_index": self.candidate_index,
            "metric_name": self.metric_name,
            "kind": self.kind,
            "threshold": self.threshold,
            "value": self.value,
        }


@dataclass(frozen=True)
class CompleteRouteEvaluationV1:
    arm_id: str
    exact_evidence_sha256: str
    exact_panel_sha256: str
    exact_preflight_sha256: str
    policy_sha256: str
    exact_oracle_decision_sha256: str
    submission_sha256: str
    status: str
    critical_false_safe_violations: tuple[SelectedRiskViolationV1, ...]
    incomplete_secondary_violations: tuple[SelectedRiskViolationV1, ...]
    exact_oracle_objective: float | None
    submitted_exact_objective: float | None
    selection_regret: float | None
    temporal_switches: int | None
    frequency_switches: int | None
    reason: str

    def validate(self) -> None:
        _text(self.arm_id, "arm_id")
        for name, value in (
            ("exact_evidence_sha256", self.exact_evidence_sha256),
            ("exact_panel_sha256", self.exact_panel_sha256),
            ("exact_preflight_sha256", self.exact_preflight_sha256),
            ("policy_sha256", self.policy_sha256),
            (
                "exact_oracle_decision_sha256",
                self.exact_oracle_decision_sha256,
            ),
            ("submission_sha256", self.submission_sha256),
        ):
            _sha(value, name)
        _text(self.reason, "reason")
        allowed = {
            "ROUTE_SAFE",
            "ROUTE_CATASTROPHIC_FALSE_SAFE",
            "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE",
            "ABSTAIN",
            "ORACLE_UNAVAILABLE",
        }
        if self.status not in allowed:
            raise CompleteRouteEvaluationError(
                f"unknown complete-route evaluation status {self.status!r}"
            )
        critical = tuple(self.critical_false_safe_violations)
        incomplete = tuple(self.incomplete_secondary_violations)
        for violation in (*critical, *incomplete):
            violation.validate()
        if critical != tuple(sorted(set(critical))):
            raise CompleteRouteEvaluationError(
                "critical false-safe violations must be unique and canonical"
            )
        if incomplete != tuple(sorted(set(incomplete))):
            raise CompleteRouteEvaluationError(
                "secondary evidence violations must be unique and canonical"
            )
        if any(
            violation.kind == "required_secondary_missing"
            for violation in critical
        ) or any(
            violation.kind != "required_secondary_missing"
            for violation in incomplete
        ):
            raise CompleteRouteEvaluationError(
                "critical and secondary violation collections are mixed"
            )
        for name, value in (
            ("temporal_switches", self.temporal_switches),
            ("frequency_switches", self.frequency_switches),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise CompleteRouteEvaluationError(
                    f"{name} must be a non-negative integer when available"
                )
        numeric = (
            self.exact_oracle_objective,
            self.submitted_exact_objective,
            self.selection_regret,
        )
        if any(
            value is not None
            and (not math.isfinite(float(value)) or float(value) < 0)
            for value in numeric
        ):
            raise CompleteRouteEvaluationError(
                "route objectives/regret must be finite and non-negative when available"
            )

        if self.status == "ROUTE_SAFE":
            if critical or incomplete:
                raise CompleteRouteEvaluationError(
                    "safe route contains exact violations"
                )
            if any(value is None for value in numeric):
                raise CompleteRouteEvaluationError(
                    "safe route lacks exact objective/regret"
                )
            if self.temporal_switches is None or self.frequency_switches is None:
                raise CompleteRouteEvaluationError(
                    "safe route lacks switch counts"
                )
        elif self.status == "ROUTE_CATASTROPHIC_FALSE_SAFE":
            if not critical:
                raise CompleteRouteEvaluationError(
                    "catastrophic false-safe status lacks critical violations"
                )
            if self.submitted_exact_objective is not None or (
                self.selection_regret is not None
            ):
                raise CompleteRouteEvaluationError(
                    "critical false-safe route must not report a feasible regret"
                )
        elif self.status == "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE":
            if critical or not incomplete:
                raise CompleteRouteEvaluationError(
                    "incomplete-evidence status has wrong violation classes"
                )
            if self.submitted_exact_objective is not None or (
                self.selection_regret is not None
            ):
                raise CompleteRouteEvaluationError(
                    "incomplete route must not report a feasible regret"
                )
        else:
            if critical or incomplete:
                raise CompleteRouteEvaluationError(
                    "abstention/oracle-unavailable status cannot select violations"
                )
            if self.submitted_exact_objective is not None or (
                self.selection_regret is not None
            ):
                raise CompleteRouteEvaluationError(
                    "abstention/oracle-unavailable status cannot report route regret"
                )
            if self.temporal_switches is not None or (
                self.frequency_switches is not None
            ):
                raise CompleteRouteEvaluationError(
                    "abstention/oracle-unavailable status cannot report switches"
                )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": EVALUATION_SCHEMA,
            "arm_id": self.arm_id,
            "exact_evidence_sha256": self.exact_evidence_sha256,
            "exact_panel_sha256": self.exact_panel_sha256,
            "exact_preflight_sha256": self.exact_preflight_sha256,
            "policy_sha256": self.policy_sha256,
            "exact_oracle_decision_sha256": (
                self.exact_oracle_decision_sha256
            ),
            "submission_sha256": self.submission_sha256,
            "status": self.status,
            "critical_false_safe_violations": [
                violation.identity_dict()
                for violation in self.critical_false_safe_violations
            ],
            "incomplete_secondary_violations": [
                violation.identity_dict()
                for violation in self.incomplete_secondary_violations
            ],
            "exact_oracle_objective": self.exact_oracle_objective,
            "submitted_exact_objective": self.submitted_exact_objective,
            "selection_regret": self.selection_regret,
            "temporal_switches": self.temporal_switches,
            "frequency_switches": self.frequency_switches,
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


def _selected_violations(
    labels: np.ndarray,
    exact_panel: PartitionedUpperRiskPanelV1,
    policy: FrozenRoutingPolicyV1,
) -> tuple[
    tuple[SelectedRiskViolationV1, ...],
    tuple[SelectedRiskViolationV1, ...],
]:
    upper = np.asarray(exact_panel.upper, dtype=np.float64)
    available = np.asarray(exact_panel.available, dtype=bool)
    metric_index = {
        name: index for index, name in enumerate(exact_panel.metric_names)
    }
    critical: list[SelectedRiskViolationV1] = []
    incomplete: list[SelectedRiskViolationV1] = []
    for time_index in range(exact_panel.time_cell_count):
        for band_index in range(exact_panel.band_count):
            candidate = int(labels[time_index, band_index])
            for name, threshold in policy.critical_thresholds:
                metric = metric_index[name]
                if not available[time_index, band_index, candidate, metric]:
                    critical.append(
                        SelectedRiskViolationV1(
                            time_index,
                            band_index,
                            candidate,
                            name,
                            "critical_missing",
                            float(threshold),
                            None,
                        )
                    )
                    continue
                value = float(
                    upper[time_index, band_index, candidate, metric]
                )
                if value > (
                    float(threshold) + float(policy.feasibility_tolerance)
                ):
                    critical.append(
                        SelectedRiskViolationV1(
                            time_index,
                            band_index,
                            candidate,
                            name,
                            "critical_above_threshold",
                            float(threshold)
                            + float(policy.feasibility_tolerance),
                            value,
                        )
                    )
            if policy.require_secondary_evidence:
                for name, _ in policy.secondary_weights:
                    metric = metric_index[name]
                    if not available[
                        time_index, band_index, candidate, metric
                    ]:
                        incomplete.append(
                            SelectedRiskViolationV1(
                                time_index,
                                band_index,
                                candidate,
                                name,
                                "required_secondary_missing",
                                None,
                                None,
                            )
                        )
    return tuple(sorted(set(critical))), tuple(sorted(set(incomplete)))


def evaluate_complete_route_submission(
    submission: ArmRouteSubmissionV1,
    exact_panel: PartitionedUpperRiskPanelV1,
    exact_preflight: RoutingPreflightV1,
    policy: FrozenRoutingPolicyV1,
    *,
    exact_evidence_sha256: str,
    maximum_cells: int = 12,
) -> CompleteRouteEvaluationV1:
    """Evaluate one D0/R0 route against exact complete-route evidence."""

    submission.validate()
    exact_panel.validate()
    exact_preflight.validate()
    policy.validate(exact_panel.metric_names)
    _sha(exact_evidence_sha256, "exact_evidence_sha256")
    oracle = decode_exhaustive_small_grid(
        exact_panel,
        exact_preflight,
        policy,
        maximum_cells=maximum_cells,
    )
    common = dict(
        arm_id=submission.arm_id,
        exact_evidence_sha256=exact_evidence_sha256,
        exact_panel_sha256=exact_panel.sha256,
        exact_preflight_sha256=exact_preflight.sha256,
        policy_sha256=policy.sha256,
        exact_oracle_decision_sha256=oracle.sha256,
        submission_sha256=submission.sha256,
    )
    if oracle.status != "ROUTE":
        result = CompleteRouteEvaluationV1(
            **common,
            status="ORACLE_UNAVAILABLE",
            critical_false_safe_violations=(),
            incomplete_secondary_violations=(),
            exact_oracle_objective=None,
            submitted_exact_objective=None,
            selection_regret=None,
            temporal_switches=None,
            frequency_switches=None,
            reason=(
                "exact complete-route oracle is unavailable; no arm can be promoted"
            ),
        )
        result.validate()
        return result
    assert oracle.objective is not None
    if submission.status == "ABSTAIN":
        result = CompleteRouteEvaluationV1(
            **common,
            status="ABSTAIN",
            critical_false_safe_violations=(),
            incomplete_secondary_violations=(),
            exact_oracle_objective=float(oracle.objective),
            submitted_exact_objective=None,
            selection_regret=None,
            temporal_switches=None,
            frequency_switches=None,
            reason="arm abstained instead of returning a complete route",
        )
        result.validate()
        return result

    assert submission.labels is not None
    labels = np.asarray(submission.labels)
    expected_shape = (
        exact_panel.time_cell_count,
        exact_panel.band_count,
    )
    if labels.shape != expected_shape:
        raise CompleteRouteEvaluationError(
            f"submitted label grid differs: {labels.shape} != {expected_shape}"
        )
    candidate_count = len(exact_panel.candidate_slot_sha256s)
    if np.any(labels < 0) or np.any(labels >= candidate_count):
        raise CompleteRouteEvaluationError(
            "submitted route names a candidate outside the exact panel"
        )
    critical, incomplete = _selected_violations(
        labels, exact_panel, policy
    )
    if critical:
        result = CompleteRouteEvaluationV1(
            **common,
            status="ROUTE_CATASTROPHIC_FALSE_SAFE",
            critical_false_safe_violations=critical,
            incomplete_secondary_violations=incomplete,
            exact_oracle_objective=float(oracle.objective),
            submitted_exact_objective=None,
            selection_regret=None,
            temporal_switches=int(
                np.count_nonzero(labels[1:] != labels[:-1])
            ),
            frequency_switches=int(
                np.count_nonzero(labels[:, 1:] != labels[:, :-1])
            ),
            reason=(
                "submitted route selected one or more exact critical failures"
            ),
        )
        # Critical false-safe may also have secondary missing evidence; keep the
        # latter as diagnostic but the status is governed by the critical event.
        result.validate()
        return result
    if incomplete:
        result = CompleteRouteEvaluationV1(
            **common,
            status="ROUTE_INCOMPLETE_REQUIRED_EVIDENCE",
            critical_false_safe_violations=(),
            incomplete_secondary_violations=incomplete,
            exact_oracle_objective=float(oracle.objective),
            submitted_exact_objective=None,
            selection_regret=None,
            temporal_switches=int(
                np.count_nonzero(labels[1:] != labels[:-1])
            ),
            frequency_switches=int(
                np.count_nonzero(labels[:, 1:] != labels[:, :-1])
            ),
            reason=(
                "submitted route lacks required exact secondary evidence"
            ),
        )
        result.validate()
        return result

    total, _, temporal, frequency = recompute_route_objective(
        labels, exact_panel, exact_preflight, policy
    )
    oracle_value = float(oracle.objective)
    if total < oracle_value and not _objective_equal(
        total, oracle_value, policy.objective_tolerance
    ):
        raise CompleteRouteEvaluationError(
            "submitted route beats the independently computed exact optimum"
        )
    regret = max(0.0, total - oracle_value)
    result = CompleteRouteEvaluationV1(
        **common,
        status="ROUTE_SAFE",
        critical_false_safe_violations=(),
        incomplete_secondary_violations=(),
        exact_oracle_objective=oracle_value,
        submitted_exact_objective=total,
        selection_regret=regret,
        temporal_switches=temporal,
        frequency_switches=frequency,
        reason="submitted route is exact-feasible; complete-route regret computed",
    )
    result.validate()
    return result
