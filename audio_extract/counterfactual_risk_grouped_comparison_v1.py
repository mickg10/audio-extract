"""Grouped held-out D0/R0 comparison with no promotion decision.

The report joins complete-route exact evaluations only when D0 and R0 were run
under one frozen, equal-compute preregistration and cover exactly the same
source-family-weighted held-out units.  It reports coverage, abstention,
catastrophic false-safe behavior, incomplete evidence, exact route regret, and
route switching.  Promotion belongs to a separate preregistered policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, median
from typing import Any

from .counterfactual_risk_complete_route_evaluation_v1 import (
    CompleteRouteEvaluationV1,
)
from .counterfactual_risk_dataset_contract_v1 import GroupFamilyIdentity

UNIT_SCHEMA = "audio-extract/counterfactual-held-out-unit/v1"
ARM_SCHEMA = "audio-extract/counterfactual-comparison-arm/v1"
PREREGISTRATION_SCHEMA = "audio-extract/d0-r0-comparison-preregistration/v1"
WORK_EVALUATION_SCHEMA = "audio-extract/d0-r0-work-evaluation/v1"
ARM_SUMMARY_SCHEMA = "audio-extract/d0-r0-arm-summary/v1"
PAIRWISE_SUMMARY_SCHEMA = "audio-extract/d0-r0-pairwise-summary/v1"
REPORT_SCHEMA = "audio-extract/d0-r0-grouped-comparison-report/v1"
REPORT_STATUS = "COMPLETE_NO_PROMOTION_DECISION"
AGGREGATION_UNIT = "source_family_sha256/v2"
AGGREGATION_WEIGHT_MODE = "one_source_family_one_vote/v1"
ARM_METHODS = {
    "D0": "structured_route_imitation/v1",
    "R0": "counterfactual_risk_prediction/v1",
}
ROUTE_STATUSES = {
    "ROUTE_SAFE",
    "ROUTE_CATASTROPHIC_FALSE_SAFE",
    "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE",
}
ALL_STATUSES = ROUTE_STATUSES | {"ABSTAIN", "ORACLE_UNAVAILABLE"}
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupedComparisonError(ValueError):
    """The comparison is incomplete, cross-contaminated, or self-inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedComparisonError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedComparisonError(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise GroupedComparisonError(
            f"{name} must be a non-empty, trim-stable string"
        )
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GroupedComparisonError(
            f"comparison value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise GroupedComparisonError(f"{name} must be finite and non-negative")
    return result


def _optional_finite(value: float | None, name: str) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise GroupedComparisonError(f"{name} must be finite when present")


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class HeldOutUnitV1:
    """One independent source-family-weighted complete-route test unit."""

    work_id: str
    recording_session_id: str
    target_singer_id: str
    source_family_sha256: str
    query_condition_sha256: str
    query_quality_stratum_sha256: str
    exact_evidence_sha256: str
    exact_panel_sha256: str
    exact_preflight_sha256: str
    exact_oracle_decision_sha256: str

    @property
    def group_family(self) -> GroupFamilyIdentity:
        return GroupFamilyIdentity(
            work_id=self.work_id,
            recording_session_id=self.recording_session_id,
            target_singer_id=self.target_singer_id,
            source_family_sha256=self.source_family_sha256,
            query_condition_sha256=self.query_condition_sha256,
        )

    @property
    def group_family_sha256(self) -> str:
        self.group_family.validate()
        return self.group_family.sha256

    def validate(self) -> None:
        self.group_family.validate()
        for name in (
            "query_quality_stratum_sha256",
            "exact_evidence_sha256",
            "exact_panel_sha256",
            "exact_preflight_sha256",
            "exact_oracle_decision_sha256",
        ):
            _sha(getattr(self, name), name)

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": UNIT_SCHEMA,
            **self.group_family.identity_dict(),
            "group_family_sha256": self.group_family_sha256,
            "query_quality_stratum_sha256": (
                self.query_quality_stratum_sha256
            ),
            "exact_evidence_sha256": self.exact_evidence_sha256,
            "exact_panel_sha256": self.exact_panel_sha256,
            "exact_preflight_sha256": self.exact_preflight_sha256,
            "exact_oracle_decision_sha256": (
                self.exact_oracle_decision_sha256
            ),
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class ComparisonArmV1:
    arm_id: str
    method_id: str
    artifact_manifest_sha256: str
    compute_budget_sha256: str
    checkpoint_rule_sha256: str
    inference_policy_sha256: str
    source_commit: str
    status: str = "frozen"

    def validate(self) -> None:
        if self.status != "frozen":
            raise GroupedComparisonError("comparison arm must be frozen")
        if self.arm_id not in ARM_METHODS:
            raise GroupedComparisonError(f"unknown arm {self.arm_id!r}")
        if self.method_id != ARM_METHODS[self.arm_id]:
            raise GroupedComparisonError(
                f"{self.arm_id} method differs from the frozen comparison"
            )
        for name in (
            "artifact_manifest_sha256",
            "compute_budget_sha256",
            "checkpoint_rule_sha256",
            "inference_policy_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.source_commit, "arm source_commit")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": ARM_SCHEMA, **self.__dict__}

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class GroupedComparisonPreregistrationV1:
    experiment_id: str
    dataset_sha256: str
    split_sha256: str
    test_subset_sha256: str
    exact_evidence_manifest_sha256: str
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    routing_policy_sha256: str
    complete_route_evaluation_contract_sha256: str
    comparison_policy_sha256: str
    promotion_policy_sha256: str
    paired_regret_tolerance: float
    d0_arm: ComparisonArmV1
    r0_arm: ComparisonArmV1
    expected_units: tuple[HeldOutUnitV1, ...]
    source_commit: str
    aggregation_unit: str = AGGREGATION_UNIT
    aggregation_weight_mode: str = AGGREGATION_WEIGHT_MODE
    status: str = "frozen"

    def validate(self) -> None:
        if self.status != "frozen":
            raise GroupedComparisonError("preregistration must be frozen")
        _text(self.experiment_id, "experiment_id")
        for name in (
            "dataset_sha256",
            "split_sha256",
            "test_subset_sha256",
            "exact_evidence_manifest_sha256",
            "candidate_panel_sha256",
            "feature_contract_sha256",
            "metric_contract_sha256",
            "routing_policy_sha256",
            "complete_route_evaluation_contract_sha256",
            "comparison_policy_sha256",
            "promotion_policy_sha256",
        ):
            _sha(getattr(self, name), name)
        _finite_nonnegative(
            self.paired_regret_tolerance, "paired_regret_tolerance"
        )
        if self.aggregation_unit != AGGREGATION_UNIT:
            raise GroupedComparisonError(
                "aggregation unit must be the source-family SHA"
            )
        if self.aggregation_weight_mode != AGGREGATION_WEIGHT_MODE:
            raise GroupedComparisonError(
                "each source family must receive exactly one vote"
            )
        self.d0_arm.validate()
        self.r0_arm.validate()
        if (self.d0_arm.arm_id, self.r0_arm.arm_id) != ("D0", "R0"):
            raise GroupedComparisonError("arms must be ordered D0, R0")
        if self.d0_arm.compute_budget_sha256 != self.r0_arm.compute_budget_sha256:
            raise GroupedComparisonError("D0/R0 compute budgets differ")
        if self.d0_arm.checkpoint_rule_sha256 != self.r0_arm.checkpoint_rule_sha256:
            raise GroupedComparisonError("D0/R0 checkpoint rules differ")
        _git(self.source_commit, "preregistration source_commit")
        if (
            self.d0_arm.source_commit != self.source_commit
            or self.r0_arm.source_commit != self.source_commit
        ):
            raise GroupedComparisonError("arm and preregistration commits differ")
        if len(self.expected_units) < 2:
            raise GroupedComparisonError(
                "comparison requires at least two held-out source families"
            )
        for unit in self.expected_units:
            unit.validate()
        ordered = tuple(sorted(self.expected_units, key=lambda row: row.sha256))
        if self.expected_units != ordered:
            raise GroupedComparisonError(
                "expected units must be in canonical SHA order"
            )
        if len({unit.sha256 for unit in self.expected_units}) != len(
            self.expected_units
        ):
            raise GroupedComparisonError("expected units contain duplicates")
        if len({unit.source_family_sha256 for unit in self.expected_units}) != len(
            self.expected_units
        ):
            raise GroupedComparisonError(
                "one source family may appear only once in comparison v1"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PREREGISTRATION_SCHEMA,
            "status": self.status,
            "experiment_id": self.experiment_id,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "test_subset_sha256": self.test_subset_sha256,
            "exact_evidence_manifest_sha256": (
                self.exact_evidence_manifest_sha256
            ),
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "routing_policy_sha256": self.routing_policy_sha256,
            "complete_route_evaluation_contract_sha256": (
                self.complete_route_evaluation_contract_sha256
            ),
            "comparison_policy_sha256": self.comparison_policy_sha256,
            "promotion_policy_sha256": self.promotion_policy_sha256,
            "paired_regret_tolerance": float(self.paired_regret_tolerance),
            "aggregation_unit": self.aggregation_unit,
            "aggregation_weight_mode": self.aggregation_weight_mode,
            "d0_arm": self.d0_arm.identity_dict(),
            "d0_arm_sha256": self.d0_arm.sha256,
            "r0_arm": self.r0_arm.identity_dict(),
            "r0_arm_sha256": self.r0_arm.sha256,
            "expected_units": [row.identity_dict() for row in self.expected_units],
            "source_commit": self.source_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class WorkArmEvaluationV1:
    unit: HeldOutUnitV1
    arm_run_sha256: str
    dataset_sha256: str
    split_sha256: str
    test_subset_sha256: str
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    preregistration_sha256: str
    evaluation: CompleteRouteEvaluationV1

    @property
    def arm_id(self) -> str:
        return self.evaluation.arm_id

    def validate(self, prereg: GroupedComparisonPreregistrationV1) -> None:
        prereg.validate()
        self.unit.validate()
        self.evaluation.validate()
        for name in (
            "arm_run_sha256",
            "dataset_sha256",
            "split_sha256",
            "test_subset_sha256",
            "candidate_panel_sha256",
            "feature_contract_sha256",
            "metric_contract_sha256",
            "preregistration_sha256",
        ):
            _sha(getattr(self, name), name)
        registered = tuple(
            row for row in prereg.expected_units if row.sha256 == self.unit.sha256
        )
        if len(registered) != 1 or registered[0] != self.unit:
            raise GroupedComparisonError("evaluation unit is not preregistered")
        bindings = {
            "dataset_sha256": prereg.dataset_sha256,
            "split_sha256": prereg.split_sha256,
            "test_subset_sha256": prereg.test_subset_sha256,
            "candidate_panel_sha256": prereg.candidate_panel_sha256,
            "feature_contract_sha256": prereg.feature_contract_sha256,
            "metric_contract_sha256": prereg.metric_contract_sha256,
            "preregistration_sha256": prereg.sha256,
        }
        different = [
            name for name, expected in bindings.items()
            if getattr(self, name) != expected
        ]
        if different:
            raise GroupedComparisonError(
                f"work evaluation context differs: {different}"
            )
        if self.evaluation.policy_sha256 != prereg.routing_policy_sha256:
            raise GroupedComparisonError("work evaluation uses another policy")
        exact = {
            "exact_evidence_sha256": self.unit.exact_evidence_sha256,
            "exact_panel_sha256": self.unit.exact_panel_sha256,
            "exact_preflight_sha256": self.unit.exact_preflight_sha256,
            "exact_oracle_decision_sha256": (
                self.unit.exact_oracle_decision_sha256
            ),
        }
        exact_drift = [
            name for name, expected in exact.items()
            if getattr(self.evaluation, name) != expected
        ]
        if exact_drift:
            raise GroupedComparisonError(
                "evaluation exact evidence differs from held-out unit: "
                f"{exact_drift}"
            )
        expected_arm = {
            "D0": prereg.d0_arm.sha256,
            "R0": prereg.r0_arm.sha256,
        }
        if self.arm_id not in expected_arm:
            raise GroupedComparisonError(f"unknown arm {self.arm_id!r}")
        if self.arm_run_sha256 != expected_arm[self.arm_id]:
            raise GroupedComparisonError("evaluation uses another arm run")

    def identity_dict(
        self, prereg: GroupedComparisonPreregistrationV1
    ) -> dict[str, Any]:
        self.validate(prereg)
        return {
            "schema": WORK_EVALUATION_SCHEMA,
            "unit": self.unit.identity_dict(),
            "unit_sha256": self.unit.sha256,
            "group_family_sha256": self.unit.group_family_sha256,
            "arm_id": self.arm_id,
            "arm_run_sha256": self.arm_run_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "test_subset_sha256": self.test_subset_sha256,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "complete_route_evaluation": self.evaluation.identity_dict(),
            "complete_route_evaluation_sha256": self.evaluation.sha256,
        }


@dataclass(frozen=True)
class ArmSummaryV1:
    arm_id: str
    unit_count: int
    oracle_available_count: int
    route_submission_count: int
    safe_route_count: int
    catastrophic_false_safe_count: int
    incomplete_evidence_count: int
    abstention_count: int
    oracle_unavailable_count: int
    critical_violation_count: int
    incomplete_secondary_violation_count: int
    route_submission_coverage: float | None
    safe_route_coverage: float | None
    abstention_rate: float | None
    mean_safe_regret: float | None
    median_safe_regret: float | None
    worst_safe_regret: float | None
    worst_safe_regret_unit_sha256: str | None
    mean_safe_temporal_switches: float | None
    mean_safe_frequency_switches: float | None

    def identity_dict(self) -> dict[str, Any]:
        return {"schema": ARM_SUMMARY_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class PairwiseSummaryV1:
    unit_count: int
    oracle_available_count: int
    both_submitted_route_count: int
    both_safe_count: int
    both_submitted_route_coverage: float | None
    both_safe_coverage: float | None
    d0_safe_r0_not_safe_count: int
    r0_safe_d0_not_safe_count: int
    both_catastrophic_count: int
    d0_catastrophic_r0_not_count: int
    r0_catastrophic_d0_not_count: int
    r0_lower_regret_count: int
    d0_lower_regret_count: int
    regret_tie_count: int
    mean_r0_minus_d0_regret: float | None
    median_r0_minus_d0_regret: float | None
    min_r0_minus_d0_regret: float | None
    max_r0_minus_d0_regret: float | None
    mean_r0_minus_d0_temporal_switches: float | None
    mean_r0_minus_d0_frequency_switches: float | None
    status_cross_tab: tuple[tuple[str, str, int], ...]

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIRWISE_SUMMARY_SCHEMA,
            **{
                name: value
                for name, value in self.__dict__.items()
                if name != "status_cross_tab"
            },
            "status_cross_tab": [list(row) for row in self.status_cross_tab],
        }


def _arm_summary(
    arm_id: str, records: Sequence[WorkArmEvaluationV1]
) -> ArmSummaryV1:
    selected = tuple(row for row in records if row.arm_id == arm_id)
    statuses = [row.evaluation.status for row in selected]
    safe = tuple(
        row for row in selected if row.evaluation.status == "ROUTE_SAFE"
    )
    regrets = [float(row.evaluation.selection_regret) for row in safe]
    temporal = [float(row.evaluation.temporal_switches) for row in safe]
    frequency = [float(row.evaluation.frequency_switches) for row in safe]
    worst_unit = None
    if safe:
        worst = max(regrets)
        worst_unit = min(
            row.unit.sha256
            for row in safe
            if float(row.evaluation.selection_regret) == worst
        )
    unavailable = statuses.count("ORACLE_UNAVAILABLE")
    available = len(selected) - unavailable
    routes = sum(status in ROUTE_STATUSES for status in statuses)
    return ArmSummaryV1(
        arm_id=arm_id,
        unit_count=len(selected),
        oracle_available_count=available,
        route_submission_count=routes,
        safe_route_count=statuses.count("ROUTE_SAFE"),
        catastrophic_false_safe_count=statuses.count(
            "ROUTE_CATASTROPHIC_FALSE_SAFE"
        ),
        incomplete_evidence_count=statuses.count(
            "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE"
        ),
        abstention_count=statuses.count("ABSTAIN"),
        oracle_unavailable_count=unavailable,
        critical_violation_count=sum(
            len(row.evaluation.critical_false_safe_violations)
            for row in selected
        ),
        incomplete_secondary_violation_count=sum(
            len(row.evaluation.incomplete_secondary_violations)
            for row in selected
        ),
        route_submission_coverage=_ratio(routes, available),
        safe_route_coverage=_ratio(len(safe), available),
        abstention_rate=_ratio(statuses.count("ABSTAIN"), available),
        mean_safe_regret=fmean(regrets) if regrets else None,
        median_safe_regret=median(regrets) if regrets else None,
        worst_safe_regret=max(regrets) if regrets else None,
        worst_safe_regret_unit_sha256=worst_unit,
        mean_safe_temporal_switches=fmean(temporal) if temporal else None,
        mean_safe_frequency_switches=fmean(frequency) if frequency else None,
    )


def _pairwise_summary(
    records: Sequence[WorkArmEvaluationV1],
    prereg: GroupedComparisonPreregistrationV1,
) -> PairwiseSummaryV1:
    by_key = {(row.unit.sha256, row.arm_id): row for row in records}
    pairs = [
        (by_key[(unit.sha256, "D0")], by_key[(unit.sha256, "R0")])
        for unit in prereg.expected_units
    ]
    oracle_available = sum(
        d0.evaluation.status != "ORACLE_UNAVAILABLE" for d0, _ in pairs
    )
    both_routes = sum(
        d0.evaluation.status in ROUTE_STATUSES
        and r0.evaluation.status in ROUTE_STATUSES
        for d0, r0 in pairs
    )
    paired_safe = [
        (d0, r0)
        for d0, r0 in pairs
        if d0.evaluation.status == r0.evaluation.status == "ROUTE_SAFE"
    ]
    regret_deltas = [
        float(r0.evaluation.selection_regret)
        - float(d0.evaluation.selection_regret)
        for d0, r0 in paired_safe
    ]
    temporal_deltas = [
        float(r0.evaluation.temporal_switches)
        - float(d0.evaluation.temporal_switches)
        for d0, r0 in paired_safe
    ]
    frequency_deltas = [
        float(r0.evaluation.frequency_switches)
        - float(d0.evaluation.frequency_switches)
        for d0, r0 in paired_safe
    ]
    cross: dict[tuple[str, str], int] = {}
    for d0, r0 in pairs:
        key = (d0.evaluation.status, r0.evaluation.status)
        cross[key] = cross.get(key, 0) + 1
    tolerance = float(prereg.paired_regret_tolerance)
    return PairwiseSummaryV1(
        unit_count=len(pairs),
        oracle_available_count=oracle_available,
        both_submitted_route_count=both_routes,
        both_safe_count=len(paired_safe),
        both_submitted_route_coverage=_ratio(both_routes, oracle_available),
        both_safe_coverage=_ratio(len(paired_safe), oracle_available),
        d0_safe_r0_not_safe_count=sum(
            d0.evaluation.status == "ROUTE_SAFE"
            and r0.evaluation.status != "ROUTE_SAFE"
            for d0, r0 in pairs
        ),
        r0_safe_d0_not_safe_count=sum(
            r0.evaluation.status == "ROUTE_SAFE"
            and d0.evaluation.status != "ROUTE_SAFE"
            for d0, r0 in pairs
        ),
        both_catastrophic_count=sum(
            d0.evaluation.status == r0.evaluation.status
            == "ROUTE_CATASTROPHIC_FALSE_SAFE"
            for d0, r0 in pairs
        ),
        d0_catastrophic_r0_not_count=sum(
            d0.evaluation.status == "ROUTE_CATASTROPHIC_FALSE_SAFE"
            and r0.evaluation.status != "ROUTE_CATASTROPHIC_FALSE_SAFE"
            for d0, r0 in pairs
        ),
        r0_catastrophic_d0_not_count=sum(
            r0.evaluation.status == "ROUTE_CATASTROPHIC_FALSE_SAFE"
            and d0.evaluation.status != "ROUTE_CATASTROPHIC_FALSE_SAFE"
            for d0, r0 in pairs
        ),
        r0_lower_regret_count=sum(x < -tolerance for x in regret_deltas),
        d0_lower_regret_count=sum(x > tolerance for x in regret_deltas),
        regret_tie_count=sum(abs(x) <= tolerance for x in regret_deltas),
        mean_r0_minus_d0_regret=(
            fmean(regret_deltas) if regret_deltas else None
        ),
        median_r0_minus_d0_regret=(
            median(regret_deltas) if regret_deltas else None
        ),
        min_r0_minus_d0_regret=min(regret_deltas) if regret_deltas else None,
        max_r0_minus_d0_regret=max(regret_deltas) if regret_deltas else None,
        mean_r0_minus_d0_temporal_switches=(
            fmean(temporal_deltas) if temporal_deltas else None
        ),
        mean_r0_minus_d0_frequency_switches=(
            fmean(frequency_deltas) if frequency_deltas else None
        ),
        status_cross_tab=tuple(
            sorted((left, right, count) for (left, right), count in cross.items())
        ),
    )


def _validate_matrix(
    records: Sequence[WorkArmEvaluationV1],
    prereg: GroupedComparisonPreregistrationV1,
) -> tuple[WorkArmEvaluationV1, ...]:
    prereg.validate()
    values = tuple(records)
    for row in values:
        row.validate(prereg)
    ordered = tuple(
        sorted(values, key=lambda row: (row.unit.sha256, row.arm_id))
    )
    if values != ordered:
        raise GroupedComparisonError(
            "work evaluations must be in canonical unit/arm order"
        )
    actual = [(row.unit.sha256, row.arm_id) for row in values]
    if len(actual) != len(set(actual)):
        raise GroupedComparisonError("duplicate unit/arm evaluation")
    expected = sorted(
        (unit.sha256, arm)
        for unit in prereg.expected_units
        for arm in ("D0", "R0")
    )
    if actual != expected:
        raise GroupedComparisonError(
            "evaluation matrix does not exactly cover preregistered units"
        )
    by_key = {(row.unit.sha256, row.arm_id): row for row in values}
    for unit in prereg.expected_units:
        d0 = by_key[(unit.sha256, "D0")].evaluation
        r0 = by_key[(unit.sha256, "R0")].evaluation
        exact_fields = (
            "exact_evidence_sha256",
            "exact_panel_sha256",
            "exact_preflight_sha256",
            "policy_sha256",
            "exact_oracle_decision_sha256",
            "exact_oracle_objective",
        )
        drift = [
            name for name in exact_fields
            if getattr(d0, name) != getattr(r0, name)
        ]
        if drift:
            raise GroupedComparisonError(
                f"D0/R0 exact evidence differs for {unit.sha256}: {drift}"
            )
        if (d0.status == "ORACLE_UNAVAILABLE") != (
            r0.status == "ORACLE_UNAVAILABLE"
        ):
            raise GroupedComparisonError(
                "oracle availability differs between arms"
            )
    if {row.evaluation.policy_sha256 for row in values} != {
        prereg.routing_policy_sha256
    }:
        raise GroupedComparisonError("comparison mixes routing policies")
    return values


@dataclass(frozen=True)
class GroupedComparisonReportV1:
    preregistration_sha256: str
    verifier_commit: str
    work_evaluations: tuple[WorkArmEvaluationV1, ...]
    d0_summary: ArmSummaryV1
    r0_summary: ArmSummaryV1
    pairwise_summary: PairwiseSummaryV1
    status: str = REPORT_STATUS

    @classmethod
    def build(
        cls,
        prereg: GroupedComparisonPreregistrationV1,
        records: Sequence[WorkArmEvaluationV1],
        *,
        verifier_commit: str,
    ) -> GroupedComparisonReportV1:
        ordered = tuple(
            sorted(records, key=lambda row: (row.unit.sha256, row.arm_id))
        )
        _validate_matrix(ordered, prereg)
        result = cls(
            preregistration_sha256=prereg.sha256,
            verifier_commit=verifier_commit,
            work_evaluations=ordered,
            d0_summary=_arm_summary("D0", ordered),
            r0_summary=_arm_summary("R0", ordered),
            pairwise_summary=_pairwise_summary(ordered, prereg),
        )
        result.validate(prereg)
        return result

    def validate(self, prereg: GroupedComparisonPreregistrationV1) -> None:
        prereg.validate()
        if self.status != REPORT_STATUS:
            raise GroupedComparisonError(
                "report must not contain a promotion decision"
            )
        _sha(self.preregistration_sha256, "preregistration_sha256")
        if self.preregistration_sha256 != prereg.sha256:
            raise GroupedComparisonError("report names another preregistration")
        _git(self.verifier_commit, "verifier_commit")
        records = _validate_matrix(self.work_evaluations, prereg)
        if self.d0_summary != _arm_summary("D0", records):
            raise GroupedComparisonError("D0 summary differs from evidence")
        if self.r0_summary != _arm_summary("R0", records):
            raise GroupedComparisonError("R0 summary differs from evidence")
        if self.pairwise_summary != _pairwise_summary(records, prereg):
            raise GroupedComparisonError("pairwise summary differs from evidence")
        for summary in (self.d0_summary, self.r0_summary):
            for name, value in summary.__dict__.items():
                if isinstance(value, float):
                    _optional_finite(value, f"{summary.arm_id}.{name}")
        for name, value in self.pairwise_summary.__dict__.items():
            if isinstance(value, float):
                _optional_finite(value, f"pairwise.{name}")

    def identity_dict(
        self, prereg: GroupedComparisonPreregistrationV1
    ) -> dict[str, Any]:
        self.validate(prereg)
        return {
            "schema": REPORT_SCHEMA,
            "status": self.status,
            "promotion_decision": None,
            "preregistration_sha256": self.preregistration_sha256,
            "verifier_commit": self.verifier_commit,
            "work_evaluations": [
                row.identity_dict(prereg) for row in self.work_evaluations
            ],
            "d0_summary": self.d0_summary.identity_dict(),
            "r0_summary": self.r0_summary.identity_dict(),
            "pairwise_summary": self.pairwise_summary.identity_dict(),
        }

    def sha256(self, prereg: GroupedComparisonPreregistrationV1) -> str:
        return _mapping_sha(self.identity_dict(prereg))
