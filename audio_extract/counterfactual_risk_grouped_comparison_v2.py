"""Identity-bound grouped held-out D0/R0 comparison v2.

V1 aggregated correct complete-route statistics but accepted two adjacent
provenance ambiguities:

* a per-work evaluation did not transitively bind the route submission to its
  held-out unit and frozen arm run;
* Python value equality could hide type-changing summary mutations.

V2 keeps the same non-promoting purpose and adds content-bearing route
provenance, exact oracle-availability semantics, and strict summary validation.
It does not train a model, render audio, or choose a winner.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

from .counterfactual_risk_complete_route_evaluation_v1 import (
    ArmRouteSubmissionV1,
    CompleteRouteEvaluationV1,
)
from .counterfactual_risk_grouped_comparison_v1 import (
    ALL_STATUSES,
    ARM_METHODS,
    ComparisonArmV1,
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
    ROUTE_STATUSES,
)

ROUTE_PROVENANCE_SCHEMA = "audio-extract/d0-r0-route-provenance/v2"
WORK_EVALUATION_SCHEMA = "audio-extract/d0-r0-work-evaluation/v2"
ARM_SUMMARY_SCHEMA = "audio-extract/d0-r0-arm-summary/v2"
PAIRWISE_SUMMARY_SCHEMA = "audio-extract/d0-r0-pairwise-summary/v2"
REPORT_SCHEMA = "audio-extract/d0-r0-grouped-comparison-report/v2"
REPORT_STATUS = "COMPLETE_NO_PROMOTION_DECISION"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupedComparisonV2Error(ValueError):
    """The v2 comparison is incomplete, misbound, or self-inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedComparisonV2Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedComparisonV2Error(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


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
        raise GroupedComparisonV2Error(
            f"comparison value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _int(value: Any, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise GroupedComparisonV2Error(
            f"{name} must be a {qualifier} integer"
        )
    return value


def _float_or_none(
    value: Any,
    name: str,
    *,
    nonnegative: bool = False,
    unit_interval: bool = False,
) -> float | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise GroupedComparisonV2Error(
            f"{name} must be a finite float or null"
        )
    if nonnegative and value < 0:
        raise GroupedComparisonV2Error(f"{name} must be non-negative")
    if unit_interval and not 0 <= value <= 1:
        raise GroupedComparisonV2Error(f"{name} must lie in [0,1]")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _same_number(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=8.0 * math.ulp(max(1.0, abs(float(expected)))),
    )


def _expected_arm(
    prereg: GroupedComparisonPreregistrationV1,
    arm_id: str,
) -> ComparisonArmV1:
    if arm_id == "D0":
        return prereg.d0_arm
    if arm_id == "R0":
        return prereg.r0_arm
    raise GroupedComparisonV2Error(f"unknown arm {arm_id!r}")


def _validate_oracle_semantics(
    evaluation: CompleteRouteEvaluationV1,
) -> None:
    evaluation.validate()
    unavailable = evaluation.status == "ORACLE_UNAVAILABLE"
    if unavailable:
        if evaluation.exact_oracle_objective is not None:
            raise GroupedComparisonV2Error(
                "ORACLE_UNAVAILABLE evaluation carries an oracle objective"
            )
    else:
        value = evaluation.exact_oracle_objective
        if (
            type(value) is not float
            or not math.isfinite(value)
            or value < 0
        ):
            raise GroupedComparisonV2Error(
                "oracle-available evaluation lacks a finite objective"
            )


@dataclass(frozen=True)
class ArmRouteProvenanceV2:
    """Content-bearing link from a route submission to one unit and arm."""

    unit_sha256: str
    group_family_sha256: str
    query_condition_sha256: str
    query_quality_stratum_sha256: str
    arm_id: str
    arm_run_sha256: str
    artifact_manifest_sha256: str
    compute_budget_sha256: str
    checkpoint_rule_sha256: str
    inference_policy_sha256: str
    source_commit: str
    route_artifact_sha256: str
    verifier_commit: str
    submission: ArmRouteSubmissionV1
    status: str = "verified"

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> None:
        prereg.validate()
        unit.validate()
        if self.status != "verified":
            raise GroupedComparisonV2Error(
                "route provenance status must be verified"
            )
        for name in (
            "unit_sha256",
            "group_family_sha256",
            "query_condition_sha256",
            "query_quality_stratum_sha256",
            "arm_run_sha256",
            "artifact_manifest_sha256",
            "compute_budget_sha256",
            "checkpoint_rule_sha256",
            "inference_policy_sha256",
            "route_artifact_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.source_commit, "route provenance source_commit")
        _git(self.verifier_commit, "route provenance verifier_commit")
        if self.unit_sha256 != unit.sha256:
            raise GroupedComparisonV2Error(
                "route provenance names another held-out unit"
            )
        if self.group_family_sha256 != unit.group_family_sha256:
            raise GroupedComparisonV2Error(
                "route provenance names another group family"
            )
        if self.query_condition_sha256 != unit.query_condition_sha256:
            raise GroupedComparisonV2Error(
                "route provenance names another query condition"
            )
        if (
            self.query_quality_stratum_sha256
            != unit.query_quality_stratum_sha256
        ):
            raise GroupedComparisonV2Error(
                "route provenance names another query-quality stratum"
            )
        arm = _expected_arm(prereg, self.arm_id)
        arm.validate()
        if self.arm_run_sha256 != arm.sha256:
            raise GroupedComparisonV2Error(
                "route provenance names another frozen arm run"
            )
        comparisons = {
            "artifact_manifest_sha256": arm.artifact_manifest_sha256,
            "compute_budget_sha256": arm.compute_budget_sha256,
            "checkpoint_rule_sha256": arm.checkpoint_rule_sha256,
            "inference_policy_sha256": arm.inference_policy_sha256,
        }
        drift = [
            name
            for name, expected in comparisons.items()
            if getattr(self, name) != expected
        ]
        if drift:
            raise GroupedComparisonV2Error(
                f"route provenance arm contract differs: {drift}"
            )
        if self.source_commit != arm.source_commit:
            raise GroupedComparisonV2Error(
                "route provenance and arm source commits differ"
            )
        self.submission.validate()
        if self.submission.arm_id != self.arm_id:
            raise GroupedComparisonV2Error(
                "route submission arm differs from provenance"
            )
        if (
            self.submission.route_evidence_sha256
            != self.route_artifact_sha256
        ):
            raise GroupedComparisonV2Error(
                "route submission names another route artifact"
            )

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> dict[str, Any]:
        self.validate(prereg, unit)
        return {
            "schema": ROUTE_PROVENANCE_SCHEMA,
            "status": self.status,
            "unit_sha256": self.unit_sha256,
            "group_family_sha256": self.group_family_sha256,
            "query_condition_sha256": self.query_condition_sha256,
            "query_quality_stratum_sha256": (
                self.query_quality_stratum_sha256
            ),
            "arm_id": self.arm_id,
            "arm_run_sha256": self.arm_run_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "compute_budget_sha256": self.compute_budget_sha256,
            "checkpoint_rule_sha256": self.checkpoint_rule_sha256,
            "inference_policy_sha256": self.inference_policy_sha256,
            "source_commit": self.source_commit,
            "route_artifact_sha256": self.route_artifact_sha256,
            "verifier_commit": self.verifier_commit,
            "submission": self.submission.identity_dict(),
            "submission_sha256": self.submission.sha256,
        }

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg, unit))


@dataclass(frozen=True)
class WorkArmEvaluationV2:
    unit: HeldOutUnitV1
    route_provenance: ArmRouteProvenanceV2
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
        return self.route_provenance.arm_id

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> None:
        prereg.validate()
        self.unit.validate()
        self.route_provenance.validate(prereg, self.unit)
        _validate_oracle_semantics(self.evaluation)
        for name in (
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
            row
            for row in prereg.expected_units
            if row.sha256 == self.unit.sha256
        )
        if len(registered) != 1 or registered[0] != self.unit:
            raise GroupedComparisonV2Error(
                "work evaluation unit is not preregistered"
            )
        expected = {
            "dataset_sha256": prereg.dataset_sha256,
            "split_sha256": prereg.split_sha256,
            "test_subset_sha256": prereg.test_subset_sha256,
            "candidate_panel_sha256": prereg.candidate_panel_sha256,
            "feature_contract_sha256": prereg.feature_contract_sha256,
            "metric_contract_sha256": prereg.metric_contract_sha256,
            "preregistration_sha256": prereg.sha256,
        }
        drift = [
            name
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise GroupedComparisonV2Error(
                f"work evaluation context differs: {drift}"
            )
        if self.evaluation.arm_id != self.arm_id:
            raise GroupedComparisonV2Error(
                "complete-route evaluation arm differs from route provenance"
            )
        if (
            self.evaluation.submission_sha256
            != self.route_provenance.submission.sha256
        ):
            raise GroupedComparisonV2Error(
                "complete-route evaluation names another route submission"
            )
        if self.evaluation.policy_sha256 != prereg.routing_policy_sha256:
            raise GroupedComparisonV2Error(
                "complete-route evaluation uses another routing policy"
            )
        exact = {
            "exact_evidence_sha256": self.unit.exact_evidence_sha256,
            "exact_panel_sha256": self.unit.exact_panel_sha256,
            "exact_preflight_sha256": self.unit.exact_preflight_sha256,
            "exact_oracle_decision_sha256": (
                self.unit.exact_oracle_decision_sha256
            ),
        }
        exact_drift = [
            name
            for name, value in exact.items()
            if getattr(self.evaluation, name) != value
        ]
        if exact_drift:
            raise GroupedComparisonV2Error(
                "complete-route evidence differs from held-out unit: "
                f"{exact_drift}"
            )

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> dict[str, Any]:
        self.validate(prereg)
        return {
            "schema": WORK_EVALUATION_SCHEMA,
            "unit": self.unit.identity_dict(),
            "unit_sha256": self.unit.sha256,
            "group_family_sha256": self.unit.group_family_sha256,
            "route_provenance": self.route_provenance.identity_dict(
                prereg, self.unit
            ),
            "route_provenance_sha256": self.route_provenance.sha256(
                prereg, self.unit
            ),
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
class ArmSummaryV2:
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

    def validate(self) -> None:
        if self.arm_id not in ARM_METHODS:
            raise GroupedComparisonV2Error("unknown arm summary")
        count_names = (
            "unit_count",
            "oracle_available_count",
            "route_submission_count",
            "safe_route_count",
            "catastrophic_false_safe_count",
            "incomplete_evidence_count",
            "abstention_count",
            "oracle_unavailable_count",
            "critical_violation_count",
            "incomplete_secondary_violation_count",
        )
        for name in count_names:
            _int(getattr(self, name), f"{self.arm_id}.{name}")
        if (
            self.oracle_available_count + self.oracle_unavailable_count
            != self.unit_count
        ):
            raise GroupedComparisonV2Error(
                "arm oracle counts do not sum to unit_count"
            )
        if (
            self.safe_route_count
            + self.catastrophic_false_safe_count
            + self.incomplete_evidence_count
            != self.route_submission_count
        ):
            raise GroupedComparisonV2Error(
                "arm route-status counts do not sum"
            )
        if (
            self.route_submission_count + self.abstention_count
            != self.oracle_available_count
        ):
            raise GroupedComparisonV2Error(
                "arm route/abstention counts do not cover oracle-available units"
            )
        if (
            self.catastrophic_false_safe_count == 0
        ) != (self.critical_violation_count == 0):
            raise GroupedComparisonV2Error(
                "catastrophic status and critical violations disagree"
            )
        if self.critical_violation_count < self.catastrophic_false_safe_count:
            raise GroupedComparisonV2Error(
                "critical violation count is smaller than catastrophic units"
            )
        if (
            self.incomplete_evidence_count == 0
        ) != (self.incomplete_secondary_violation_count == 0):
            raise GroupedComparisonV2Error(
                "incomplete status and secondary violations disagree"
            )
        if (
            self.incomplete_secondary_violation_count
            < self.incomplete_evidence_count
        ):
            raise GroupedComparisonV2Error(
                "secondary violation count is smaller than incomplete units"
            )
        for name, numerator in (
            ("route_submission_coverage", self.route_submission_count),
            ("safe_route_coverage", self.safe_route_count),
            ("abstention_rate", self.abstention_count),
        ):
            value = _float_or_none(
                getattr(self, name),
                f"{self.arm_id}.{name}",
                unit_interval=True,
            )
            expected = _ratio(numerator, self.oracle_available_count)
            if not _same_number(value, expected):
                raise GroupedComparisonV2Error(
                    f"{self.arm_id}.{name} differs from counts"
                )
        regret_names = (
            "mean_safe_regret",
            "median_safe_regret",
            "worst_safe_regret",
            "mean_safe_temporal_switches",
            "mean_safe_frequency_switches",
        )
        for name in regret_names:
            _float_or_none(
                getattr(self, name),
                f"{self.arm_id}.{name}",
                nonnegative=True,
            )
        if self.safe_route_count == 0:
            if (
                any(getattr(self, name) is not None for name in regret_names)
                or self.worst_safe_regret_unit_sha256 is not None
            ):
                raise GroupedComparisonV2Error(
                    "zero-safe arm contains safe-route aggregates"
                )
        else:
            if any(getattr(self, name) is None for name in regret_names):
                raise GroupedComparisonV2Error(
                    "safe arm lacks safe-route aggregates"
                )
            _sha(
                self.worst_safe_regret_unit_sha256,
                "worst_safe_regret_unit_sha256",
            )
            if (
                self.mean_safe_regret > self.worst_safe_regret
                or self.median_safe_regret > self.worst_safe_regret
            ):
                raise GroupedComparisonV2Error(
                    "safe regret aggregates exceed the reported worst"
                )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": ARM_SUMMARY_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class PairwiseSummaryV2:
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

    def validate(self) -> None:
        count_names = (
            "unit_count",
            "oracle_available_count",
            "both_submitted_route_count",
            "both_safe_count",
            "d0_safe_r0_not_safe_count",
            "r0_safe_d0_not_safe_count",
            "both_catastrophic_count",
            "d0_catastrophic_r0_not_count",
            "r0_catastrophic_d0_not_count",
            "r0_lower_regret_count",
            "d0_lower_regret_count",
            "regret_tie_count",
        )
        for name in count_names:
            _int(getattr(self, name), f"pairwise.{name}")
        if self.oracle_available_count > self.unit_count:
            raise GroupedComparisonV2Error(
                "pairwise oracle count exceeds unit count"
            )
        if self.both_safe_count > self.both_submitted_route_count:
            raise GroupedComparisonV2Error(
                "both-safe count exceeds both-route count"
            )
        if self.both_submitted_route_count > self.oracle_available_count:
            raise GroupedComparisonV2Error(
                "both-route count exceeds oracle-available count"
            )
        if (
            self.r0_lower_regret_count
            + self.d0_lower_regret_count
            + self.regret_tie_count
            != self.both_safe_count
        ):
            raise GroupedComparisonV2Error(
                "paired regret classifications do not cover both-safe units"
            )
        for name, numerator in (
            (
                "both_submitted_route_coverage",
                self.both_submitted_route_count,
            ),
            ("both_safe_coverage", self.both_safe_count),
        ):
            value = _float_or_none(
                getattr(self, name),
                f"pairwise.{name}",
                unit_interval=True,
            )
            expected = _ratio(numerator, self.oracle_available_count)
            if not _same_number(value, expected):
                raise GroupedComparisonV2Error(
                    f"pairwise.{name} differs from counts"
                )
        aggregate_names = (
            "mean_r0_minus_d0_regret",
            "median_r0_minus_d0_regret",
            "min_r0_minus_d0_regret",
            "max_r0_minus_d0_regret",
            "mean_r0_minus_d0_temporal_switches",
            "mean_r0_minus_d0_frequency_switches",
        )
        for name in aggregate_names:
            _float_or_none(getattr(self, name), f"pairwise.{name}")
        if self.both_safe_count == 0:
            if any(
                getattr(self, name) is not None for name in aggregate_names
            ):
                raise GroupedComparisonV2Error(
                    "zero both-safe units contain paired aggregates"
                )
        else:
            if any(
                getattr(self, name) is None for name in aggregate_names
            ):
                raise GroupedComparisonV2Error(
                    "both-safe units lack paired aggregates"
                )
            if not (
                self.min_r0_minus_d0_regret
                <= self.median_r0_minus_d0_regret
                <= self.max_r0_minus_d0_regret
            ):
                raise GroupedComparisonV2Error(
                    "paired regret extrema/median are incoherent"
                )
            if not (
                self.min_r0_minus_d0_regret
                <= self.mean_r0_minus_d0_regret
                <= self.max_r0_minus_d0_regret
            ):
                raise GroupedComparisonV2Error(
                    "paired regret mean lies outside extrema"
                )
        if type(self.status_cross_tab) is not tuple:
            raise GroupedComparisonV2Error(
                "status_cross_tab must be a tuple"
            )
        if self.status_cross_tab != tuple(sorted(self.status_cross_tab)):
            raise GroupedComparisonV2Error(
                "status_cross_tab must be canonical"
            )
        if len(set(self.status_cross_tab)) != len(self.status_cross_tab):
            raise GroupedComparisonV2Error(
                "status_cross_tab contains duplicates"
            )
        total = 0
        unavailable = 0
        for row in self.status_cross_tab:
            if type(row) is not tuple or len(row) != 3:
                raise GroupedComparisonV2Error(
                    "status_cross_tab row must be a 3-tuple"
                )
            left, right, count = row
            if left not in ALL_STATUSES or right not in ALL_STATUSES:
                raise GroupedComparisonV2Error(
                    "status_cross_tab contains an unknown status"
                )
            _int(count, "status_cross_tab count", positive=True)
            if (left == "ORACLE_UNAVAILABLE") != (
                right == "ORACLE_UNAVAILABLE"
            ):
                raise GroupedComparisonV2Error(
                    "oracle availability differs inside status cross-tab"
                )
            if left == right == "ORACLE_UNAVAILABLE":
                unavailable += count
            total += count
        if total != self.unit_count:
            raise GroupedComparisonV2Error(
                "status_cross_tab does not cover all units"
            )
        if self.unit_count - unavailable != self.oracle_available_count:
            raise GroupedComparisonV2Error(
                "status_cross_tab oracle count differs"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
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
    arm_id: str,
    records: Sequence[WorkArmEvaluationV2],
) -> ArmSummaryV2:
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
    result = ArmSummaryV2(
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
        mean_safe_temporal_switches=(
            fmean(temporal) if temporal else None
        ),
        mean_safe_frequency_switches=(
            fmean(frequency) if frequency else None
        ),
    )
    result.validate()
    return result


def _pairwise_summary(
    records: Sequence[WorkArmEvaluationV2],
    prereg: GroupedComparisonPreregistrationV1,
) -> PairwiseSummaryV2:
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
    result = PairwiseSummaryV2(
        unit_count=len(pairs),
        oracle_available_count=oracle_available,
        both_submitted_route_count=both_routes,
        both_safe_count=len(paired_safe),
        both_submitted_route_coverage=_ratio(
            both_routes, oracle_available
        ),
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
        r0_lower_regret_count=sum(
            value < -tolerance for value in regret_deltas
        ),
        d0_lower_regret_count=sum(
            value > tolerance for value in regret_deltas
        ),
        regret_tie_count=sum(
            abs(value) <= tolerance for value in regret_deltas
        ),
        mean_r0_minus_d0_regret=(
            fmean(regret_deltas) if regret_deltas else None
        ),
        median_r0_minus_d0_regret=(
            median(regret_deltas) if regret_deltas else None
        ),
        min_r0_minus_d0_regret=(
            min(regret_deltas) if regret_deltas else None
        ),
        max_r0_minus_d0_regret=(
            max(regret_deltas) if regret_deltas else None
        ),
        mean_r0_minus_d0_temporal_switches=(
            fmean(temporal_deltas) if temporal_deltas else None
        ),
        mean_r0_minus_d0_frequency_switches=(
            fmean(frequency_deltas) if frequency_deltas else None
        ),
        status_cross_tab=tuple(
            sorted(
                (left, right, count)
                for (left, right), count in cross.items()
            )
        ),
    )
    result.validate()
    return result


def _validate_matrix(
    records: Sequence[WorkArmEvaluationV2],
    prereg: GroupedComparisonPreregistrationV1,
) -> tuple[WorkArmEvaluationV2, ...]:
    prereg.validate()
    values = tuple(records)
    for row in values:
        row.validate(prereg)
    ordered = tuple(
        sorted(values, key=lambda row: (row.unit.sha256, row.arm_id))
    )
    if values != ordered:
        raise GroupedComparisonV2Error(
            "work evaluations must be in canonical unit/arm order"
        )
    actual = [(row.unit.sha256, row.arm_id) for row in values]
    if len(actual) != len(set(actual)):
        raise GroupedComparisonV2Error("duplicate unit/arm evaluation")
    expected = sorted(
        (unit.sha256, arm)
        for unit in prereg.expected_units
        for arm in ("D0", "R0")
    )
    if actual != expected:
        raise GroupedComparisonV2Error(
            "evaluation matrix does not exactly cover preregistered units"
        )
    by_key = {(row.unit.sha256, row.arm_id): row for row in values}
    for unit in prereg.expected_units:
        d0 = by_key[(unit.sha256, "D0")].evaluation
        r0 = by_key[(unit.sha256, "R0")].evaluation
        _validate_oracle_semantics(d0)
        _validate_oracle_semantics(r0)
        exact_fields = (
            "exact_evidence_sha256",
            "exact_panel_sha256",
            "exact_preflight_sha256",
            "policy_sha256",
            "exact_oracle_decision_sha256",
            "exact_oracle_objective",
        )
        drift = [
            name
            for name in exact_fields
            if getattr(d0, name) != getattr(r0, name)
        ]
        if drift:
            raise GroupedComparisonV2Error(
                f"D0/R0 exact evidence differs for {unit.sha256}: {drift}"
            )
        if (d0.status == "ORACLE_UNAVAILABLE") != (
            r0.status == "ORACLE_UNAVAILABLE"
        ):
            raise GroupedComparisonV2Error(
                "oracle availability differs between arms"
            )
    return values


@dataclass(frozen=True)
class GroupedComparisonReportV2:
    preregistration_sha256: str
    verifier_commit: str
    work_evaluations: tuple[WorkArmEvaluationV2, ...]
    d0_summary: ArmSummaryV2
    r0_summary: ArmSummaryV2
    pairwise_summary: PairwiseSummaryV2
    status: str = REPORT_STATUS

    @classmethod
    def build(
        cls,
        prereg: GroupedComparisonPreregistrationV1,
        records: Sequence[WorkArmEvaluationV2],
        *,
        verifier_commit: str,
    ) -> "GroupedComparisonReportV2":
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

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> None:
        prereg.validate()
        if self.status != REPORT_STATUS:
            raise GroupedComparisonV2Error(
                "report must not contain a promotion decision"
            )
        _sha(self.preregistration_sha256, "preregistration_sha256")
        if self.preregistration_sha256 != prereg.sha256:
            raise GroupedComparisonV2Error(
                "report names another preregistration"
            )
        _git(self.verifier_commit, "verifier_commit")
        records = _validate_matrix(self.work_evaluations, prereg)
        self.d0_summary.validate()
        self.r0_summary.validate()
        self.pairwise_summary.validate()
        if self.d0_summary != _arm_summary("D0", records):
            raise GroupedComparisonV2Error(
                "D0 summary differs from work evidence"
            )
        if self.r0_summary != _arm_summary("R0", records):
            raise GroupedComparisonV2Error(
                "R0 summary differs from work evidence"
            )
        if self.pairwise_summary != _pairwise_summary(records, prereg):
            raise GroupedComparisonV2Error(
                "pairwise summary differs from work evidence"
            )

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
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

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg))
