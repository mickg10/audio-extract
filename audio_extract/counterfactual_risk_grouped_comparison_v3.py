"""Transitive-provenance grouped D0/R0 comparison v3.

V2 bound a route submission to a held-out unit and repeated the frozen arm facts,
but those facts were still sibling declarations.  V3 makes the chain
content-bearing:

    preregistered arm
      -> exact held-out inference run manifest
      -> route output
      -> route artifact manifest
      -> ArmRouteSubmissionV1.route_evidence_sha256
      -> CompleteRouteEvaluationV1.submission_sha256
      -> grouped held-out report

The report remains descriptive and non-promoting.  It does not fit a model,
render audio, select a winner, or enter the production path.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .counterfactual_risk_complete_route_evaluation_v1 import (
    ArmRouteSubmissionV1,
    CompleteRouteEvaluationV1,
)
from .counterfactual_risk_grouped_comparison_v1 import (
    ROUTE_STATUSES,
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
)
from .counterfactual_risk_grouped_comparison_v2 import (
    ArmRouteProvenanceV2,
    ArmSummaryV2,
    GroupedComparisonReportV2,
    PairwiseSummaryV2,
    WorkArmEvaluationV2,
)

INFERENCE_RUN_SCHEMA = "audio-extract/d0-r0-arm-inference-run/v3"
ROUTE_OUTPUT_SCHEMA = "audio-extract/d0-r0-route-output/v3"
ROUTE_ARTIFACT_SCHEMA = "audio-extract/d0-r0-route-artifact-manifest/v3"
ROUTE_PROVENANCE_SCHEMA = "audio-extract/d0-r0-route-provenance/v3"
WORK_EVALUATION_SCHEMA = "audio-extract/d0-r0-work-evaluation/v3"
REPORT_SCHEMA = "audio-extract/d0-r0-grouped-comparison-report/v3"
REPORT_STATUS = "COMPLETE_NO_PROMOTION_DECISION"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupedComparisonV3Error(ValueError):
    """The grouped comparison is incomplete or not transitively attributable."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedComparisonV3Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedComparisonV3Error(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GroupedComparisonV3Error(
            f"comparison value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _expected_arm(prereg: GroupedComparisonPreregistrationV1, arm_id: str):
    if arm_id == "D0":
        return prereg.d0_arm
    if arm_id == "R0":
        return prereg.r0_arm
    raise GroupedComparisonV3Error(f"unknown comparison arm {arm_id!r}")


@dataclass(frozen=True)
class FrozenArmInferenceRunV3:
    """One immutable arm execution on one exact held-out unit."""

    preregistration_sha256: str
    unit_sha256: str
    group_family_sha256: str
    query_condition_sha256: str
    query_quality_stratum_sha256: str
    arm_id: str
    arm_run_sha256: str
    arm_artifact_manifest_sha256: str
    compute_budget_sha256: str
    checkpoint_rule_sha256: str
    inference_policy_sha256: str
    dataset_sha256: str
    split_sha256: str
    test_subset_sha256: str
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    inference_input_manifest_sha256: str
    runner_bundle_sha256: str
    source_commit: str
    runner_commit: str
    status: str = "verified"

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> None:
        prereg.validate()
        unit.validate()
        if self.status != "verified":
            raise GroupedComparisonV3Error(
                "arm inference-run status must be verified"
            )
        for name in (
            "preregistration_sha256",
            "unit_sha256",
            "group_family_sha256",
            "query_condition_sha256",
            "query_quality_stratum_sha256",
            "arm_run_sha256",
            "arm_artifact_manifest_sha256",
            "compute_budget_sha256",
            "checkpoint_rule_sha256",
            "inference_policy_sha256",
            "dataset_sha256",
            "split_sha256",
            "test_subset_sha256",
            "candidate_panel_sha256",
            "feature_contract_sha256",
            "metric_contract_sha256",
            "inference_input_manifest_sha256",
            "runner_bundle_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.source_commit, "inference-run source_commit")
        _git(self.runner_commit, "inference-run runner_commit")
        if self.preregistration_sha256 != prereg.sha256:
            raise GroupedComparisonV3Error(
                "arm inference run names another preregistration"
            )
        expected_unit = {
            "unit_sha256": unit.sha256,
            "group_family_sha256": unit.group_family_sha256,
            "query_condition_sha256": unit.query_condition_sha256,
            "query_quality_stratum_sha256": (
                unit.query_quality_stratum_sha256
            ),
        }
        drift = [
            name for name, expected in expected_unit.items()
            if getattr(self, name) != expected
        ]
        if drift:
            raise GroupedComparisonV3Error(
                f"arm inference run names another held-out unit: {drift}"
            )
        arm = _expected_arm(prereg, self.arm_id)
        arm.validate()
        expected_arm = {
            "arm_run_sha256": arm.sha256,
            "arm_artifact_manifest_sha256": arm.artifact_manifest_sha256,
            "compute_budget_sha256": arm.compute_budget_sha256,
            "checkpoint_rule_sha256": arm.checkpoint_rule_sha256,
            "inference_policy_sha256": arm.inference_policy_sha256,
        }
        arm_drift = [
            name for name, expected in expected_arm.items()
            if getattr(self, name) != expected
        ]
        if arm_drift:
            raise GroupedComparisonV3Error(
                f"arm inference run differs from frozen arm: {arm_drift}"
            )
        expected_context = {
            "dataset_sha256": prereg.dataset_sha256,
            "split_sha256": prereg.split_sha256,
            "test_subset_sha256": prereg.test_subset_sha256,
            "candidate_panel_sha256": prereg.candidate_panel_sha256,
            "feature_contract_sha256": prereg.feature_contract_sha256,
            "metric_contract_sha256": prereg.metric_contract_sha256,
        }
        context_drift = [
            name for name, expected in expected_context.items()
            if getattr(self, name) != expected
        ]
        if context_drift:
            raise GroupedComparisonV3Error(
                f"arm inference run context differs: {context_drift}"
            )
        if self.source_commit != prereg.source_commit:
            raise GroupedComparisonV3Error(
                "arm inference run source commit differs"
            )

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> dict[str, Any]:
        self.validate(prereg, unit)
        return {"schema": INFERENCE_RUN_SCHEMA, **self.__dict__}

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg, unit))


@dataclass(frozen=True)
class RouteOutputV3:
    """The route/abstention payload before it is wrapped in lineage."""

    unit_sha256: str
    arm_id: str
    source_status: str
    status: str
    labels_shape: tuple[int, int] | None
    labels_sha256: str | None
    reason: str

    @classmethod
    def from_submission(
        cls,
        unit: HeldOutUnitV1,
        submission: ArmRouteSubmissionV1,
    ) -> RouteOutputV3:
        submission.validate()
        labels = None if submission.labels is None else np.asarray(
            submission.labels
        )
        return cls(
            unit_sha256=unit.sha256,
            arm_id=submission.arm_id,
            source_status=submission.source_status,
            status=submission.status,
            labels_shape=(
                None if labels is None else tuple(int(x) for x in labels.shape)
            ),
            labels_sha256=submission.labels_sha256,
            reason=submission.reason,
        )

    def validate(
        self,
        unit: HeldOutUnitV1,
        arm_id: str,
    ) -> None:
        unit.validate()
        _sha(self.unit_sha256, "route-output unit_sha256")
        if self.unit_sha256 != unit.sha256:
            raise GroupedComparisonV3Error(
                "route output names another held-out unit"
            )
        if self.arm_id != arm_id or arm_id not in {"D0", "R0"}:
            raise GroupedComparisonV3Error(
                "route output names another comparison arm"
            )
        if not isinstance(self.source_status, str) or not self.source_status:
            raise GroupedComparisonV3Error(
                "route output source status must be non-empty"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise GroupedComparisonV3Error(
                "route output reason must be non-empty"
            )
        if self.status == "ROUTE":
            if (
                type(self.labels_shape) is not tuple
                or len(self.labels_shape) != 2
                or any(type(value) is not int or value < 1 for value in self.labels_shape)
            ):
                raise GroupedComparisonV3Error(
                    "ROUTE output requires a positive two-dimensional label shape"
                )
            _sha(self.labels_sha256, "route-output labels_sha256")
        elif self.status == "ABSTAIN":
            if self.labels_shape is not None or self.labels_sha256 is not None:
                raise GroupedComparisonV3Error(
                    "ABSTAIN output must not expose labels"
                )
        else:
            raise GroupedComparisonV3Error(
                "route output status must be ROUTE or ABSTAIN"
            )

    def matches_submission(
        self,
        unit: HeldOutUnitV1,
        submission: ArmRouteSubmissionV1,
    ) -> None:
        submission.validate()
        self.validate(unit, submission.arm_id)
        expected = RouteOutputV3.from_submission(unit, submission)
        if self != expected:
            raise GroupedComparisonV3Error(
                "route artifact output differs from route submission"
            )

    def identity_dict(
        self,
        unit: HeldOutUnitV1,
        arm_id: str,
    ) -> dict[str, Any]:
        self.validate(unit, arm_id)
        return {
            "schema": ROUTE_OUTPUT_SCHEMA,
            "unit_sha256": self.unit_sha256,
            "arm_id": self.arm_id,
            "source_status": self.source_status,
            "status": self.status,
            "labels_shape": (
                None if self.labels_shape is None else list(self.labels_shape)
            ),
            "labels_sha256": self.labels_sha256,
            "reason": self.reason,
        }

    def sha256(self, unit: HeldOutUnitV1, arm_id: str) -> str:
        return _mapping_sha(self.identity_dict(unit, arm_id))


@dataclass(frozen=True)
class RouteArtifactManifestV3:
    """Content-bearing route artifact with verified arm/run parents."""

    inference_run: FrozenArmInferenceRunV3
    route_output: RouteOutputV3
    verifier_commit: str
    status: str = "verified"

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> None:
        if self.status != "verified":
            raise GroupedComparisonV3Error(
                "route artifact manifest status must be verified"
            )
        _git(self.verifier_commit, "route artifact verifier_commit")
        self.inference_run.validate(prereg, unit)
        self.route_output.validate(unit, self.inference_run.arm_id)

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> dict[str, Any]:
        self.validate(prereg, unit)
        return {
            "schema": ROUTE_ARTIFACT_SCHEMA,
            "status": self.status,
            "verifier_commit": self.verifier_commit,
            "inference_run": self.inference_run.identity_dict(prereg, unit),
            "inference_run_sha256": self.inference_run.sha256(prereg, unit),
            "route_output": self.route_output.identity_dict(
                unit, self.inference_run.arm_id
            ),
            "route_output_sha256": self.route_output.sha256(
                unit, self.inference_run.arm_id
            ),
        }

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg, unit))


@dataclass(frozen=True)
class ArmRouteProvenanceV3:
    route_artifact: RouteArtifactManifestV3
    submission: ArmRouteSubmissionV1

    @property
    def arm_id(self) -> str:
        return self.route_artifact.inference_run.arm_id

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> None:
        self.route_artifact.validate(prereg, unit)
        self.submission.validate()
        self.route_artifact.route_output.matches_submission(
            unit, self.submission
        )
        if self.submission.arm_id != self.arm_id:
            raise GroupedComparisonV3Error(
                "route submission arm differs from artifact lineage"
            )
        expected = self.route_artifact.sha256(prereg, unit)
        if self.submission.route_evidence_sha256 != expected:
            raise GroupedComparisonV3Error(
                "route submission is not bound to the route artifact manifest"
            )

    def to_v2(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> ArmRouteProvenanceV2:
        self.validate(prereg, unit)
        run = self.route_artifact.inference_run
        return ArmRouteProvenanceV2(
            unit_sha256=run.unit_sha256,
            group_family_sha256=run.group_family_sha256,
            query_condition_sha256=run.query_condition_sha256,
            query_quality_stratum_sha256=run.query_quality_stratum_sha256,
            arm_id=run.arm_id,
            arm_run_sha256=run.arm_run_sha256,
            artifact_manifest_sha256=run.arm_artifact_manifest_sha256,
            compute_budget_sha256=run.compute_budget_sha256,
            checkpoint_rule_sha256=run.checkpoint_rule_sha256,
            inference_policy_sha256=run.inference_policy_sha256,
            source_commit=run.source_commit,
            route_artifact_sha256=self.route_artifact.sha256(prereg, unit),
            verifier_commit=self.route_artifact.verifier_commit,
            submission=self.submission,
        )

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> dict[str, Any]:
        self.validate(prereg, unit)
        return {
            "schema": ROUTE_PROVENANCE_SCHEMA,
            "route_artifact": self.route_artifact.identity_dict(prereg, unit),
            "route_artifact_sha256": self.route_artifact.sha256(prereg, unit),
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
class WorkArmEvaluationV3:
    unit: HeldOutUnitV1
    route_provenance: ArmRouteProvenanceV3
    evaluation: CompleteRouteEvaluationV1

    @property
    def arm_id(self) -> str:
        return self.route_provenance.arm_id

    def _validate_status_match(self) -> None:
        evaluation_status = self.evaluation.status
        submission_status = self.route_provenance.submission.status
        if evaluation_status == "ORACLE_UNAVAILABLE":
            return
        if evaluation_status == "ABSTAIN":
            expected = "ABSTAIN"
        elif evaluation_status in ROUTE_STATUSES:
            expected = "ROUTE"
        else:
            raise GroupedComparisonV3Error(
                f"unknown complete-route status {evaluation_status!r}"
            )
        if submission_status != expected:
            raise GroupedComparisonV3Error(
                "complete-route evaluation status contradicts the bound submission"
            )

    def to_v2(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> WorkArmEvaluationV2:
        self.unit.validate()
        self.route_provenance.validate(prereg, self.unit)
        run = self.route_provenance.route_artifact.inference_run
        return WorkArmEvaluationV2(
            unit=self.unit,
            route_provenance=self.route_provenance.to_v2(prereg, self.unit),
            dataset_sha256=run.dataset_sha256,
            split_sha256=run.split_sha256,
            test_subset_sha256=run.test_subset_sha256,
            candidate_panel_sha256=run.candidate_panel_sha256,
            feature_contract_sha256=run.feature_contract_sha256,
            metric_contract_sha256=run.metric_contract_sha256,
            preregistration_sha256=run.preregistration_sha256,
            evaluation=self.evaluation,
        )

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> None:
        self.route_provenance.validate(prereg, self.unit)
        self.evaluation.validate()
        self._validate_status_match()
        if self.evaluation.arm_id != self.arm_id:
            raise GroupedComparisonV3Error(
                "complete-route evaluation arm differs from route lineage"
            )
        if self.evaluation.submission_sha256 != (
            self.route_provenance.submission.sha256
        ):
            raise GroupedComparisonV3Error(
                "complete-route evaluation names another submission"
            )
        self.to_v2(prereg).validate(prereg)

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> dict[str, Any]:
        self.validate(prereg)
        return {
            "schema": WORK_EVALUATION_SCHEMA,
            "unit": self.unit.identity_dict(),
            "unit_sha256": self.unit.sha256,
            "route_provenance": self.route_provenance.identity_dict(
                prereg, self.unit
            ),
            "route_provenance_sha256": self.route_provenance.sha256(
                prereg, self.unit
            ),
            "complete_route_evaluation": self.evaluation.identity_dict(),
            "complete_route_evaluation_sha256": self.evaluation.sha256,
        }


@dataclass(frozen=True)
class GroupedComparisonReportV3:
    preregistration_sha256: str
    verifier_commit: str
    work_evaluations: tuple[WorkArmEvaluationV3, ...]
    d0_summary: ArmSummaryV2
    r0_summary: ArmSummaryV2
    pairwise_summary: PairwiseSummaryV2
    status: str = REPORT_STATUS

    @staticmethod
    def _validate_unique_lineage(
        records: Sequence[WorkArmEvaluationV3],
        prereg: GroupedComparisonPreregistrationV1,
    ) -> None:
        collections = {
            "inference-run": [
                row.route_provenance.route_artifact.inference_run.sha256(
                    prereg, row.unit
                )
                for row in records
            ],
            "route-output": [
                row.route_provenance.route_artifact.route_output.sha256(
                    row.unit, row.arm_id
                )
                for row in records
            ],
            "route-artifact": [
                row.route_provenance.route_artifact.sha256(prereg, row.unit)
                for row in records
            ],
            "submission": [
                row.route_provenance.submission.sha256 for row in records
            ],
        }
        for name, values in collections.items():
            if len(values) != len(set(values)):
                raise GroupedComparisonV3Error(
                    f"comparison reuses one {name} across unit/arm cells"
                )

    @classmethod
    def build(
        cls,
        prereg: GroupedComparisonPreregistrationV1,
        records: Sequence[WorkArmEvaluationV3],
        *,
        verifier_commit: str,
    ) -> GroupedComparisonReportV3:
        ordered = tuple(
            sorted(records, key=lambda row: (row.unit.sha256, row.arm_id))
        )
        for row in ordered:
            row.validate(prereg)
        cls._validate_unique_lineage(ordered, prereg)
        v2 = GroupedComparisonReportV2.build(
            prereg,
            [row.to_v2(prereg) for row in ordered],
            verifier_commit=verifier_commit,
        )
        result = cls(
            preregistration_sha256=prereg.sha256,
            verifier_commit=verifier_commit,
            work_evaluations=ordered,
            d0_summary=v2.d0_summary,
            r0_summary=v2.r0_summary,
            pairwise_summary=v2.pairwise_summary,
        )
        result.validate(prereg)
        return result

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> None:
        prereg.validate()
        if self.status != REPORT_STATUS:
            raise GroupedComparisonV3Error(
                "v3 report must not contain a promotion decision"
            )
        _sha(self.preregistration_sha256, "preregistration_sha256")
        _git(self.verifier_commit, "verifier_commit")
        if self.preregistration_sha256 != prereg.sha256:
            raise GroupedComparisonV3Error(
                "v3 report names another preregistration"
            )
        ordered = tuple(
            sorted(
                self.work_evaluations,
                key=lambda row: (row.unit.sha256, row.arm_id),
            )
        )
        if self.work_evaluations != ordered:
            raise GroupedComparisonV3Error(
                "v3 work evaluations are not in canonical order"
            )
        for row in ordered:
            row.validate(prereg)
        self._validate_unique_lineage(ordered, prereg)
        v2 = GroupedComparisonReportV2.build(
            prereg,
            [row.to_v2(prereg) for row in ordered],
            verifier_commit=self.verifier_commit,
        )
        self.d0_summary.validate()
        self.r0_summary.validate()
        self.pairwise_summary.validate()
        if self.d0_summary != v2.d0_summary:
            raise GroupedComparisonV3Error(
                "D0 summary differs from transitive work evidence"
            )
        if self.r0_summary != v2.r0_summary:
            raise GroupedComparisonV3Error(
                "R0 summary differs from transitive work evidence"
            )
        if self.pairwise_summary != v2.pairwise_summary:
            raise GroupedComparisonV3Error(
                "pairwise summary differs from transitive work evidence"
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
