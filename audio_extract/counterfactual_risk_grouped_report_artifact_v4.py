"""Immutable paired serialization artifact for grouped D0/R0 reports v4.

The in-memory v3 report contains ``ArmRouteSubmissionV1`` objects whose labels
are NumPy arrays. A caller retaining an array reference can mutate it after the
report is built, changing the submission SHA and destabilizing the object graph.
V4 is the public artifact boundary: it snapshots the fully validated v3 report
into canonical UTF-8 bytes and groups D0/R0 evidence by held-out unit.

The frozen validator is intentionally independent of caller-owned Python
objects. It verifies the complete closed payload, every nested semantic hash,
outer/inner unit equality, arm/status/oracle/violation rules, all aggregate
summaries, and reconstructs the embedded v3 report identity.

This artifact is descriptive and contains no promotion decision.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from statistics import fmean, median
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

from .counterfactual_risk_grouped_comparison_v1 import (
    ALL_STATUSES,
    ROUTE_STATUSES,
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
)
from .counterfactual_risk_grouped_comparison_v2 import (
    ArmSummaryV2,
    PairwiseSummaryV2,
)
from .counterfactual_risk_grouped_comparison_v3 import GroupedComparisonReportV3

REPORT_SCHEMA = "audio-extract/d0-r0-grouped-comparison-report/v4"
SOURCE_REPORT_SCHEMA = "audio-extract/d0-r0-grouped-comparison-report/v3"
PAIRED_UNIT_SCHEMA = "audio-extract/d0-r0-paired-work-evaluation/v4"
STATUS = "COMPLETE_NO_PROMOTION_DECISION"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupedReportArtifactV4Error(ValueError):
    """The immutable grouped-report artifact is malformed or source-drifted."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedReportArtifactV4Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedReportArtifactV4Error(
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
        raise GroupedReportArtifactV4Error(
            f"report value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GroupedReportArtifactV4Error(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise GroupedReportArtifactV4Error(f"{name} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise GroupedReportArtifactV4Error(
            f"{name} keys differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _nonempty_text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise GroupedReportArtifactV4Error(
            f"{name} must be a non-empty trim-stable string"
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise GroupedReportArtifactV4Error(
            f"{name} must be a non-negative integer"
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise GroupedReportArtifactV4Error(
            f"{name} must be a finite non-negative float"
        )
    return value


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_float(value, name)


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name)


def _same_number(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    return math.isclose(
        first,
        second,
        rel_tol=0.0,
        abs_tol=8.0 * math.ulp(max(1.0, abs(second))),
    )


def _unit_from_dict(value: Any) -> HeldOutUnitV1:
    row = _mapping(value, "paired unit identity")
    expected = {
        "schema",
        "work_id",
        "recording_session_id",
        "target_singer_id",
        "source_family_sha256",
        "query_condition_sha256",
        "group_family_sha256",
        "query_quality_stratum_sha256",
        "exact_evidence_sha256",
        "exact_panel_sha256",
        "exact_preflight_sha256",
        "exact_oracle_decision_sha256",
    }
    _exact_keys(row, expected, "paired unit identity")
    if row["schema"] != "audio-extract/counterfactual-held-out-unit/v1":
        raise GroupedReportArtifactV4Error("unknown held-out unit schema")
    unit = HeldOutUnitV1(
        work_id=row["work_id"],
        recording_session_id=row["recording_session_id"],
        target_singer_id=row["target_singer_id"],
        source_family_sha256=row["source_family_sha256"],
        query_condition_sha256=row["query_condition_sha256"],
        query_quality_stratum_sha256=row[
            "query_quality_stratum_sha256"
        ],
        exact_evidence_sha256=row["exact_evidence_sha256"],
        exact_panel_sha256=row["exact_panel_sha256"],
        exact_preflight_sha256=row["exact_preflight_sha256"],
        exact_oracle_decision_sha256=row[
            "exact_oracle_decision_sha256"
        ],
    )
    unit.validate()
    if unit.identity_dict() != dict(row):
        raise GroupedReportArtifactV4Error(
            "held-out unit identity is not canonical"
        )
    return unit


def _validate_labels(
    status: str,
    shape: Any,
    labels_sha256: Any,
    name: str,
) -> tuple[int, int] | None:
    if status == "ABSTAIN":
        if shape is not None or labels_sha256 is not None:
            raise GroupedReportArtifactV4Error(
                f"{name} abstention unexpectedly exposes labels"
            )
        return None
    if status != "ROUTE":
        raise GroupedReportArtifactV4Error(f"unknown {name} status {status!r}")
    values = _list(shape, f"{name}.labels_shape")
    if len(values) != 2 or any(type(value) is not int or value < 1 for value in values):
        raise GroupedReportArtifactV4Error(
            f"{name} route labels must have a positive two-dimensional shape"
        )
    _sha(labels_sha256, f"{name}.labels_sha256")
    return int(values[0]), int(values[1])


def _validate_violation(value: Any, *, critical: bool) -> tuple[Any, ...]:
    row = _mapping(value, "selected-risk violation")
    expected = {
        "schema",
        "time_index",
        "band_index",
        "candidate_index",
        "metric_name",
        "kind",
        "threshold",
        "value",
    }
    _exact_keys(row, expected, "selected-risk violation")
    if row["schema"] != "audio-extract/counterfactual-route-violation/v1":
        raise GroupedReportArtifactV4Error("unknown selected-risk violation schema")
    for name in ("time_index", "band_index", "candidate_index"):
        _nonnegative_int(row[name], f"violation.{name}")
    _nonempty_text(row["metric_name"], "violation.metric_name")
    kind = row["kind"]
    if critical:
        if kind not in {"critical_missing", "critical_above_threshold"}:
            raise GroupedReportArtifactV4Error(
                "critical list contains a noncritical violation"
            )
    elif kind != "required_secondary_missing":
        raise GroupedReportArtifactV4Error(
            "secondary list contains a nonsecondary violation"
        )
    threshold = row["threshold"]
    measured = row["value"]
    if kind == "critical_above_threshold":
        threshold_value = _nonnegative_float(
            threshold, "above-threshold threshold"
        )
        value = _nonnegative_float(measured, "above-threshold value")
        if threshold_value <= 0 or value <= threshold_value:
            raise GroupedReportArtifactV4Error(
                "above-threshold violation does not exceed a positive threshold"
            )
    else:
        if measured is not None:
            raise GroupedReportArtifactV4Error(
                "missing-evidence violation invents a measured value"
            )
        if threshold is not None:
            _nonnegative_float(threshold, "missing-evidence threshold")
    return (
        row["time_index"],
        row["band_index"],
        row["candidate_index"],
        row["metric_name"],
        kind,
        threshold,
        measured,
    )


def _validate_evaluation(
    value: Any,
    *,
    unit: HeldOutUnitV1,
    arm_id: str,
    submission_sha256: str,
    submission_status: str,
) -> Mapping[str, Any]:
    row = _mapping(value, "complete-route evaluation")
    expected = {
        "schema",
        "arm_id",
        "exact_evidence_sha256",
        "exact_panel_sha256",
        "exact_preflight_sha256",
        "policy_sha256",
        "exact_oracle_decision_sha256",
        "submission_sha256",
        "status",
        "critical_false_safe_violations",
        "incomplete_secondary_violations",
        "exact_oracle_objective",
        "submitted_exact_objective",
        "selection_regret",
        "temporal_switches",
        "frequency_switches",
        "reason",
    }
    _exact_keys(row, expected, "complete-route evaluation")
    if row["schema"] != "audio-extract/counterfactual-route-evaluation/v1":
        raise GroupedReportArtifactV4Error("unknown complete-route evaluation schema")
    if row["arm_id"] != arm_id:
        raise GroupedReportArtifactV4Error(
            "complete-route evaluation arm differs from paired branch"
        )
    expected_exact = {
        "exact_evidence_sha256": unit.exact_evidence_sha256,
        "exact_panel_sha256": unit.exact_panel_sha256,
        "exact_preflight_sha256": unit.exact_preflight_sha256,
        "exact_oracle_decision_sha256": unit.exact_oracle_decision_sha256,
        "submission_sha256": submission_sha256,
    }
    drift = [
        name
        for name, expected_value in expected_exact.items()
        if row[name] != expected_value
    ]
    if drift:
        raise GroupedReportArtifactV4Error(
            f"complete-route evaluation lineage differs: {drift}"
        )
    _sha(row["policy_sha256"], "evaluation.policy_sha256")
    _nonempty_text(row["reason"], "evaluation.reason")
    critical_rows = _list(
        row["critical_false_safe_violations"],
        "critical_false_safe_violations",
    )
    incomplete_rows = _list(
        row["incomplete_secondary_violations"],
        "incomplete_secondary_violations",
    )
    critical = [
        _validate_violation(item, critical=True) for item in critical_rows
    ]
    incomplete = [
        _validate_violation(item, critical=False) for item in incomplete_rows
    ]
    if critical != sorted(set(critical)) or incomplete != sorted(set(incomplete)):
        raise GroupedReportArtifactV4Error(
            "selected-risk violations are not unique and canonical"
        )
    status = row["status"]
    if status not in ALL_STATUSES:
        raise GroupedReportArtifactV4Error(
            f"unknown complete-route status {status!r}"
        )
    exact = _optional_float(
        row["exact_oracle_objective"], "exact_oracle_objective"
    )
    submitted = _optional_float(
        row["submitted_exact_objective"], "submitted_exact_objective"
    )
    regret = _optional_float(row["selection_regret"], "selection_regret")
    temporal = _optional_int(row["temporal_switches"], "temporal_switches")
    frequency = _optional_int(row["frequency_switches"], "frequency_switches")
    if status == "ORACLE_UNAVAILABLE":
        if any(value is not None for value in (exact, submitted, regret, temporal, frequency)):
            raise GroupedReportArtifactV4Error(
                "oracle-unavailable evaluation contains route/objective facts"
            )
        if critical or incomplete:
            raise GroupedReportArtifactV4Error(
                "oracle-unavailable evaluation contains violations"
            )
        return row
    if exact is None:
        raise GroupedReportArtifactV4Error(
            "oracle-available evaluation lacks its exact objective"
        )
    if status in ROUTE_STATUSES:
        if submission_status != "ROUTE":
            raise GroupedReportArtifactV4Error(
                "route evaluation is not backed by route lineage"
            )
        if temporal is None or frequency is None:
            raise GroupedReportArtifactV4Error(
                "submitted route lacks switch counts"
            )
    elif status == "ABSTAIN":
        if submission_status != "ABSTAIN":
            raise GroupedReportArtifactV4Error(
                "abstention evaluation is not backed by abstention lineage"
            )
        if any(value is not None for value in (submitted, regret, temporal, frequency)):
            raise GroupedReportArtifactV4Error(
                "abstention evaluation contains route result facts"
            )
        if critical or incomplete:
            raise GroupedReportArtifactV4Error(
                "abstention evaluation contains violations"
            )
        return row
    if status == "ROUTE_SAFE":
        if critical or incomplete or submitted is None or regret is None:
            raise GroupedReportArtifactV4Error(
                "safe route has incomplete or contradictory result facts"
            )
        expected_regret = submitted - exact
        if expected_regret < -8.0 * math.ulp(max(1.0, abs(exact))):
            raise GroupedReportArtifactV4Error(
                "safe submitted objective is below the exact optimum"
            )
        if not _same_number(regret, max(0.0, expected_regret)):
            raise GroupedReportArtifactV4Error(
                "safe route selection regret differs from objectives"
            )
    elif status == "ROUTE_CATASTROPHIC_FALSE_SAFE":
        if not critical or incomplete or submitted is not None or regret is not None:
            raise GroupedReportArtifactV4Error(
                "catastrophic route has inconsistent violation/objective facts"
            )
    elif status == "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE":
        if critical or not incomplete or submitted is not None or regret is not None:
            raise GroupedReportArtifactV4Error(
                "incomplete route has inconsistent violation/objective facts"
            )
    return row


def _validate_work_branch(
    value: Any,
    *,
    unit: HeldOutUnitV1,
    arm_id: str,
    preregistration_sha256: str,
) -> Mapping[str, Any]:
    row = _mapping(value, f"{arm_id} work evaluation")
    expected = {
        "schema",
        "unit",
        "unit_sha256",
        "route_provenance",
        "route_provenance_sha256",
        "complete_route_evaluation",
        "complete_route_evaluation_sha256",
    }
    _exact_keys(row, expected, f"{arm_id} work evaluation")
    if row["schema"] != "audio-extract/d0-r0-work-evaluation/v3":
        raise GroupedReportArtifactV4Error("unknown v3 work-evaluation schema")
    if row["unit"] != unit.identity_dict() or row["unit_sha256"] != unit.sha256:
        raise GroupedReportArtifactV4Error(
            f"{arm_id} branch is not bound to the outer held-out unit"
        )
    provenance = _mapping(row["route_provenance"], "route provenance")
    _exact_keys(
        provenance,
        {
            "schema",
            "route_artifact",
            "route_artifact_sha256",
            "submission",
            "submission_sha256",
        },
        "route provenance",
    )
    if provenance["schema"] != "audio-extract/d0-r0-route-provenance/v3":
        raise GroupedReportArtifactV4Error("unknown route-provenance schema")
    artifact = _mapping(provenance["route_artifact"], "route artifact")
    _exact_keys(
        artifact,
        {
            "schema",
            "status",
            "verifier_commit",
            "inference_run",
            "inference_run_sha256",
            "route_output",
            "route_output_sha256",
        },
        "route artifact",
    )
    if artifact["schema"] != "audio-extract/d0-r0-route-artifact-manifest/v3" or artifact["status"] != "verified":
        raise GroupedReportArtifactV4Error("route artifact is not verified v3")
    _git(artifact["verifier_commit"], "route artifact verifier_commit")
    run = _mapping(artifact["inference_run"], "inference run")
    run_keys = {
        "schema",
        "preregistration_sha256",
        "unit_sha256",
        "group_family_sha256",
        "query_condition_sha256",
        "query_quality_stratum_sha256",
        "arm_id",
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
        "source_commit",
        "runner_commit",
        "status",
    }
    _exact_keys(run, run_keys, "inference run")
    if run["schema"] != "audio-extract/d0-r0-arm-inference-run/v3" or run["status"] != "verified":
        raise GroupedReportArtifactV4Error("inference run is not verified v3")
    expected_run = {
        "preregistration_sha256": preregistration_sha256,
        "unit_sha256": unit.sha256,
        "group_family_sha256": unit.group_family_sha256,
        "query_condition_sha256": unit.query_condition_sha256,
        "query_quality_stratum_sha256": unit.query_quality_stratum_sha256,
        "arm_id": arm_id,
    }
    drift = [name for name, expected_value in expected_run.items() if run[name] != expected_value]
    if drift:
        raise GroupedReportArtifactV4Error(
            f"{arm_id} inference run differs from paired context: {drift}"
        )
    for name in (
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
        _sha(run[name], f"inference run {name}")
    _git(run["source_commit"], "inference run source_commit")
    _git(run["runner_commit"], "inference run runner_commit")
    if artifact["inference_run_sha256"] != _mapping_sha(run):
        raise GroupedReportArtifactV4Error("inference-run semantic SHA mismatch")
    output = _mapping(artifact["route_output"], "route output")
    _exact_keys(
        output,
        {
            "schema",
            "unit_sha256",
            "arm_id",
            "source_status",
            "status",
            "labels_shape",
            "labels_sha256",
            "reason",
        },
        "route output",
    )
    if output["schema"] != "audio-extract/d0-r0-route-output/v3":
        raise GroupedReportArtifactV4Error("unknown route-output schema")
    if output["unit_sha256"] != unit.sha256 or output["arm_id"] != arm_id:
        raise GroupedReportArtifactV4Error(
            "route output differs from paired unit/arm"
        )
    _nonempty_text(output["source_status"], "route output source_status")
    _nonempty_text(output["reason"], "route output reason")
    _validate_labels(
        output["status"],
        output["labels_shape"],
        output["labels_sha256"],
        "route output",
    )
    if artifact["route_output_sha256"] != _mapping_sha(output):
        raise GroupedReportArtifactV4Error("route-output semantic SHA mismatch")
    if provenance["route_artifact_sha256"] != _mapping_sha(artifact):
        raise GroupedReportArtifactV4Error("route-artifact semantic SHA mismatch")
    submission = _mapping(provenance["submission"], "route submission")
    _exact_keys(
        submission,
        {
            "schema",
            "arm_id",
            "route_evidence_sha256",
            "source_status",
            "status",
            "labels_shape",
            "labels_sha256",
            "reason",
        },
        "route submission",
    )
    if submission["schema"] != "audio-extract/counterfactual-route-submission/v1" or submission["arm_id"] != arm_id:
        raise GroupedReportArtifactV4Error("route submission has wrong schema/arm")
    _sha(submission["route_evidence_sha256"], "route_evidence_sha256")
    _nonempty_text(submission["source_status"], "submission.source_status")
    _nonempty_text(submission["reason"], "submission.reason")
    _validate_labels(
        submission["status"],
        submission["labels_shape"],
        submission["labels_sha256"],
        "route submission",
    )
    for name in ("source_status", "status", "labels_shape", "labels_sha256", "reason"):
        if submission[name] != output[name]:
            raise GroupedReportArtifactV4Error(
                f"route output/submission differ at {name}"
            )
    route_artifact_sha = provenance["route_artifact_sha256"]
    if submission["route_evidence_sha256"] != route_artifact_sha:
        raise GroupedReportArtifactV4Error(
            "submission does not name the route artifact"
        )
    if provenance["submission_sha256"] != _mapping_sha(submission):
        raise GroupedReportArtifactV4Error("submission semantic SHA mismatch")
    if row["route_provenance_sha256"] != _mapping_sha(provenance):
        raise GroupedReportArtifactV4Error("route-provenance semantic SHA mismatch")
    evaluation = _validate_evaluation(
        row["complete_route_evaluation"],
        unit=unit,
        arm_id=arm_id,
        submission_sha256=provenance["submission_sha256"],
        submission_status=submission["status"],
    )
    if row["complete_route_evaluation_sha256"] != _mapping_sha(evaluation):
        raise GroupedReportArtifactV4Error(
            "complete-route evaluation semantic SHA mismatch"
        )
    return row


def _parse_arm_summary(value: Any, arm_id: str) -> ArmSummaryV2:
    row = _mapping(value, f"{arm_id} summary")
    names = {field.name for field in fields(ArmSummaryV2)}
    _exact_keys(row, {"schema", *names}, f"{arm_id} summary")
    if row["schema"] != "audio-extract/d0-r0-arm-summary/v2":
        raise GroupedReportArtifactV4Error("unknown arm-summary schema")
    result = ArmSummaryV2(**{name: row[name] for name in names})
    result.validate()
    if result.arm_id != arm_id:
        raise GroupedReportArtifactV4Error(
            f"{arm_id} summary names another arm"
        )
    return result


def _parse_pairwise_summary(value: Any) -> PairwiseSummaryV2:
    row = _mapping(value, "pairwise summary")
    names = {field.name for field in fields(PairwiseSummaryV2)}
    _exact_keys(row, {"schema", *names}, "pairwise summary")
    if row["schema"] != "audio-extract/d0-r0-pairwise-summary/v2":
        raise GroupedReportArtifactV4Error("unknown pairwise-summary schema")
    values = {name: row[name] for name in names}
    cross = _list(values["status_cross_tab"], "status_cross_tab")
    values["status_cross_tab"] = tuple(
        tuple(item) if type(item) is list else item for item in cross
    )
    result = PairwiseSummaryV2(**values)
    result.validate()
    return result


def _expected_arm_summary(
    arm_id: str,
    branches: Sequence[tuple[str, Mapping[str, Any]]],
) -> ArmSummaryV2:
    evaluations = [(unit_sha, row["complete_route_evaluation"]) for unit_sha, row in branches]
    statuses = [row["status"] for _, row in evaluations]
    safe = [(unit_sha, row) for unit_sha, row in evaluations if row["status"] == "ROUTE_SAFE"]
    regrets = [float(row["selection_regret"]) for _, row in safe]
    temporal = [float(row["temporal_switches"]) for _, row in safe]
    frequency = [float(row["frequency_switches"]) for _, row in safe]
    worst_unit = None
    if safe:
        worst = max(regrets)
        worst_unit = min(
            unit_sha
            for unit_sha, row in safe
            if float(row["selection_regret"]) == worst
        )
    unavailable = statuses.count("ORACLE_UNAVAILABLE")
    available = len(statuses) - unavailable
    routes = sum(status in ROUTE_STATUSES for status in statuses)
    result = ArmSummaryV2(
        arm_id=arm_id,
        unit_count=len(statuses),
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
            len(row["critical_false_safe_violations"])
            for _, row in evaluations
        ),
        incomplete_secondary_violation_count=sum(
            len(row["incomplete_secondary_violations"])
            for _, row in evaluations
        ),
        route_submission_coverage=(None if available == 0 else routes / available),
        safe_route_coverage=(None if available == 0 else len(safe) / available),
        abstention_rate=(
            None if available == 0 else statuses.count("ABSTAIN") / available
        ),
        mean_safe_regret=fmean(regrets) if regrets else None,
        median_safe_regret=median(regrets) if regrets else None,
        worst_safe_regret=max(regrets) if regrets else None,
        worst_safe_regret_unit_sha256=worst_unit,
        mean_safe_temporal_switches=fmean(temporal) if temporal else None,
        mean_safe_frequency_switches=fmean(frequency) if frequency else None,
    )
    result.validate()
    return result


def _expected_pairwise_summary(
    pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    tolerance: float,
) -> PairwiseSummaryV2:
    evaluations = [
        (unit_sha, d0["complete_route_evaluation"], r0["complete_route_evaluation"])
        for unit_sha, d0, r0 in pairs
    ]
    oracle_available = sum(d0["status"] != "ORACLE_UNAVAILABLE" for _, d0, _ in evaluations)
    both_routes = sum(
        d0["status"] in ROUTE_STATUSES and r0["status"] in ROUTE_STATUSES
        for _, d0, r0 in evaluations
    )
    safe = [
        (unit_sha, d0, r0)
        for unit_sha, d0, r0 in evaluations
        if d0["status"] == r0["status"] == "ROUTE_SAFE"
    ]
    regret = [
        float(r0["selection_regret"]) - float(d0["selection_regret"])
        for _, d0, r0 in safe
    ]
    temporal = [
        float(r0["temporal_switches"]) - float(d0["temporal_switches"])
        for _, d0, r0 in safe
    ]
    frequency = [
        float(r0["frequency_switches"]) - float(d0["frequency_switches"])
        for _, d0, r0 in safe
    ]
    cross: dict[tuple[str, str], int] = {}
    for _, d0, r0 in evaluations:
        key = (d0["status"], r0["status"])
        cross[key] = cross.get(key, 0) + 1
    result = PairwiseSummaryV2(
        unit_count=len(evaluations),
        oracle_available_count=oracle_available,
        both_submitted_route_count=both_routes,
        both_safe_count=len(safe),
        both_submitted_route_coverage=(
            None if oracle_available == 0 else both_routes / oracle_available
        ),
        both_safe_coverage=(
            None if oracle_available == 0 else len(safe) / oracle_available
        ),
        d0_safe_r0_not_safe_count=sum(
            d0["status"] == "ROUTE_SAFE" and r0["status"] != "ROUTE_SAFE"
            for _, d0, r0 in evaluations
        ),
        r0_safe_d0_not_safe_count=sum(
            r0["status"] == "ROUTE_SAFE" and d0["status"] != "ROUTE_SAFE"
            for _, d0, r0 in evaluations
        ),
        both_catastrophic_count=sum(
            d0["status"] == r0["status"] == "ROUTE_CATASTROPHIC_FALSE_SAFE"
            for _, d0, r0 in evaluations
        ),
        d0_catastrophic_r0_not_count=sum(
            d0["status"] == "ROUTE_CATASTROPHIC_FALSE_SAFE"
            and r0["status"] != "ROUTE_CATASTROPHIC_FALSE_SAFE"
            for _, d0, r0 in evaluations
        ),
        r0_catastrophic_d0_not_count=sum(
            r0["status"] == "ROUTE_CATASTROPHIC_FALSE_SAFE"
            and d0["status"] != "ROUTE_CATASTROPHIC_FALSE_SAFE"
            for _, d0, r0 in evaluations
        ),
        r0_lower_regret_count=sum(value < -tolerance for value in regret),
        d0_lower_regret_count=sum(value > tolerance for value in regret),
        regret_tie_count=sum(abs(value) <= tolerance for value in regret),
        mean_r0_minus_d0_regret=fmean(regret) if regret else None,
        median_r0_minus_d0_regret=median(regret) if regret else None,
        min_r0_minus_d0_regret=min(regret) if regret else None,
        max_r0_minus_d0_regret=max(regret) if regret else None,
        mean_r0_minus_d0_temporal_switches=(
            fmean(temporal) if temporal else None
        ),
        mean_r0_minus_d0_frequency_switches=(
            fmean(frequency) if frequency else None
        ),
        status_cross_tab=tuple(
            sorted((left, right, count) for (left, right), count in cross.items())
        ),
    )
    result.validate()
    return result


def _payload(
    report: GroupedComparisonReportV3,
    prereg: GroupedComparisonPreregistrationV1,
    verifier_commit: str,
) -> dict[str, Any]:
    report.validate(prereg)
    _git(verifier_commit, "v4 artifact verifier_commit")
    by_key = {
        (row.unit.sha256, row.arm_id): row
        for row in report.work_evaluations
    }
    pairs = []
    for unit in prereg.expected_units:
        d0 = by_key[(unit.sha256, "D0")]
        r0 = by_key[(unit.sha256, "R0")]
        if (d0.unit, r0.unit) != (unit, unit):
            raise GroupedReportArtifactV4Error(
                "paired report rows do not share the preregistered unit"
            )
        if (d0.arm_id, r0.arm_id) != ("D0", "R0"):
            raise GroupedReportArtifactV4Error(
                "paired report rows are not ordered D0/R0"
            )
        d0_unavailable = d0.evaluation.status == "ORACLE_UNAVAILABLE"
        r0_unavailable = r0.evaluation.status == "ORACLE_UNAVAILABLE"
        if d0_unavailable != r0_unavailable:
            raise GroupedReportArtifactV4Error(
                "paired report has one-sided oracle availability"
            )
        pairs.append(
            {
                "schema": PAIRED_UNIT_SCHEMA,
                "unit": unit.identity_dict(),
                "unit_sha256": unit.sha256,
                "d0": d0.identity_dict(prereg),
                "r0": r0.identity_dict(prereg),
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": STATUS,
        "promotion_decision": None,
        "preregistration_sha256": prereg.sha256,
        "source_v3_report_sha256": report.sha256(prereg),
        "source_v3_verifier_commit": report.verifier_commit,
        "paired_regret_tolerance": float(prereg.paired_regret_tolerance),
        "verifier_commit": verifier_commit,
        "paired_units": pairs,
        "d0_summary": report.d0_summary.identity_dict(),
        "r0_summary": report.r0_summary.identity_dict(),
        "pairwise_summary": report.pairwise_summary.identity_dict(),
    }


def _validate_complete_payload(
    value: Mapping[str, Any],
    *,
    preregistration_sha256: str,
    source_v3_report_sha256: str,
    verifier_commit: str,
) -> None:
    expected_root = {
        "schema",
        "status",
        "promotion_decision",
        "preregistration_sha256",
        "source_v3_report_sha256",
        "source_v3_verifier_commit",
        "paired_regret_tolerance",
        "verifier_commit",
        "paired_units",
        "d0_summary",
        "r0_summary",
        "pairwise_summary",
    }
    _exact_keys(value, expected_root, "v4 report payload")
    expected_header = {
        "schema": REPORT_SCHEMA,
        "status": STATUS,
        "promotion_decision": None,
        "preregistration_sha256": preregistration_sha256,
        "source_v3_report_sha256": source_v3_report_sha256,
        "verifier_commit": verifier_commit,
    }
    drift = [
        name
        for name, expected_value in expected_header.items()
        if value[name] != expected_value
    ]
    if drift:
        raise GroupedReportArtifactV4Error(
            f"v4 report payload header differs: {drift}"
        )
    _git(value["source_v3_verifier_commit"], "source_v3_verifier_commit")
    tolerance = _nonnegative_float(
        value["paired_regret_tolerance"], "paired_regret_tolerance"
    )
    paired = _list(value["paired_units"], "paired_units")
    if len(paired) < 2:
        raise GroupedReportArtifactV4Error(
            "v4 report requires at least two paired held-out units"
        )
    unit_shas = []
    validated_pairs = []
    d0_branches = []
    r0_branches = []
    flattened = []
    policy_sha = None
    for index, item in enumerate(paired):
        pair = _mapping(item, f"paired_units[{index}]")
        _exact_keys(
            pair,
            {"schema", "unit", "unit_sha256", "d0", "r0"},
            f"paired_units[{index}]",
        )
        if pair["schema"] != PAIRED_UNIT_SCHEMA:
            raise GroupedReportArtifactV4Error("unknown paired-unit schema")
        unit = _unit_from_dict(pair["unit"])
        if pair["unit_sha256"] != unit.sha256:
            raise GroupedReportArtifactV4Error(
                "paired unit SHA differs from its content"
            )
        d0 = _validate_work_branch(
            pair["d0"],
            unit=unit,
            arm_id="D0",
            preregistration_sha256=preregistration_sha256,
        )
        r0 = _validate_work_branch(
            pair["r0"],
            unit=unit,
            arm_id="R0",
            preregistration_sha256=preregistration_sha256,
        )
        d0_eval = d0["complete_route_evaluation"]
        r0_eval = r0["complete_route_evaluation"]
        exact_fields = (
            "exact_evidence_sha256",
            "exact_panel_sha256",
            "exact_preflight_sha256",
            "policy_sha256",
            "exact_oracle_decision_sha256",
            "exact_oracle_objective",
        )
        if any(d0_eval[name] != r0_eval[name] for name in exact_fields):
            raise GroupedReportArtifactV4Error(
                "D0/R0 paired exact evidence or oracle objective differs"
            )
        if (d0_eval["status"] == "ORACLE_UNAVAILABLE") != (
            r0_eval["status"] == "ORACLE_UNAVAILABLE"
        ):
            raise GroupedReportArtifactV4Error(
                "paired unit has one-sided oracle availability"
            )
        if policy_sha is None:
            policy_sha = d0_eval["policy_sha256"]
        elif policy_sha != d0_eval["policy_sha256"]:
            raise GroupedReportArtifactV4Error(
                "paired units use different routing policies"
            )
        unit_shas.append(unit.sha256)
        validated_pairs.append((unit.sha256, d0, r0))
        d0_branches.append((unit.sha256, d0))
        r0_branches.append((unit.sha256, r0))
        flattened.extend((d0, r0))
    if unit_shas != sorted(unit_shas) or len(set(unit_shas)) != len(unit_shas):
        raise GroupedReportArtifactV4Error(
            "paired units must be unique and in canonical SHA order"
        )
    d0_summary = _parse_arm_summary(value["d0_summary"], "D0")
    r0_summary = _parse_arm_summary(value["r0_summary"], "R0")
    pairwise = _parse_pairwise_summary(value["pairwise_summary"])
    if d0_summary != _expected_arm_summary("D0", d0_branches):
        raise GroupedReportArtifactV4Error(
            "D0 summary differs from paired work evidence"
        )
    if r0_summary != _expected_arm_summary("R0", r0_branches):
        raise GroupedReportArtifactV4Error(
            "R0 summary differs from paired work evidence"
        )
    if pairwise != _expected_pairwise_summary(validated_pairs, tolerance):
        raise GroupedReportArtifactV4Error(
            "pairwise summary differs from paired work evidence"
        )
    source_v3 = {
        "schema": SOURCE_REPORT_SCHEMA,
        "status": STATUS,
        "promotion_decision": None,
        "preregistration_sha256": preregistration_sha256,
        "verifier_commit": value["source_v3_verifier_commit"],
        "work_evaluations": flattened,
        "d0_summary": value["d0_summary"],
        "r0_summary": value["r0_summary"],
        "pairwise_summary": value["pairwise_summary"],
    }
    if _mapping_sha(source_v3) != source_v3_report_sha256:
        raise GroupedReportArtifactV4Error(
            "embedded paired evidence does not reconstruct the named v3 report"
        )


@dataclass(frozen=True)
class GroupedComparisonReportArtifactV4:
    preregistration_sha256: str
    source_v3_report_sha256: str
    verifier_commit: str
    payload_utf8: bytes
    payload_sha256: str
    status: str = STATUS

    @classmethod
    def build(
        cls,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        *,
        verifier_commit: str,
    ) -> "GroupedComparisonReportArtifactV4":
        value = _payload(report, prereg, verifier_commit)
        payload = _canonical(value)
        result = cls(
            preregistration_sha256=prereg.sha256,
            source_v3_report_sha256=report.sha256(prereg),
            verifier_commit=verifier_commit,
            payload_utf8=payload,
            payload_sha256=(
                "sha256:" + hashlib.sha256(payload).hexdigest()
            ),
        )
        result.validate_frozen()
        return result

    def validate_frozen(self) -> None:
        if self.status != STATUS:
            raise GroupedReportArtifactV4Error(
                "v4 report artifact must remain non-promoting"
            )
        _sha(self.preregistration_sha256, "preregistration_sha256")
        _sha(self.source_v3_report_sha256, "source_v3_report_sha256")
        _sha(self.payload_sha256, "payload_sha256")
        _git(self.verifier_commit, "verifier_commit")
        if type(self.payload_utf8) is not bytes or not self.payload_utf8:
            raise GroupedReportArtifactV4Error(
                "v4 report payload must be non-empty immutable bytes"
            )
        actual = "sha256:" + hashlib.sha256(self.payload_utf8).hexdigest()
        if actual != self.payload_sha256:
            raise GroupedReportArtifactV4Error(
                "v4 report payload bytes differ from their identity"
            )
        try:
            value = json.loads(self.payload_utf8.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GroupedReportArtifactV4Error(
                f"v4 report payload is not canonical UTF-8 JSON: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise GroupedReportArtifactV4Error(
                "v4 report payload is not a JSON object"
            )
        if _canonical(value) != self.payload_utf8:
            raise GroupedReportArtifactV4Error(
                "v4 report payload is not in canonical form"
            )
        _validate_complete_payload(
            value,
            preregistration_sha256=self.preregistration_sha256,
            source_v3_report_sha256=self.source_v3_report_sha256,
            verifier_commit=self.verifier_commit,
        )

    def validate_against_source(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> None:
        self.validate_frozen()
        expected = _canonical(_payload(report, prereg, self.verifier_commit))
        if expected != self.payload_utf8:
            raise GroupedReportArtifactV4Error(
                "v4 report artifact differs from the current source object graph"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate_frozen()
        return json.loads(self.payload_utf8.decode("utf-8"))

    @property
    def sha256(self) -> str:
        self.validate_frozen()
        return self.payload_sha256
