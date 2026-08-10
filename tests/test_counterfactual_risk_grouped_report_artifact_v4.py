import dataclasses
import hashlib
import json

import numpy as np
import pytest

from audio_extract.counterfactual_risk_complete_route_evaluation_v1 import (
    ArmRouteSubmissionV1,
    CompleteRouteEvaluationV1,
)
from audio_extract.counterfactual_risk_grouped_comparison_v1 import (
    ComparisonArmV1,
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
)
from audio_extract.counterfactual_risk_grouped_comparison_v3 import (
    ArmRouteProvenanceV3,
    FrozenArmInferenceRunV3,
    GroupedComparisonReportV3,
    RouteArtifactManifestV3,
    RouteOutputV3,
    WorkArmEvaluationV3,
)
from audio_extract.counterfactual_risk_grouped_report_artifact_v4 import (
    GroupedComparisonReportArtifactV4,
    GroupedReportArtifactV4Error,
    _validate_violation,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def unit(name: str) -> HeldOutUnitV1:
    return HeldOutUnitV1(
        work_id=name,
        recording_session_id=f"session-{name}",
        target_singer_id=f"singer-{name}",
        source_family_sha256=sha(f"source-{name}"),
        query_condition_sha256=sha(f"query-{name}"),
        query_quality_stratum_sha256=sha(f"quality-{name}"),
        exact_evidence_sha256=sha(f"evidence-{name}"),
        exact_panel_sha256=sha(f"panel-{name}"),
        exact_preflight_sha256=sha(f"preflight-{name}"),
        exact_oracle_decision_sha256=sha(f"oracle-{name}"),
    )


def arm(arm_id: str) -> ComparisonArmV1:
    return ComparisonArmV1(
        arm_id=arm_id,
        method_id={
            "D0": "structured_route_imitation/v1",
            "R0": "counterfactual_risk_prediction/v1",
        }[arm_id],
        artifact_manifest_sha256=sha(f"artifact-{arm_id}"),
        compute_budget_sha256=sha("equal-compute"),
        checkpoint_rule_sha256=sha("checkpoint-rule"),
        inference_policy_sha256=sha(f"inference-{arm_id}"),
        source_commit=git("source"),
    )


def prereg(*units: HeldOutUnitV1) -> GroupedComparisonPreregistrationV1:
    return GroupedComparisonPreregistrationV1(
        experiment_id="D0_R0_IMMUTABLE_ARTIFACT_V4",
        dataset_sha256=sha("dataset"),
        split_sha256=sha("split"),
        test_subset_sha256=sha("test-subset"),
        exact_evidence_manifest_sha256=sha("evidence-manifest"),
        candidate_panel_sha256=sha("candidate-panel"),
        feature_contract_sha256=sha("feature-contract"),
        metric_contract_sha256=sha("metric-contract"),
        routing_policy_sha256=sha("routing-policy"),
        complete_route_evaluation_contract_sha256=sha("evaluation-contract"),
        comparison_policy_sha256=sha("comparison-policy"),
        promotion_policy_sha256=sha("promotion-separate"),
        paired_regret_tolerance=1e-8,
        d0_arm=arm("D0"),
        r0_arm=arm("R0"),
        expected_units=tuple(sorted(units, key=lambda row: row.sha256)),
        source_commit=git("source"),
    )


def work_row(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
) -> WorkArmEvaluationV3:
    frozen = p.d0_arm if arm_id == "D0" else p.r0_arm
    labels = np.zeros((2, 2), dtype=np.int32)
    provisional = ArmRouteSubmissionV1(
        arm_id=arm_id,
        route_evidence_sha256=sha("placeholder"),
        source_status="ROUTE",
        status="ROUTE",
        labels=labels,
        reason="route",
    )
    run = FrozenArmInferenceRunV3(
        preregistration_sha256=p.sha256,
        unit_sha256=held_out.sha256,
        group_family_sha256=held_out.group_family_sha256,
        query_condition_sha256=held_out.query_condition_sha256,
        query_quality_stratum_sha256=(
            held_out.query_quality_stratum_sha256
        ),
        arm_id=arm_id,
        arm_run_sha256=frozen.sha256,
        arm_artifact_manifest_sha256=frozen.artifact_manifest_sha256,
        compute_budget_sha256=frozen.compute_budget_sha256,
        checkpoint_rule_sha256=frozen.checkpoint_rule_sha256,
        inference_policy_sha256=frozen.inference_policy_sha256,
        dataset_sha256=p.dataset_sha256,
        split_sha256=p.split_sha256,
        test_subset_sha256=p.test_subset_sha256,
        candidate_panel_sha256=p.candidate_panel_sha256,
        feature_contract_sha256=p.feature_contract_sha256,
        metric_contract_sha256=p.metric_contract_sha256,
        inference_input_manifest_sha256=sha(
            f"inputs-{held_out.work_id}-{arm_id}"
        ),
        runner_bundle_sha256=sha(f"runner-{arm_id}"),
        source_commit=p.source_commit,
        runner_commit=git(f"runner-{arm_id}"),
    )
    artifact = RouteArtifactManifestV3(
        run,
        RouteOutputV3.from_submission(held_out, provisional),
        git(f"artifact-verifier-{arm_id}"),
    )
    submission = dataclasses.replace(
        provisional,
        route_evidence_sha256=artifact.sha256(p, held_out),
    )
    provenance = ArmRouteProvenanceV3(artifact, submission)
    evaluation = CompleteRouteEvaluationV1(
        arm_id=arm_id,
        exact_evidence_sha256=held_out.exact_evidence_sha256,
        exact_panel_sha256=held_out.exact_panel_sha256,
        exact_preflight_sha256=held_out.exact_preflight_sha256,
        policy_sha256=p.routing_policy_sha256,
        exact_oracle_decision_sha256=(
            held_out.exact_oracle_decision_sha256
        ),
        submission_sha256=submission.sha256,
        status="ROUTE_SAFE",
        critical_false_safe_violations=(),
        incomplete_secondary_violations=(),
        exact_oracle_objective=1.0,
        submitted_exact_objective=1.0,
        selection_regret=0.0,
        temporal_switches=0,
        frequency_switches=0,
        reason="safe",
    )
    return WorkArmEvaluationV3(held_out, provenance, evaluation)


def fixture():
    p = prereg(unit("one"), unit("two"))
    rows = [
        work_row(p, held_out, arm_id)
        for held_out in p.expected_units
        for arm_id in ("D0", "R0")
    ]
    report = GroupedComparisonReportV3.build(
        p, rows, verifier_commit=git("report-verifier")
    )
    artifact = GroupedComparisonReportArtifactV4.build(
        report,
        p,
        verifier_commit=git("snapshot-verifier"),
    )
    return p, report, artifact


def repack(artifact, payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return dataclasses.replace(
        artifact,
        payload_utf8=encoded,
        payload_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
    )


def test_v4_artifact_is_canonical_paired_and_nonpromoting():
    p, report, artifact = fixture()
    artifact.validate_frozen()
    artifact.validate_against_source(report, p)
    payload = artifact.identity_dict()
    assert payload["promotion_decision"] is None
    assert payload["status"] == "COMPLETE_NO_PROMOTION_DECISION"
    assert payload["paired_regret_tolerance"] == p.paired_regret_tolerance
    assert len(payload["paired_units"]) == len(p.expected_units)
    assert all(
        row["d0"]["route_provenance"]["submission"]["arm_id"] == "D0"
        and row["r0"]["route_provenance"]["submission"]["arm_id"] == "R0"
        for row in payload["paired_units"]
    )


def test_mutating_source_labels_cannot_change_issued_artifact_bytes():
    p, report, artifact = fixture()
    before_bytes = artifact.payload_utf8
    before_sha = artifact.sha256
    labels = report.work_evaluations[0].route_provenance.submission.labels
    labels[0, 0] = 1
    assert artifact.payload_utf8 == before_bytes
    assert artifact.sha256 == before_sha
    artifact.validate_frozen()
    with pytest.raises(Exception):
        artifact.validate_against_source(report, p)


def test_payload_byte_tampering_is_detected():
    _p, _report, artifact = fixture()
    broken = dataclasses.replace(
        artifact,
        payload_utf8=artifact.payload_utf8 + b" ",
    )
    with pytest.raises(
        GroupedReportArtifactV4Error,
        match="payload bytes differ",
    ):
        broken.validate_frozen()


def test_header_substitution_is_detected_even_with_rehashed_bytes():
    _p, _report, artifact = fixture()
    payload = artifact.identity_dict()
    payload["promotion_decision"] = "R0"
    broken = repack(artifact, payload)
    with pytest.raises(
        GroupedReportArtifactV4Error,
        match="header differs",
    ):
        broken.validate_frozen()


def test_rehashed_malformed_payload_is_rejected():
    _p, _report, artifact = fixture()
    payload = artifact.identity_dict()
    payload["paired_units"] = [{}]
    broken = repack(artifact, payload)
    with pytest.raises(GroupedReportArtifactV4Error):
        broken.validate_frozen()


def test_swapping_d0_branches_between_outer_units_is_rejected():
    _p, _report, artifact = fixture()
    payload = artifact.identity_dict()
    first, second = payload["paired_units"]
    first["d0"], second["d0"] = second["d0"], first["d0"]
    broken = repack(artifact, payload)
    with pytest.raises(
        GroupedReportArtifactV4Error,
        match="not bound to the outer held-out unit",
    ):
        broken.validate_frozen()


def test_rehashed_nested_hash_substitution_is_rejected():
    _p, _report, artifact = fixture()
    payload = artifact.identity_dict()
    payload["paired_units"][0]["d0"]["route_provenance"][
        "submission_sha256"
    ] = sha("other-submission")
    broken = repack(artifact, payload)
    with pytest.raises(
        GroupedReportArtifactV4Error,
        match="submission semantic SHA mismatch",
    ):
        broken.validate_frozen()


def test_source_v3_report_is_reconstructed_from_paired_payload():
    _p, _report, artifact = fixture()
    payload = artifact.identity_dict()
    payload["source_v3_verifier_commit"] = git("other-v3-verifier")
    broken = repack(artifact, payload)
    with pytest.raises(
        GroupedReportArtifactV4Error,
        match="does not reconstruct the named v3 report",
    ):
        broken.validate_frozen()


def test_above_threshold_numeric_facts_are_semantically_validated():
    violation = {
        "schema": "audio-extract/counterfactual-route-violation/v1",
        "time_index": 0,
        "band_index": 0,
        "candidate_index": 0,
        "metric_name": "voice",
        "kind": "critical_above_threshold",
        "threshold": 1.0,
        "value": 0.5,
    }
    with pytest.raises(
        GroupedReportArtifactV4Error,
        match="does not exceed",
    ):
        _validate_violation(violation, critical=True)
