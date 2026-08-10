import dataclasses
import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_complete_route_evaluation_v1 import (
    ArmRouteSubmissionV1,
    CompleteRouteEvaluationV1,
    SelectedRiskViolationV1,
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
    GroupedComparisonV3Error,
    RouteArtifactManifestV3,
    RouteOutputV3,
    WorkArmEvaluationV3,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def unit(name: str, *, shared_exact: str | None = None) -> HeldOutUnitV1:
    exact = shared_exact or name
    return HeldOutUnitV1(
        work_id=name,
        recording_session_id=f"session-{name}",
        target_singer_id=f"singer-{name}",
        source_family_sha256=sha(f"source-{name}"),
        query_condition_sha256=sha(f"query-{name}"),
        query_quality_stratum_sha256=sha(f"quality-{name}"),
        exact_evidence_sha256=sha(f"exact-{exact}"),
        exact_panel_sha256=sha(f"panel-{exact}"),
        exact_preflight_sha256=sha(f"preflight-{exact}"),
        exact_oracle_decision_sha256=sha(f"oracle-{exact}"),
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
    ordered = tuple(sorted(units, key=lambda row: row.sha256))
    return GroupedComparisonPreregistrationV1(
        experiment_id="D0_R0_HELD_OUT_V3",
        dataset_sha256=sha("dataset"),
        split_sha256=sha("split"),
        test_subset_sha256=sha("test-subset"),
        exact_evidence_manifest_sha256=sha("exact-manifest"),
        candidate_panel_sha256=sha("candidate-panel"),
        feature_contract_sha256=sha("feature-contract"),
        metric_contract_sha256=sha("metric-contract"),
        routing_policy_sha256=sha("routing-policy"),
        complete_route_evaluation_contract_sha256=sha("evaluator-v1"),
        comparison_policy_sha256=sha("comparison-v3"),
        promotion_policy_sha256=sha("promotion-separate"),
        paired_regret_tolerance=1e-8,
        d0_arm=arm("D0"),
        r0_arm=arm("R0"),
        expected_units=ordered,
        source_commit=git("source"),
    )


def critical_violation() -> SelectedRiskViolationV1:
    return SelectedRiskViolationV1(
        0, 0, 0, "voice", "critical_above_threshold", 1.0, 1.2
    )


def incomplete_violation() -> SelectedRiskViolationV1:
    return SelectedRiskViolationV1(
        0, 0, 0, "artifact", "required_secondary_missing", None, None
    )


def provisional_submission(
    arm_id: str,
    *,
    status: str,
) -> ArmRouteSubmissionV1:
    return ArmRouteSubmissionV1(
        arm_id=arm_id,
        route_evidence_sha256=sha("placeholder"),
        source_status=("ROUTE" if status == "ROUTE" else "ABSTAIN_UNCERTAIN"),
        status=status,
        labels=(np.asarray([[0]], dtype=np.int32) if status == "ROUTE" else None),
        reason=f"{arm_id} {status}",
    )


def inference_run(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
) -> FrozenArmInferenceRunV3:
    frozen = p.d0_arm if arm_id == "D0" else p.r0_arm
    return FrozenArmInferenceRunV3(
        preregistration_sha256=p.sha256,
        unit_sha256=held_out.sha256,
        group_family_sha256=held_out.group_family_sha256,
        query_condition_sha256=held_out.query_condition_sha256,
        query_quality_stratum_sha256=held_out.query_quality_stratum_sha256,
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
            f"input-{held_out.work_id}-{arm_id}"
        ),
        runner_bundle_sha256=sha(f"runner-{arm_id}"),
        source_commit=frozen.source_commit,
        runner_commit=git(f"runner-{arm_id}"),
    )


def provenance(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    *,
    submission_status: str,
) -> ArmRouteProvenanceV3:
    provisional = provisional_submission(arm_id, status=submission_status)
    output = RouteOutputV3.from_submission(held_out, provisional)
    artifact = RouteArtifactManifestV3(
        inference_run=inference_run(p, held_out, arm_id),
        route_output=output,
        verifier_commit=git(f"artifact-verifier-{arm_id}"),
    )
    final = dataclasses.replace(
        provisional,
        route_evidence_sha256=artifact.sha256(p, held_out),
    )
    return ArmRouteProvenanceV3(artifact, final)


def evaluation(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    route: ArmRouteProvenanceV3,
    *,
    status: str,
    regret: float = 0.0,
) -> CompleteRouteEvaluationV1:
    critical = (
        (critical_violation(),)
        if status == "ROUTE_CATASTROPHIC_FALSE_SAFE"
        else ()
    )
    incomplete = (
        (incomplete_violation(),)
        if status == "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE"
        else ()
    )
    route_status = status.startswith("ROUTE_")
    safe = status == "ROUTE_SAFE"
    return CompleteRouteEvaluationV1(
        arm_id=route.arm_id,
        exact_evidence_sha256=held_out.exact_evidence_sha256,
        exact_panel_sha256=held_out.exact_panel_sha256,
        exact_preflight_sha256=held_out.exact_preflight_sha256,
        policy_sha256=p.routing_policy_sha256,
        exact_oracle_decision_sha256=held_out.exact_oracle_decision_sha256,
        submission_sha256=route.submission.sha256,
        status=status,
        critical_false_safe_violations=critical,
        incomplete_secondary_violations=incomplete,
        exact_oracle_objective=(
            None if status == "ORACLE_UNAVAILABLE" else 1.0
        ),
        submitted_exact_objective=(1.0 + regret if safe else None),
        selection_regret=(regret if safe else None),
        temporal_switches=(0 if route_status else None),
        frequency_switches=(0 if route_status else None),
        reason=f"{route.arm_id} {status}",
    )


def record(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    *,
    evaluation_status: str = "ROUTE_SAFE",
    submission_status: str | None = None,
    regret: float = 0.0,
) -> WorkArmEvaluationV3:
    if submission_status is None:
        submission_status = (
            "ABSTAIN" if evaluation_status == "ABSTAIN" else "ROUTE"
        )
    route = provenance(
        p,
        held_out,
        arm_id,
        submission_status=submission_status,
    )
    return WorkArmEvaluationV3(
        unit=held_out,
        route_provenance=route,
        evaluation=evaluation(
            p,
            held_out,
            route,
            status=evaluation_status,
            regret=regret,
        ),
    )


def matrix(p: GroupedComparisonPreregistrationV1):
    return [
        record(p, held_out, arm_id)
        for held_out in p.expected_units
        for arm_id in ("D0", "R0")
    ]


def test_v3_builds_nonpromoting_report_through_v2_statistics():
    p = prereg(unit("one"), unit("two"))
    report = GroupedComparisonReportV3.build(
        p,
        matrix(p),
        verifier_commit=git("report-verifier"),
    )
    report.validate(p)
    payload = report.identity_dict(p)
    assert payload["promotion_decision"] is None
    assert payload["schema"].endswith("grouped-comparison-report/v3")
    assert report.d0_summary.safe_route_count == 2
    assert report.r0_summary.safe_route_count == 2


def test_route_artifact_binds_the_frozen_arm_manifest_and_inference_run():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(p, held_out, "D0")
    run = value.route_provenance.route_artifact.inference_run
    broken_run = dataclasses.replace(
        run,
        arm_artifact_manifest_sha256=p.r0_arm.artifact_manifest_sha256,
    )
    broken_artifact = dataclasses.replace(
        value.route_provenance.route_artifact,
        inference_run=broken_run,
    )
    broken_provenance = dataclasses.replace(
        value.route_provenance,
        route_artifact=broken_artifact,
    )
    with pytest.raises(
        GroupedComparisonV3Error,
        match="differs from frozen arm",
    ):
        dataclasses.replace(
            value, route_provenance=broken_provenance
        ).validate(p)


def test_submission_must_name_the_content_bearing_route_manifest():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(p, held_out, "D0")
    broken_submission = dataclasses.replace(
        value.route_provenance.submission,
        route_evidence_sha256=sha("unrelated-route-manifest"),
    )
    with pytest.raises(
        GroupedComparisonV3Error,
        match="not bound to the route artifact manifest",
    ):
        dataclasses.replace(
            value,
            route_provenance=dataclasses.replace(
                value.route_provenance,
                submission=broken_submission,
            ),
        ).validate(p)


def test_route_output_and_submission_must_match_exactly():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(p, held_out, "R0")
    output = value.route_provenance.route_artifact.route_output
    changed = dataclasses.replace(output, reason="different result")
    artifact = dataclasses.replace(
        value.route_provenance.route_artifact,
        route_output=changed,
    )
    with pytest.raises(
        GroupedComparisonV3Error,
        match="differs from route submission",
    ):
        dataclasses.replace(
            value,
            route_provenance=dataclasses.replace(
                value.route_provenance,
                route_artifact=artifact,
            ),
        ).validate(p)


@pytest.mark.parametrize(
    ("submission_status", "evaluation_status"),
    [
        ("ABSTAIN", "ROUTE_SAFE"),
        ("ROUTE", "ABSTAIN"),
    ],
)
def test_evaluation_status_must_match_the_bound_submission(
    submission_status,
    evaluation_status,
):
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(
        p,
        held_out,
        "D0",
        submission_status=submission_status,
        evaluation_status=evaluation_status,
    )
    with pytest.raises(
        GroupedComparisonV3Error,
        match="contradicts the bound submission",
    ):
        value.validate(p)


def test_oracle_unavailable_can_record_either_submission_kind():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    for submission_status in ("ROUTE", "ABSTAIN"):
        value = record(
            p,
            held_out,
            "D0",
            submission_status=submission_status,
            evaluation_status="ORACLE_UNAVAILABLE",
        )
        value.validate(p)


def test_evaluation_submission_hash_is_transitively_bound():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(p, held_out, "R0")
    changed = dataclasses.replace(
        value.evaluation,
        submission_sha256=sha("other-submission"),
    )
    with pytest.raises(
        GroupedComparisonV3Error,
        match="names another submission",
    ):
        dataclasses.replace(value, evaluation=changed).validate(p)


def test_manifest_identity_changes_with_inference_input_parent():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    first = record(p, held_out, "D0")
    run = first.route_provenance.route_artifact.inference_run
    second_run = dataclasses.replace(
        run,
        inference_input_manifest_sha256=sha("different-inputs"),
    )
    second_artifact = dataclasses.replace(
        first.route_provenance.route_artifact,
        inference_run=second_run,
    )
    assert first.route_provenance.route_artifact.sha256(
        p, held_out
    ) != second_artifact.sha256(p, held_out)


def test_unit_swap_is_refused_even_when_exact_evidence_is_shared():
    first_unit = unit("first", shared_exact="shared")
    second_unit = unit("second", shared_exact="shared")
    p = prereg(first_unit, second_unit)
    value = record(p, first_unit, "D0")
    with pytest.raises(
        GroupedComparisonV3Error,
        match="another held-out unit",
    ):
        dataclasses.replace(value, unit=second_unit).validate(p)
