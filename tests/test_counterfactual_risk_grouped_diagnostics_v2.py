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
    RouteArtifactManifestV3,
    RouteOutputV3,
    WorkArmEvaluationV3,
)
from audio_extract.counterfactual_risk_grouped_diagnostics_v2 import (
    ExactPartitionCertificateV2,
    GeometryManifestV2,
    GroupedComparisonDiagnosticsV2,
    GroupedDiagnosticsV2Error,
    build_grouped_diagnostics_v2,
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
        exact_evidence_sha256=sha(f"exact-{name}"),
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
        experiment_id="D0_R0_DIAGNOSTICS_V2",
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
        expected_units=tuple(sorted(units, key=lambda row: row.sha256)),
        source_commit=git("source"),
    )


def critical_violation() -> SelectedRiskViolationV1:
    return SelectedRiskViolationV1(
        0, 0, 0, "voice", "critical_above_threshold", 1.0, 1.2
    )


def route_provenance(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
) -> ArmRouteProvenanceV3:
    frozen = p.d0_arm if arm_id == "D0" else p.r0_arm
    provisional = ArmRouteSubmissionV1(
        arm_id=arm_id,
        route_evidence_sha256=sha("placeholder"),
        source_status="ROUTE",
        status="ROUTE",
        labels=np.asarray([[0]], dtype=np.int32),
        reason=f"{arm_id} route",
    )
    run = FrozenArmInferenceRunV3(
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
            f"inputs-{held_out.work_id}-{arm_id}"
        ),
        runner_bundle_sha256=sha(f"runner-{arm_id}"),
        source_commit=frozen.source_commit,
        runner_commit=git(f"runner-{arm_id}"),
    )
    artifact = RouteArtifactManifestV3(
        run,
        RouteOutputV3.from_submission(held_out, provisional),
        git(f"artifact-verifier-{arm_id}"),
    )
    final = dataclasses.replace(
        provisional,
        route_evidence_sha256=artifact.sha256(p, held_out),
    )
    return ArmRouteProvenanceV3(artifact, final)


def record(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    *,
    status: str = "ROUTE_SAFE",
    temporal: int = 0,
    frequency: int = 0,
) -> WorkArmEvaluationV3:
    route = route_provenance(p, held_out, arm_id)
    critical = (
        (critical_violation(),)
        if status == "ROUTE_CATASTROPHIC_FALSE_SAFE"
        else ()
    )
    evaluation = CompleteRouteEvaluationV1(
        arm_id=arm_id,
        exact_evidence_sha256=held_out.exact_evidence_sha256,
        exact_panel_sha256=held_out.exact_panel_sha256,
        exact_preflight_sha256=held_out.exact_preflight_sha256,
        policy_sha256=p.routing_policy_sha256,
        exact_oracle_decision_sha256=held_out.exact_oracle_decision_sha256,
        submission_sha256=route.submission.sha256,
        status=status,
        critical_false_safe_violations=critical,
        incomplete_secondary_violations=(),
        exact_oracle_objective=1.0,
        submitted_exact_objective=(1.0 if status == "ROUTE_SAFE" else None),
        selection_regret=(0.0 if status == "ROUTE_SAFE" else None),
        temporal_switches=temporal,
        frequency_switches=frequency,
        reason=f"{arm_id} {status}",
    )
    return WorkArmEvaluationV3(held_out, route, evaluation)


def report(p: GroupedComparisonPreregistrationV1, *, override=None):
    override = override or {}
    rows = []
    for held_out in p.expected_units:
        for arm_id in ("D0", "R0"):
            options = override.get((held_out.work_id, arm_id), {})
            rows.append(record(p, held_out, arm_id, **options))
    return GroupedComparisonReportV3.build(
        p,
        rows,
        verifier_commit=git("report-verifier"),
    )


def certificate(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    *,
    time_cells: int = 2,
    bands: int = 2,
) -> ExactPartitionCertificateV2:
    return ExactPartitionCertificateV2(
        preregistration_sha256=p.sha256,
        unit_sha256=held_out.sha256,
        group_family_sha256=held_out.group_family_sha256,
        exact_panel_sha256=held_out.exact_panel_sha256,
        exact_preflight_sha256=held_out.exact_preflight_sha256,
        exact_oracle_decision_sha256=held_out.exact_oracle_decision_sha256,
        spectral_grid_sha256=sha("common-spectral-grid"),
        partition_contract_sha256=sha("partition-contract"),
        partition_artifact_sha256=sha(f"partition-{held_out.work_id}"),
        time_cell_count=time_cells,
        band_count=bands,
        source_commit=p.source_commit,
        verifier_commit=git("partition-verifier"),
    )


def geometry(
    p: GroupedComparisonPreregistrationV1,
    *,
    time_cells: int = 2,
    bands: int = 2,
) -> GeometryManifestV2:
    return GeometryManifestV2(
        preregistration_sha256=p.sha256,
        certificates=tuple(
            sorted(
                (
                    certificate(
                        p,
                        held_out,
                        time_cells=time_cells,
                        bands=bands,
                    )
                    for held_out in p.expected_units
                ),
                key=lambda row: row.unit_sha256,
            )
        ),
    )


def test_v2_diagnostics_build_from_v3_report_and_certified_geometry():
    p = prereg(unit("one"), unit("two"))
    value = report(p)
    grid = geometry(p)
    result = build_grouped_diagnostics_v2(
        value,
        p,
        grid,
        verifier_commit=git("diagnostic-verifier"),
    )
    result.validate(value, p, grid)
    payload = result.identity_dict(value, p, grid)
    assert payload["promotion_decision"] is None
    assert payload["schema"].endswith("grouped-diagnostics/v2")


def test_partition_certificate_cannot_be_swapped_between_units():
    first = unit("first")
    second = unit("second")
    p = prereg(first, second)
    value = certificate(p, first)
    with pytest.raises(
        GroupedDiagnosticsV2Error,
        match="another exact unit",
    ):
        value.validate(p, second)


def test_partition_certificate_must_name_the_exact_panel():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = dataclasses.replace(
        certificate(p, held_out),
        exact_panel_sha256=sha("different-panel"),
    )
    with pytest.raises(
        GroupedDiagnosticsV2Error,
        match="another exact unit",
    ):
        value.validate(p, held_out)


def test_impossible_switches_are_rejected_for_catastrophic_routes_too():
    one = unit("one")
    two = unit("two")
    p = prereg(one, two)
    value = report(
        p,
        override={
            ("one", "D0"): {
                "status": "ROUTE_CATASTROPHIC_FALSE_SAFE",
                "temporal": 1,
            }
        },
    )
    grid = geometry(p, time_cells=1, bands=1)
    with pytest.raises(
        GroupedDiagnosticsV2Error,
        match="temporal switches exceed unit boundaries",
    ):
        build_grouped_diagnostics_v2(
            value,
            p,
            grid,
            verifier_commit=git("diagnostic-verifier"),
        )


def test_stored_summary_type_tampering_is_refused_before_equality():
    p = prereg(unit("one"), unit("two"))
    value = report(p)
    grid = geometry(p)
    result = build_grouped_diagnostics_v2(
        value,
        p,
        grid,
        verifier_commit=git("diagnostic-verifier"),
    )
    broken_switch = dataclasses.replace(
        result.d0_switching,
        safe_unit_count=True,
    )
    with pytest.raises(
        GroupedDiagnosticsV2Error,
        match="non-negative integer",
    ):
        dataclasses.replace(
            result,
            d0_switching=broken_switch,
        ).validate(value, p, grid)
    broken_status = dataclasses.replace(
        result.status_dominance,
        equal_status_count=2.0,
    )
    with pytest.raises(
        GroupedDiagnosticsV2Error,
        match="non-negative integer",
    ):
        dataclasses.replace(
            result,
            status_dominance=broken_status,
        ).validate(value, p, grid)


def test_partition_certificate_identity_changes_with_partition_artifact():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    first = certificate(p, held_out)
    second = dataclasses.replace(
        first,
        partition_artifact_sha256=sha("different-partition"),
    )
    assert first.sha256(p, held_out) != second.sha256(p, held_out)
