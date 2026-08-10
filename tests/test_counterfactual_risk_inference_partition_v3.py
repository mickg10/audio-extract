import dataclasses
import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import (
    CellPartitionCertificate,
    CellPartitionEntry,
    CellPartitionRegistry,
    RationalMeasure,
)
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
from audio_extract.counterfactual_risk_grouped_diagnostics_v4 import (
    GroupedDiagnosticsV4Error,
    build_grouped_diagnostics_v4,
)
from audio_extract.counterfactual_risk_inference_partition_manifest_v2 import (
    ExactSourceLineageV2,
    InferenceInputPartitionManifestV2,
)
from audio_extract.counterfactual_risk_route_partition_attestation_v2 import (
    RoutePartitionAttestationRegistryV2,
    RoutePartitionAttestationV2,
    RoutePartitionAttestationV2Error,
)
from audio_extract.counterfactual_risk_source_lineage_v3 import (
    ExactSourceLineageV3,
    ExactSourceLineageV3Error,
    ExactSourceReportDocumentV3,
    InferenceInputPartitionManifestV3,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def unit_bundle(name: str):
    seed = HeldOutUnitV1(
        work_id=name,
        recording_session_id=f"session-{name}",
        target_singer_id=f"singer-{name}",
        source_family_sha256=sha(f"source-{name}"),
        query_condition_sha256=sha(f"query-{name}"),
        query_quality_stratum_sha256=sha(f"quality-{name}"),
        exact_evidence_sha256=sha("placeholder"),
        exact_panel_sha256=sha(f"panel-{name}"),
        exact_preflight_sha256=sha(f"preflight-{name}"),
        exact_oracle_decision_sha256=sha(f"oracle-{name}"),
    )
    parents = {
        "truth_manifest_sha256": sha(f"truth-{name}"),
        "mixture_pcm_sha256": sha(f"mixture-{name}"),
        "accompaniment_pcm_sha256": sha(f"accompaniment-{name}"),
        "vocal_pcm_sha256": sha(f"vocal-{name}"),
    }
    risk_report = ExactSourceReportDocumentV3.build(
        role="risk_evidence",
        group_family_sha256=seed.group_family_sha256,
        source_artifact_sha256=sha(f"risk-artifact-{name}"),
        source_artifact_schema="audio-extract/exact-risk-evidence/v1",
        verifier_commit=git("risk-report-verifier"),
        **parents,
    )
    unit = dataclasses.replace(
        seed,
        exact_evidence_sha256=risk_report.sha256,
    )
    return unit, risk_report, parents


def comparison_arm(arm_id: str) -> ComparisonArmV1:
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


def preregistration(*units: HeldOutUnitV1):
    return GroupedComparisonPreregistrationV1(
        experiment_id="D0_R0_RESOLVED_PARTITION_V3",
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
        d0_arm=comparison_arm("D0"),
        r0_arm=comparison_arm("R0"),
        expected_units=tuple(sorted(units, key=lambda value: value.sha256)),
        source_commit=git("source"),
    )


def certificate_bundle(unit, parents, resolution_ms):
    partition_report = ExactSourceReportDocumentV3.build(
        role="partition_source",
        group_family_sha256=unit.group_family_sha256,
        source_artifact_sha256=sha(
            f"partition-artifact-{unit.work_id}-{resolution_ms}"
        ),
        source_artifact_schema="audio-extract/cell-partition-source/v1",
        verifier_commit=git("partition-verifier"),
        **parents,
    )
    certificate = CellPartitionCertificate(
        group_family_sha256=unit.group_family_sha256,
        spectral_grid_sha256=sha(f"grid-{resolution_ms}"),
        resolution_ms=resolution_ms,
        time_cell_count=2,
        band_count=2,
        entries=tuple(
            CellPartitionEntry(
                time_index,
                band_index,
                sha(
                    f"cell-{unit.work_id}-{resolution_ms}-"
                    f"{time_index}-{band_index}"
                ),
                RationalMeasure(1),
            )
            for time_index in range(2)
            for band_index in range(2)
        ),
        time_ranges_sha256=sha(
            f"time-ranges-{unit.work_id}-{resolution_ms}"
        ),
        frequency_ranges_sha256=sha(
            f"frequency-ranges-{resolution_ms}"
        ),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=partition_report.sha256,
        verifier_commit=git("partition-verifier"),
    )
    return certificate, partition_report


def legacy_lineage(unit, selected, parents):
    return ExactSourceLineageV2(
        unit_sha256=unit.sha256,
        group_family_sha256=unit.group_family_sha256,
        exact_evidence_sha256=unit.exact_evidence_sha256,
        exact_source_report_sha256=selected.exact_source_report_sha256,
        exact_evidence_builder_commit=git("risk-report-verifier"),
        partition_source_verifier_commit=selected.verifier_commit,
        lineage_verifier_commit=git("legacy-lineage-verifier"),
        **parents,
    )


def input_manifest(
    p,
    unit,
    risk_report,
    partition_report,
    parents,
    arm_id,
    selected,
):
    arm = p.d0_arm if arm_id == "D0" else p.r0_arm
    base = InferenceInputPartitionManifestV2(
        preregistration_sha256=p.sha256,
        unit_sha256=unit.sha256,
        group_family_sha256=unit.group_family_sha256,
        query_condition_sha256=unit.query_condition_sha256,
        query_quality_stratum_sha256=unit.query_quality_stratum_sha256,
        arm_id=arm_id,
        arm_run_sha256=arm.sha256,
        dataset_sha256=p.dataset_sha256,
        split_sha256=p.split_sha256,
        test_subset_sha256=p.test_subset_sha256,
        candidate_panel_sha256=p.candidate_panel_sha256,
        feature_contract_sha256=p.feature_contract_sha256,
        metric_contract_sha256=p.metric_contract_sha256,
        partition_certificate_sha256=selected.sha256,
        spectral_grid_sha256=selected.spectral_grid_sha256,
        resolution_ms=selected.resolution_ms,
        time_ranges_sha256=selected.time_ranges_sha256,
        frequency_ranges_sha256=selected.frequency_ranges_sha256,
        measure_contract_sha256=selected.measure_contract_sha256,
        source_lineage=legacy_lineage(unit, selected, parents),
        builder_commit=git("input-builder"),
    )
    return InferenceInputPartitionManifestV3(
        base=base,
        source_lineage=ExactSourceLineageV3(
            risk_evidence_report=risk_report,
            partition_source_report=partition_report,
            lineage_verifier_commit=git("source-lineage-v3-verifier"),
        ),
    )


def route_cell(
    p,
    unit,
    risk_report,
    partition_report,
    parents,
    arm_id,
    selected,
    *,
    evaluation_status="ROUTE_SAFE",
):
    arm = p.d0_arm if arm_id == "D0" else p.r0_arm
    manifest = input_manifest(
        p,
        unit,
        risk_report,
        partition_report,
        parents,
        arm_id,
        selected,
    )
    provisional = ArmRouteSubmissionV1(
        arm_id=arm_id,
        route_evidence_sha256=sha("placeholder"),
        source_status="ROUTE",
        status="ROUTE",
        labels=np.zeros((2, 2), dtype=np.int32),
        reason="route",
    )
    run = FrozenArmInferenceRunV3(
        preregistration_sha256=p.sha256,
        unit_sha256=unit.sha256,
        group_family_sha256=unit.group_family_sha256,
        query_condition_sha256=unit.query_condition_sha256,
        query_quality_stratum_sha256=unit.query_quality_stratum_sha256,
        arm_id=arm_id,
        arm_run_sha256=arm.sha256,
        arm_artifact_manifest_sha256=arm.artifact_manifest_sha256,
        compute_budget_sha256=arm.compute_budget_sha256,
        checkpoint_rule_sha256=arm.checkpoint_rule_sha256,
        inference_policy_sha256=arm.inference_policy_sha256,
        dataset_sha256=p.dataset_sha256,
        split_sha256=p.split_sha256,
        test_subset_sha256=p.test_subset_sha256,
        candidate_panel_sha256=p.candidate_panel_sha256,
        feature_contract_sha256=p.feature_contract_sha256,
        metric_contract_sha256=p.metric_contract_sha256,
        inference_input_manifest_sha256=manifest.sha256(
            p,
            unit,
            selected,
        ),
        runner_bundle_sha256=sha(f"runner-{arm_id}"),
        source_commit=p.source_commit,
        runner_commit=git(f"runner-{arm_id}"),
    )
    artifact = RouteArtifactManifestV3(
        run,
        RouteOutputV3.from_submission(unit, provisional),
        git(f"artifact-verifier-{arm_id}"),
    )
    submission = dataclasses.replace(
        provisional,
        route_evidence_sha256=artifact.sha256(p, unit),
    )
    critical = (
        (
            SelectedRiskViolationV1(
                0,
                0,
                0,
                "voice",
                "critical_above_threshold",
                1.0,
                1.2,
            ),
        )
        if evaluation_status == "ROUTE_CATASTROPHIC_FALSE_SAFE"
        else ()
    )
    evaluation = CompleteRouteEvaluationV1(
        arm_id=arm_id,
        exact_evidence_sha256=unit.exact_evidence_sha256,
        exact_panel_sha256=unit.exact_panel_sha256,
        exact_preflight_sha256=unit.exact_preflight_sha256,
        policy_sha256=p.routing_policy_sha256,
        exact_oracle_decision_sha256=unit.exact_oracle_decision_sha256,
        submission_sha256=submission.sha256,
        status=evaluation_status,
        critical_false_safe_violations=critical,
        incomplete_secondary_violations=(),
        exact_oracle_objective=1.0,
        submitted_exact_objective=(
            1.0 if evaluation_status == "ROUTE_SAFE" else None
        ),
        selection_regret=(0.0 if evaluation_status == "ROUTE_SAFE" else None),
        temporal_switches=(
            3 if evaluation_status == "ROUTE_CATASTROPHIC_FALSE_SAFE" else 0
        ),
        frequency_switches=0,
        reason=evaluation_status,
    )
    row = WorkArmEvaluationV3(
        unit,
        ArmRouteProvenanceV3(artifact, submission),
        evaluation,
    )
    return row, RoutePartitionAttestationV2(
        input_manifest=manifest,
        route_artifact_sha256=artifact.sha256(p, unit),
        route_output_sha256=artifact.route_output.sha256(unit, arm_id),
        submission_sha256=submission.sha256,
        partition_certificate_sha256=selected.sha256,
        labels_shape=artifact.route_output.labels_shape,
        labels_sha256=artifact.route_output.labels_sha256,
        verifier_commit=git("attestation-verifier"),
    )


def fixture(*, mismatch=False, catastrophic=False):
    bundles = [unit_bundle("one"), unit_bundle("two")]
    p = preregistration(*(value[0] for value in bundles))
    by_unit = {unit.sha256: (unit, report, parents) for unit, report, parents in bundles}
    certificates = {}
    partition_reports = {}
    for unit, _risk_report, parents in bundles:
        for resolution in (250, 500):
            certificate, partition_report = certificate_bundle(
                unit,
                parents,
                resolution,
            )
            certificates[(unit.sha256, resolution)] = certificate
            partition_reports[(unit.sha256, resolution)] = partition_report
    registry = CellPartitionRegistry.build(
        tuple(certificates.values()),
        source_commit=p.source_commit,
    )
    rows = []
    attestations = []
    first_unit = p.expected_units[0].sha256
    for unit in p.expected_units:
        _same_unit, risk_report, parents = by_unit[unit.sha256]
        for arm_id in ("D0", "R0"):
            resolution = (
                250
                if mismatch and unit.sha256 == first_unit and arm_id == "R0"
                else 500
            )
            status = (
                "ROUTE_CATASTROPHIC_FALSE_SAFE"
                if catastrophic and unit.sha256 == first_unit and arm_id == "D0"
                else "ROUTE_SAFE"
            )
            row, attestation = route_cell(
                p,
                unit,
                risk_report,
                partition_reports[(unit.sha256, resolution)],
                parents,
                arm_id,
                certificates[(unit.sha256, resolution)],
                evaluation_status=status,
            )
            rows.append(row)
            attestations.append(attestation)
    report = GroupedComparisonReportV3.build(
        p,
        rows,
        verifier_commit=git("report-verifier"),
    )
    attestation_registry = RoutePartitionAttestationRegistryV2(
        preregistration_sha256=p.sha256,
        report_sha256=report.sha256(p),
        partition_registry_sha256=registry.sha256,
        attestations=tuple(sorted(attestations, key=lambda value: value.key)),
        source_commit=p.source_commit,
        verifier_commit=git("registry-verifier"),
    )
    return p, report, registry, attestation_registry


def test_report_derived_source_lineage_validates():
    p, report, registry, attestations = fixture()
    first = attestations.attestations[0].input_manifest.source_lineage
    assert first.risk_evidence_report.sha256 != (
        first.partition_source_report.sha256
    )
    attestations.validate(report, p, registry)
    build_grouped_diagnostics_v4(
        report,
        p,
        registry,
        attestations,
        verifier_commit=git("diagnostics-verifier"),
    )


def test_attestation_rejects_legacy_v2_manifest():
    p, report, registry, attestations = fixture()
    first = attestations.attestations[0]
    broken = dataclasses.replace(
        first,
        input_manifest=first.input_manifest.base,
    )
    with pytest.raises(
        RoutePartitionAttestationV2Error,
        match="requires report-derived source lineage v3",
    ):
        dataclasses.replace(
            attestations,
            attestations=(broken, *attestations.attestations[1:]),
        ).validate(report, p, registry)


def test_extra_certified_resolutions_may_remain_unused():
    p, report, registry, attestations = fixture()
    assert len(registry.certificates) == 2 * len(p.expected_units)
    attestations.validate(report, p, registry)


def test_run_must_name_exact_report_derived_input_manifest():
    p, report, registry, attestations = fixture()
    first = attestations.attestations[0]
    row = next(
        row
        for row in report.work_evaluations
        if first.key == (row.unit.sha256, row.arm_id)
    )
    selected = registry.by_key()[first.input_manifest.partition_key]
    changed = dataclasses.replace(
        first.input_manifest,
        base=dataclasses.replace(
            first.input_manifest.base,
            builder_commit=git("other-builder"),
        ),
    )
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="does not name the report-derived input manifest",
    ):
        changed.validate(
            p,
            row.unit,
            row.route_provenance.route_artifact.inference_run,
            selected,
        )


def test_same_shaped_alternative_partition_is_not_interchangeable():
    p, report, registry, attestations = fixture(mismatch=True)
    with pytest.raises(
        RoutePartitionAttestationV2Error,
        match="selected different certified partitions",
    ):
        attestations.validate(report, p, registry)


def test_route_shape_must_equal_selected_partition():
    p, report, registry, attestations = fixture()
    broken = dataclasses.replace(
        attestations.attestations[0],
        labels_shape=(1, 4),
    )
    with pytest.raises(
        RoutePartitionAttestationV2Error,
        match="label shape differs",
    ):
        dataclasses.replace(
            attestations,
            attestations=(broken, *attestations.attestations[1:]),
        ).validate(report, p, registry)


def test_failed_route_switches_are_checked_against_selected_partition():
    p, report, registry, attestations = fixture(catastrophic=True)
    attestations.validate(report, p, registry)
    with pytest.raises(
        GroupedDiagnosticsV4Error,
        match="temporal switches exceed selected partition",
    ):
        build_grouped_diagnostics_v4(
            report,
            p,
            registry,
            attestations,
            verifier_commit=git("diagnostics-verifier"),
        )
