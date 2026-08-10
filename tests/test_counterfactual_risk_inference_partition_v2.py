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
    CertifiedGroupedDiagnosticsEnvelopeV2,
    GroupedDiagnosticsV4Error,
    build_grouped_diagnostics_v4,
)
from audio_extract.counterfactual_risk_inference_partition_manifest_v2 import (
    ExactSourceLineageV2,
    InferenceInputPartitionManifestV2,
    InferencePartitionManifestError,
)
from audio_extract.counterfactual_risk_route_partition_attestation_v2 import (
    RoutePartitionAttestationRegistryV2,
    RoutePartitionAttestationV2,
    RoutePartitionAttestationV2Error,
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
        exact_evidence_sha256=sha(f"risk-evidence-{name}"),
        exact_panel_sha256=sha(f"panel-{name}"),
        exact_preflight_sha256=sha(f"preflight-{name}"),
        exact_oracle_decision_sha256=sha(f"oracle-{name}"),
    )


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


def prereg(*units: HeldOutUnitV1) -> GroupedComparisonPreregistrationV1:
    return GroupedComparisonPreregistrationV1(
        experiment_id="D0_R0_RESOLVED_PARTITION_V2",
        dataset_sha256=sha("dataset"),
        split_sha256=sha("split"),
        test_subset_sha256=sha("test-subset"),
        exact_evidence_manifest_sha256=sha("exact-evidence-manifest"),
        candidate_panel_sha256=sha("candidate-panel"),
        feature_contract_sha256=sha("feature-contract"),
        metric_contract_sha256=sha("metric-contract"),
        routing_policy_sha256=sha("routing-policy"),
        complete_route_evaluation_contract_sha256=sha("evaluation-contract"),
        comparison_policy_sha256=sha("comparison-policy"),
        promotion_policy_sha256=sha("separate-promotion-policy"),
        paired_regret_tolerance=1e-8,
        d0_arm=comparison_arm("D0"),
        r0_arm=comparison_arm("R0"),
        expected_units=tuple(sorted(units, key=lambda row: row.sha256)),
        source_commit=git("source"),
    )


def partition(
    held_out: HeldOutUnitV1,
    *,
    resolution_ms: int,
    time_count: int = 2,
    band_count: int = 2,
) -> CellPartitionCertificate:
    entries = tuple(
        CellPartitionEntry(
            t,
            b,
            sha(
                f"cell-{held_out.work_id}-{resolution_ms}-{t}-{b}"
            ),
            RationalMeasure(1),
        )
        for t in range(time_count)
        for b in range(band_count)
    )
    return CellPartitionCertificate(
        group_family_sha256=held_out.group_family_sha256,
        spectral_grid_sha256=sha(f"grid-{resolution_ms}"),
        resolution_ms=resolution_ms,
        time_cell_count=time_count,
        band_count=band_count,
        entries=entries,
        time_ranges_sha256=sha(
            f"time-ranges-{held_out.work_id}-{resolution_ms}"
        ),
        frequency_ranges_sha256=sha(f"frequency-ranges-{resolution_ms}"),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=sha(
            f"partition-source-report-{held_out.work_id}"
        ),
        verifier_commit=git("partition-verifier"),
    )


def source_lineage(
    held_out: HeldOutUnitV1,
    certificate: CellPartitionCertificate,
) -> ExactSourceLineageV2:
    return ExactSourceLineageV2(
        unit_sha256=held_out.sha256,
        group_family_sha256=held_out.group_family_sha256,
        exact_evidence_sha256=held_out.exact_evidence_sha256,
        exact_source_report_sha256=(
            certificate.exact_source_report_sha256
        ),
        truth_manifest_sha256=sha(f"truth-{held_out.work_id}"),
        mixture_pcm_sha256=sha(f"mixture-{held_out.work_id}"),
        accompaniment_pcm_sha256=sha(
            f"accompaniment-{held_out.work_id}"
        ),
        vocal_pcm_sha256=sha(f"vocal-{held_out.work_id}"),
        exact_evidence_builder_commit=git("evidence-builder"),
        partition_source_verifier_commit=certificate.verifier_commit,
        lineage_verifier_commit=git("lineage-verifier"),
    )


def input_manifest(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    certificate: CellPartitionCertificate,
) -> InferenceInputPartitionManifestV2:
    frozen = p.d0_arm if arm_id == "D0" else p.r0_arm
    return InferenceInputPartitionManifestV2(
        preregistration_sha256=p.sha256,
        unit_sha256=held_out.sha256,
        group_family_sha256=held_out.group_family_sha256,
        query_condition_sha256=held_out.query_condition_sha256,
        query_quality_stratum_sha256=(
            held_out.query_quality_stratum_sha256
        ),
        arm_id=arm_id,
        arm_run_sha256=frozen.sha256,
        dataset_sha256=p.dataset_sha256,
        split_sha256=p.split_sha256,
        test_subset_sha256=p.test_subset_sha256,
        candidate_panel_sha256=p.candidate_panel_sha256,
        feature_contract_sha256=p.feature_contract_sha256,
        metric_contract_sha256=p.metric_contract_sha256,
        partition_certificate_sha256=certificate.sha256,
        spectral_grid_sha256=certificate.spectral_grid_sha256,
        resolution_ms=certificate.resolution_ms,
        time_ranges_sha256=certificate.time_ranges_sha256,
        frequency_ranges_sha256=certificate.frequency_ranges_sha256,
        measure_contract_sha256=certificate.measure_contract_sha256,
        source_lineage=source_lineage(held_out, certificate),
        builder_commit=git("input-builder"),
    )


def provenance(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    certificate: CellPartitionCertificate,
    *,
    status: str = "ROUTE",
) -> tuple[ArmRouteProvenanceV3, InferenceInputPartitionManifestV2]:
    frozen = p.d0_arm if arm_id == "D0" else p.r0_arm
    manifest = input_manifest(p, held_out, arm_id, certificate)
    provisional = ArmRouteSubmissionV1(
        arm_id=arm_id,
        route_evidence_sha256=sha("placeholder"),
        source_status=("ROUTE" if status == "ROUTE" else "ABSTAIN_UNCERTAIN"),
        status=status,
        labels=(
            np.zeros(
                (certificate.time_cell_count, certificate.band_count),
                dtype=np.int32,
            )
            if status == "ROUTE"
            else None
        ),
        reason=status,
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
        inference_input_manifest_sha256=manifest.sha256(
            p, held_out, certificate
        ),
        runner_bundle_sha256=sha(f"runner-{arm_id}"),
        source_commit=frozen.source_commit,
        runner_commit=git(f"runner-{arm_id}"),
    )
    artifact = RouteArtifactManifestV3(
        inference_run=run,
        route_output=RouteOutputV3.from_submission(
            held_out, provisional
        ),
        verifier_commit=git(f"artifact-verifier-{arm_id}"),
    )
    submission = dataclasses.replace(
        provisional,
        route_evidence_sha256=artifact.sha256(p, held_out),
    )
    return ArmRouteProvenanceV3(artifact, submission), manifest


def evaluation(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    provenance_value: ArmRouteProvenanceV3,
    *,
    status: str = "ROUTE_SAFE",
    temporal: int = 0,
    frequency: int = 0,
) -> CompleteRouteEvaluationV1:
    route_status = status.startswith("ROUTE_")
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
        if status == "ROUTE_CATASTROPHIC_FALSE_SAFE"
        else ()
    )
    incomplete = (
        (
            SelectedRiskViolationV1(
                0,
                0,
                0,
                "artifact",
                "required_secondary_missing",
                None,
                None,
            ),
        )
        if status == "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE"
        else ()
    )
    return CompleteRouteEvaluationV1(
        arm_id=provenance_value.arm_id,
        exact_evidence_sha256=held_out.exact_evidence_sha256,
        exact_panel_sha256=held_out.exact_panel_sha256,
        exact_preflight_sha256=held_out.exact_preflight_sha256,
        policy_sha256=p.routing_policy_sha256,
        exact_oracle_decision_sha256=(
            held_out.exact_oracle_decision_sha256
        ),
        submission_sha256=provenance_value.submission.sha256,
        status=status,
        critical_false_safe_violations=critical,
        incomplete_secondary_violations=incomplete,
        exact_oracle_objective=1.0,
        submitted_exact_objective=(1.0 if status == "ROUTE_SAFE" else None),
        selection_regret=(0.0 if status == "ROUTE_SAFE" else None),
        temporal_switches=(temporal if route_status else None),
        frequency_switches=(frequency if route_status else None),
        reason=status,
    )


def fixture():
    p = prereg(unit("one"), unit("two"))
    selected = {
        held_out.sha256: partition(held_out, resolution_ms=500)
        for held_out in p.expected_units
    }
    extras = tuple(
        partition(held_out, resolution_ms=250, time_count=4)
        for held_out in p.expected_units
    )
    registry = CellPartitionRegistry.build(
        tuple(selected.values()) + extras,
        source_commit=p.source_commit,
    )
    rows = []
    manifests = {}
    for held_out in p.expected_units:
        for arm_id in ("D0", "R0"):
            route, manifest = provenance(
                p,
                held_out,
                arm_id,
                selected[held_out.sha256],
            )
            rows.append(
                WorkArmEvaluationV3(
                    held_out,
                    route,
                    evaluation(p, held_out, route),
                )
            )
            manifests[(held_out.sha256, arm_id)] = manifest
    grouped = GroupedComparisonReportV3.build(
        p, rows, verifier_commit=git("report-verifier")
    )
    by_key = {
        (row.unit.sha256, row.arm_id): row
        for row in grouped.work_evaluations
    }
    attestations = []
    for key, row in by_key.items():
        certificate = selected[row.unit.sha256]
        artifact = row.route_provenance.route_artifact
        output = artifact.route_output
        submission = row.route_provenance.submission
        attestations.append(
            RoutePartitionAttestationV2(
                input_manifest=manifests[key],
                route_artifact_sha256=artifact.sha256(p, row.unit),
                route_output_sha256=output.sha256(row.unit, row.arm_id),
                submission_sha256=submission.sha256,
                partition_certificate_sha256=certificate.sha256,
                labels_shape=output.labels_shape,
                labels_sha256=output.labels_sha256,
                verifier_commit=git("attestation-verifier"),
            )
        )
    attestation_registry = RoutePartitionAttestationRegistryV2(
        preregistration_sha256=p.sha256,
        report_sha256=grouped.sha256(p),
        partition_registry_sha256=registry.sha256,
        attestations=tuple(sorted(attestations, key=lambda row: row.key)),
        source_commit=p.source_commit,
        verifier_commit=git("registry-verifier"),
    )
    diagnostics = build_grouped_diagnostics_v4(
        grouped,
        p,
        registry,
        attestation_registry,
        verifier_commit=git("diagnostics-verifier"),
    )
    envelope = CertifiedGroupedDiagnosticsEnvelopeV2(
        report_sha256=grouped.sha256(p),
        attestation_registry_sha256=attestation_registry.sha256(
            grouped, p, registry
        ),
        diagnostics_sha256=diagnostics.sha256(
            grouped, p, registry, attestation_registry
        ),
        verifier_commit=git("envelope-verifier"),
    )
    return p, grouped, registry, attestation_registry, diagnostics, envelope


def test_distinct_source_report_and_risk_evidence_are_linked_not_equal():
    p, grouped, registry, attestations, diagnostics, envelope = fixture()
    first = attestations.attestations[0]
    assert (
        first.input_manifest.source_lineage.exact_source_report_sha256
        != first.input_manifest.source_lineage.exact_evidence_sha256
    )
    attestations.validate(grouped, p, registry)
    diagnostics.validate(grouped, p, registry, attestations)
    envelope.validate(grouped, p, registry, attestations, diagnostics)


def test_unused_certified_resolutions_are_allowed():
    p, grouped, registry, attestations, _diagnostics, _envelope = fixture()
    assert len(registry.certificates) == 2 * len(p.expected_units)
    assert len(attestations.attestations) == 2 * len(p.expected_units)
    attestations.validate(grouped, p, registry)


def test_run_must_name_the_exact_content_bearing_input_manifest():
    p, grouped, registry, attestations, _diagnostics, _envelope = fixture()
    first = attestations.attestations[0]
    row = next(row for row in grouped.work_evaluations if (
        row.unit.sha256, row.arm_id
    ) == first.key)
    certificate = registry.by_key()[first.input_manifest.partition_key]
    changed = dataclasses.replace(
        first.input_manifest,
        builder_commit=git("another-builder"),
    )
    with pytest.raises(
        InferencePartitionManifestError,
        match="does not name this input/partition manifest",
    ):
        changed.validate(
            p,
            row.unit,
            row.route_provenance.route_artifact.inference_run,
            certificate,
        )


def test_same_shaped_alternative_partition_cannot_be_substituted():
    p, grouped, registry, attestations, _diagnostics, _envelope = fixture()
    first = attestations.attestations[0]
    row = next(row for row in grouped.work_evaluations if (
        row.unit.sha256, row.arm_id
    ) == first.key)
    alternative = partition(
        row.unit,
        resolution_ms=750,
        time_count=2,
        band_count=2,
    )
    changed_manifest = dataclasses.replace(
        first.input_manifest,
        partition_certificate_sha256=alternative.sha256,
        spectral_grid_sha256=alternative.spectral_grid_sha256,
        resolution_ms=alternative.resolution_ms,
        time_ranges_sha256=alternative.time_ranges_sha256,
        frequency_ranges_sha256=alternative.frequency_ranges_sha256,
        measure_contract_sha256=alternative.measure_contract_sha256,
        source_lineage=source_lineage(row.unit, alternative),
    )
    changed = dataclasses.replace(
        first,
        input_manifest=changed_manifest,
        partition_certificate_sha256=alternative.sha256,
    )
    with pytest.raises(
        InferencePartitionManifestError,
        match="does not name this input/partition manifest",
    ):
        changed.validate(p, row, alternative)


def test_d0_and_r0_must_select_the_same_partition_for_a_unit():
    p, grouped, registry, attestations, _diagnostics, _envelope = fixture()
    values = list(attestations.attestations)
    first = values[0]
    same_unit_other = next(
        index
        for index, row in enumerate(values)
        if row.unit_sha256 == first.unit_sha256 and row.arm_id != first.arm_id
    )
    row = next(
        item
        for item in grouped.work_evaluations
        if (item.unit.sha256, item.arm_id) == values[same_unit_other].key
    )
    alternative = next(
        cert
        for cert in registry.certificates
        if cert.group_family_sha256 == row.unit.group_family_sha256
        and cert.resolution_ms != first.input_manifest.resolution_ms
    )
    changed_manifest = input_manifest(p, row.unit, row.arm_id, alternative)
    run = dataclasses.replace(
        row.route_provenance.route_artifact.inference_run,
        inference_input_manifest_sha256=changed_manifest.sha256(
            p, row.unit, alternative
        ),
    )
    artifact = dataclasses.replace(
        row.route_provenance.route_artifact,
        inference_run=run,
    )
    provisional = row.route_provenance.submission
    changed_submission = dataclasses.replace(
        provisional,
        route_evidence_sha256=artifact.sha256(p, row.unit),
    )
    changed_route = dataclasses.replace(
        row.route_provenance,
        route_artifact=artifact,
        submission=changed_submission,
    )
    changed_evaluation = dataclasses.replace(
        row.evaluation,
        submission_sha256=changed_submission.sha256,
    )
    changed_row = WorkArmEvaluationV3(
        row.unit, changed_route, changed_evaluation
    )
    changed_report_rows = tuple(
        changed_row if item is row else item
        for item in grouped.work_evaluations
    )
    changed_report = GroupedComparisonReportV3.build(
        p,
        changed_report_rows,
        verifier_commit=grouped.verifier_commit,
    )
    values[same_unit_other] = RoutePartitionAttestationV2(
        input_manifest=changed_manifest,
        route_artifact_sha256=artifact.sha256(p, row.unit),
        route_output_sha256=artifact.route_output.sha256(
            row.unit, row.arm_id
        ),
        submission_sha256=changed_submission.sha256,
        partition_certificate_sha256=alternative.sha256,
        labels_shape=artifact.route_output.labels_shape,
        labels_sha256=artifact.route_output.labels_sha256,
        verifier_commit=git("attestation-verifier"),
    )
    changed_registry = dataclasses.replace(
        attestations,
        report_sha256=changed_report.sha256(p),
        attestations=tuple(sorted(values, key=lambda item: item.key)),
    )
    with pytest.raises(
        RoutePartitionAttestationV2Error,
        match="selected different certified partitions",
    ):
        changed_registry.validate(changed_report, p, registry)


def test_attestation_rejects_route_shape_different_from_selected_grid():
    p, grouped, registry, attestations, _diagnostics, _envelope = fixture()
    broken = dataclasses.replace(
        attestations.attestations[0], labels_shape=(1, 4)
    )
    with pytest.raises(
        RoutePartitionAttestationV2Error,
        match="label shape differs",
    ):
        dataclasses.replace(
            attestations,
            attestations=(broken, *attestations.attestations[1:]),
        ).validate(grouped, p, registry)


def test_diagnostics_reject_impossible_switch_count_for_failed_route():
    p, grouped, registry, attestations, _diagnostics, _envelope = fixture()
    target = grouped.work_evaluations[0]
    broken_evaluation = evaluation(
        p,
        target.unit,
        target.route_provenance,
        status="ROUTE_CATASTROPHIC_FALSE_SAFE",
        temporal=3,
        frequency=0,
    )
    broken_row = dataclasses.replace(
        target, evaluation=broken_evaluation
    )
    broken_rows = tuple(
        broken_row if row is target else row
        for row in grouped.work_evaluations
    )
    broken_report = GroupedComparisonReportV3.build(
        p,
        broken_rows,
        verifier_commit=grouped.verifier_commit,
    )
    broken_attestations = dataclasses.replace(
        attestations,
        report_sha256=broken_report.sha256(p),
    )
    with pytest.raises(
        GroupedDiagnosticsV4Error,
        match="temporal switches exceed",
    ):
        build_grouped_diagnostics_v4(
            broken_report,
            p,
            registry,
            broken_attestations,
            verifier_commit=git("diagnostics-verifier"),
        )
