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
from audio_extract.counterfactual_risk_grouped_diagnostics_v3 import (
    GeometryManifestV3,
    UnitPartitionBindingV3,
    build_grouped_diagnostics_v3,
)
from audio_extract.counterfactual_risk_route_partition_attestation_v1 import (
    CertifiedGroupedDiagnosticsEnvelopeV1,
    RoutePartitionAttestationError,
    RoutePartitionAttestationRegistryV1,
    RoutePartitionAttestationV1,
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
        experiment_id="D0_R0_ATTESTATION_V1",
        dataset_sha256=sha("dataset"),
        split_sha256=sha("split"),
        test_subset_sha256=sha("test-subset"),
        exact_evidence_manifest_sha256=sha("exact-manifest"),
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


def partition(held_out: HeldOutUnitV1) -> CellPartitionCertificate:
    entries = tuple(
        CellPartitionEntry(t, b, sha(f"cell-{held_out.work_id}-{t}-{b}"), RationalMeasure(1))
        for t in range(2)
        for b in range(2)
    )
    return CellPartitionCertificate(
        group_family_sha256=held_out.group_family_sha256,
        spectral_grid_sha256=sha("spectral-grid"),
        resolution_ms=500,
        time_cell_count=2,
        band_count=2,
        entries=entries,
        time_ranges_sha256=sha(f"time-ranges-{held_out.work_id}"),
        frequency_ranges_sha256=sha("frequency-ranges"),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=held_out.exact_evidence_sha256,
        verifier_commit=git("partition-verifier"),
    )


def geometry(
    p: GroupedComparisonPreregistrationV1,
) -> GeometryManifestV3:
    certificates = tuple(partition(row) for row in p.expected_units)
    registry = CellPartitionRegistry.build(
        certificates, source_commit=p.source_commit
    )
    bindings = tuple(
        sorted(
            (
                UnitPartitionBindingV3(
                    unit_sha256=held_out.sha256,
                    group_family_sha256=held_out.group_family_sha256,
                    spectral_grid_sha256=certificate.spectral_grid_sha256,
                    resolution_ms=certificate.resolution_ms,
                    partition_certificate_sha256=certificate.sha256,
                    exact_evidence_sha256=held_out.exact_evidence_sha256,
                    exact_panel_sha256=held_out.exact_panel_sha256,
                    exact_preflight_sha256=held_out.exact_preflight_sha256,
                    exact_oracle_decision_sha256=(
                        held_out.exact_oracle_decision_sha256
                    ),
                )
                for held_out, certificate in zip(
                    p.expected_units, certificates, strict=True
                )
            ),
            key=lambda row: row.unit_sha256,
        )
    )
    return GeometryManifestV3(
        p.sha256,
        registry,
        bindings,
        git("geometry-verifier"),
    )


def provenance(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    *,
    abstain: bool = False,
) -> ArmRouteProvenanceV3:
    frozen = p.d0_arm if arm_id == "D0" else p.r0_arm
    provisional = ArmRouteSubmissionV1(
        arm_id=arm_id,
        route_evidence_sha256=sha("placeholder"),
        source_status=("ABSTAIN_UNCERTAIN" if abstain else "ROUTE"),
        status=("ABSTAIN" if abstain else "ROUTE"),
        labels=(None if abstain else np.zeros((2, 2), dtype=np.int32)),
        reason=("uncertain" if abstain else "route"),
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
    submission = dataclasses.replace(
        provisional,
        route_evidence_sha256=artifact.sha256(p, held_out),
    )
    return ArmRouteProvenanceV3(artifact, submission)


def work_row(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    *,
    abstain: bool = False,
) -> WorkArmEvaluationV3:
    route = provenance(p, held_out, arm_id, abstain=abstain)
    status = "ABSTAIN" if abstain else "ROUTE_SAFE"
    evaluation = CompleteRouteEvaluationV1(
        arm_id=arm_id,
        exact_evidence_sha256=held_out.exact_evidence_sha256,
        exact_panel_sha256=held_out.exact_panel_sha256,
        exact_preflight_sha256=held_out.exact_preflight_sha256,
        policy_sha256=p.routing_policy_sha256,
        exact_oracle_decision_sha256=held_out.exact_oracle_decision_sha256,
        submission_sha256=route.submission.sha256,
        status=status,
        critical_false_safe_violations=(),
        incomplete_secondary_violations=(),
        exact_oracle_objective=1.0,
        submitted_exact_objective=(None if abstain else 1.0),
        selection_regret=(None if abstain else 0.0),
        temporal_switches=(None if abstain else 0),
        frequency_switches=(None if abstain else 0),
        reason=status,
    )
    return WorkArmEvaluationV3(held_out, route, evaluation)


def report(p: GroupedComparisonPreregistrationV1) -> GroupedComparisonReportV3:
    return GroupedComparisonReportV3.build(
        p,
        [
            work_row(p, held_out, arm_id)
            for held_out in p.expected_units
            for arm_id in ("D0", "R0")
        ],
        verifier_commit=git("report-verifier"),
    )


def attestation(
    p: GroupedComparisonPreregistrationV1,
    row: WorkArmEvaluationV3,
    certificate: CellPartitionCertificate,
) -> RoutePartitionAttestationV1:
    artifact = row.route_provenance.route_artifact
    output = artifact.route_output
    submission = row.route_provenance.submission
    return RoutePartitionAttestationV1(
        preregistration_sha256=p.sha256,
        unit_sha256=row.unit.sha256,
        arm_id=row.arm_id,
        route_artifact_sha256=artifact.sha256(p, row.unit),
        route_output_sha256=output.sha256(row.unit, row.arm_id),
        submission_sha256=submission.sha256,
        inference_input_manifest_sha256=(
            artifact.inference_run.inference_input_manifest_sha256
        ),
        partition_certificate_sha256=certificate.sha256,
        spectral_grid_sha256=certificate.spectral_grid_sha256,
        resolution_ms=certificate.resolution_ms,
        time_cell_count=certificate.time_cell_count,
        band_count=certificate.band_count,
        labels_shape=output.labels_shape,
        labels_sha256=output.labels_sha256,
        source_commit=p.source_commit,
        verifier_commit=git("attestation-verifier"),
    )


def fixture():
    p = prereg(unit("one"), unit("two"))
    grouped = report(p)
    grid = geometry(p)
    partitions = grid.resolved(p)
    attestations = tuple(
        sorted(
            (
                attestation(p, row, partitions[row.unit.sha256])
                for row in grouped.work_evaluations
            ),
            key=lambda row: row.key,
        )
    )
    registry = RoutePartitionAttestationRegistryV1(
        preregistration_sha256=p.sha256,
        report_sha256=grouped.sha256(p),
        geometry_manifest_sha256=grid.sha256(p),
        attestations=attestations,
        source_commit=p.source_commit,
        verifier_commit=git("registry-verifier"),
    )
    diagnostics = build_grouped_diagnostics_v3(
        grouped,
        p,
        grid,
        verifier_commit=git("diagnostics-verifier"),
    )
    envelope = CertifiedGroupedDiagnosticsEnvelopeV1(
        report_sha256=grouped.sha256(p),
        geometry_manifest_sha256=grid.sha256(p),
        diagnostics_sha256=diagnostics.sha256(grouped, p, grid),
        attestation_registry_sha256=registry.sha256(grouped, p, grid),
        verifier_commit=git("envelope-verifier"),
    )
    return p, grouped, grid, diagnostics, registry, envelope


def test_valid_attestation_registry_and_nonpromoting_envelope():
    p, grouped, grid, diagnostics, registry, envelope = fixture()
    registry.validate(grouped, p, grid)
    envelope.validate(grouped, p, grid, diagnostics, registry)
    assert envelope.identity_dict(
        grouped, p, grid, diagnostics, registry
    )["promotion_decision"] is None


def test_route_shape_must_equal_certified_partition_shape():
    p, grouped, grid, _diagnostics, registry, _envelope = fixture()
    broken = dataclasses.replace(
        registry.attestations[0], labels_shape=(1, 4)
    )
    with pytest.raises(
        RoutePartitionAttestationError,
        match="label shape differs",
    ):
        dataclasses.replace(
            registry,
            attestations=(broken, *registry.attestations[1:]),
        ).validate(grouped, p, grid)


def test_route_cannot_be_attested_to_another_partition_certificate():
    p, grouped, grid, _diagnostics, registry, _envelope = fixture()
    first = registry.attestations[0]
    partitions = grid.resolved(p)
    other_unit = next(
        unit_sha for unit_sha in partitions if unit_sha != first.unit_sha256
    )
    other = partitions[other_unit]
    broken = dataclasses.replace(
        first,
        partition_certificate_sha256=other.sha256,
        spectral_grid_sha256=other.spectral_grid_sha256,
        resolution_ms=other.resolution_ms,
        time_cell_count=other.time_cell_count,
        band_count=other.band_count,
    )
    with pytest.raises(RoutePartitionAttestationError):
        dataclasses.replace(
            registry,
            attestations=(broken, *registry.attestations[1:]),
        ).validate(grouped, p, grid)


def test_changed_inference_input_manifest_invalidates_attestation():
    p, grouped, grid, _diagnostics, registry, _envelope = fixture()
    broken = dataclasses.replace(
        registry.attestations[0],
        inference_input_manifest_sha256=sha("different-inputs"),
    )
    with pytest.raises(
        RoutePartitionAttestationError,
        match="inference_input_manifest_sha256",
    ):
        dataclasses.replace(
            registry,
            attestations=(broken, *registry.attestations[1:]),
        ).validate(grouped, p, grid)


def test_registry_must_exactly_cover_every_report_cell_once():
    p, grouped, grid, _diagnostics, registry, _envelope = fixture()
    with pytest.raises(
        RoutePartitionAttestationError,
        match="does not exactly cover report cells",
    ):
        dataclasses.replace(
            registry,
            attestations=registry.attestations[:-1],
        ).validate(grouped, p, grid)


def test_envelope_detects_diagnostics_or_registry_substitution():
    p, grouped, grid, diagnostics, registry, envelope = fixture()
    with pytest.raises(
        RoutePartitionAttestationError,
        match="certified diagnostics envelope differs",
    ):
        dataclasses.replace(
            envelope,
            diagnostics_sha256=sha("other-diagnostics"),
        ).validate(grouped, p, grid, diagnostics, registry)
