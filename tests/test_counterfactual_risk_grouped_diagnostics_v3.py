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
from audio_extract.counterfactual_risk_grouped_diagnostics_v3 import (
    GeometryManifestV3,
    GroupedDiagnosticsV3Error,
    UnitPartitionBindingV3,
    build_grouped_diagnostics_v3,
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
        experiment_id="D0_R0_DIAGNOSTICS_V3",
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


def provenance(
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
    route = provenance(p, held_out, arm_id)
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
            rows.append(
                record(
                    p,
                    held_out,
                    arm_id,
                    **override.get((held_out.work_id, arm_id), {}),
                )
            )
    return GroupedComparisonReportV3.build(
        p, rows, verifier_commit=git("report-verifier")
    )


def partition(
    held_out: HeldOutUnitV1,
    *,
    resolution_ms: int = 500,
    time_cells: int = 2,
    bands: int = 2,
) -> CellPartitionCertificate:
    entries = tuple(
        CellPartitionEntry(
            time_index,
            band_index,
            sha(f"cell-{held_out.work_id}-{time_index}-{band_index}"),
            RationalMeasure(1),
        )
        for time_index in range(time_cells)
        for band_index in range(bands)
    )
    return CellPartitionCertificate(
        group_family_sha256=held_out.group_family_sha256,
        spectral_grid_sha256=sha("spectral-grid"),
        resolution_ms=resolution_ms,
        time_cell_count=time_cells,
        band_count=bands,
        entries=entries,
        time_ranges_sha256=sha(f"time-ranges-{held_out.work_id}"),
        frequency_ranges_sha256=sha("frequency-ranges"),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=held_out.exact_evidence_sha256,
        verifier_commit=git("partition-verifier"),
    )


def geometry(
    p: GroupedComparisonPreregistrationV1,
    *,
    time_cells: int = 2,
    bands: int = 2,
) -> GeometryManifestV3:
    certificates = tuple(
        partition(held_out, time_cells=time_cells, bands=bands)
        for held_out in p.expected_units
    )
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
        preregistration_sha256=p.sha256,
        partition_registry=registry,
        bindings=bindings,
        verifier_commit=git("geometry-verifier"),
    )


def test_v3_diagnostics_build_from_complete_cell_partition_registry():
    p = prereg(unit("one"), unit("two"))
    grouped = report(p)
    grid = geometry(p)
    result = build_grouped_diagnostics_v3(
        grouped,
        p,
        grid,
        verifier_commit=git("diagnostics-verifier"),
    )
    result.validate(grouped, p, grid)
    payload = result.identity_dict(grouped, p, grid)
    assert payload["promotion_decision"] is None
    assert payload["schema"].endswith("grouped-diagnostics/v3")


def test_binding_rejects_another_certificate_for_the_same_unit():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    grid = geometry(p)
    binding = next(
        row for row in grid.bindings if row.unit_sha256 == held_out.sha256
    )
    certificate = grid.partition_registry.by_key()[binding.partition_key]
    broken = dataclasses.replace(
        binding,
        partition_certificate_sha256=sha("different-certificate"),
    )
    with pytest.raises(
        GroupedDiagnosticsV3Error,
        match="another certificate",
    ):
        broken.validate(held_out, certificate)


def test_partition_source_report_must_match_unit_exact_evidence():
    held_out = unit("one")
    good = partition(held_out)
    bad = dataclasses.replace(
        good,
        exact_source_report_sha256=sha("other-exact-report"),
    )
    binding = UnitPartitionBindingV3(
        unit_sha256=held_out.sha256,
        group_family_sha256=held_out.group_family_sha256,
        spectral_grid_sha256=bad.spectral_grid_sha256,
        resolution_ms=bad.resolution_ms,
        partition_certificate_sha256=bad.sha256,
        exact_evidence_sha256=held_out.exact_evidence_sha256,
        exact_panel_sha256=held_out.exact_panel_sha256,
        exact_preflight_sha256=held_out.exact_preflight_sha256,
        exact_oracle_decision_sha256=held_out.exact_oracle_decision_sha256,
    )
    with pytest.raises(
        GroupedDiagnosticsV3Error,
        match="another exact source report",
    ):
        binding.validate(held_out, bad)


def test_registry_may_not_contain_unbound_extra_partition():
    one = unit("one")
    two = unit("two")
    p = prereg(one, two)
    grid = geometry(p)
    extra_unit = unit("extra")
    registry = CellPartitionRegistry.build(
        (*grid.partition_registry.certificates, partition(extra_unit)),
        source_commit=p.source_commit,
    )
    with pytest.raises(
        GroupedDiagnosticsV3Error,
        match="unused or unbound",
    ):
        dataclasses.replace(grid, partition_registry=registry).validate(p)


def test_registry_source_commit_is_bound_to_preregistration():
    p = prereg(unit("one"), unit("two"))
    grid = geometry(p)
    broken = CellPartitionRegistry(
        grid.partition_registry.certificates,
        git("other-source"),
    )
    with pytest.raises(
        GroupedDiagnosticsV3Error,
        match="source commit differs",
    ):
        dataclasses.replace(grid, partition_registry=broken).validate(p)


def test_catastrophic_route_cannot_exceed_certified_boundaries():
    p = prereg(unit("one"), unit("two"))
    grouped = report(
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
        GroupedDiagnosticsV3Error,
        match="temporal switches exceed certified boundaries",
    ):
        build_grouped_diagnostics_v3(
            grouped,
            p,
            grid,
            verifier_commit=git("diagnostics-verifier"),
        )


def test_complete_cell_entries_change_geometry_identity():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    grid = geometry(p)
    first = grid.sha256(p)
    certificates = list(grid.partition_registry.certificates)
    index = next(
        i for i, row in enumerate(certificates)
        if row.group_family_sha256 == held_out.group_family_sha256
    )
    certificate = certificates[index]
    entries = list(certificate.entries)
    entries[0] = dataclasses.replace(
        entries[0], cell_sha256=sha("changed-cell")
    )
    certificates[index] = dataclasses.replace(
        certificate, entries=tuple(entries)
    )
    registry = CellPartitionRegistry.build(
        certificates, source_commit=p.source_commit
    )
    bindings = tuple(
        dataclasses.replace(
            binding,
            partition_certificate_sha256=(
                registry.by_key()[binding.partition_key].sha256
            ),
        )
        for binding in grid.bindings
    )
    second = dataclasses.replace(
        grid,
        partition_registry=registry,
        bindings=bindings,
    )
    assert second.sha256(p) != first
