import dataclasses
import hashlib

import pytest

from audio_extract.counterfactual_risk_complete_route_evaluation_v1 import (
    CompleteRouteEvaluationV1,
)
from audio_extract.counterfactual_risk_grouped_comparison_v1 import (
    ComparisonArmV1,
    GroupedComparisonPreregistrationV1,
    GroupedComparisonReportV1,
    HeldOutUnitV1,
    WorkArmEvaluationV1,
)
from audio_extract.counterfactual_risk_grouped_diagnostics_v1 import (
    GeometryManifestV1,
    GroupedDiagnosticsError,
    UnitRoutingGeometryV1,
    build_grouped_diagnostics_v1,
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
        source_family_sha256=sha(f"family-{name}"),
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
        compute_budget_sha256=sha("compute"),
        checkpoint_rule_sha256=sha("checkpoint"),
        inference_policy_sha256=sha(f"inference-{arm_id}"),
        source_commit=git("source"),
    )


def prereg() -> GroupedComparisonPreregistrationV1:
    units = tuple(sorted((unit("one"), unit("two")), key=lambda row: row.sha256))
    return GroupedComparisonPreregistrationV1(
        experiment_id="D0_R0_DIAGNOSTICS_V1",
        dataset_sha256=sha("dataset"),
        split_sha256=sha("split"),
        test_subset_sha256=sha("test"),
        exact_evidence_manifest_sha256=sha("evidence-manifest"),
        candidate_panel_sha256=sha("candidate-panel"),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        routing_policy_sha256=sha("policy"),
        complete_route_evaluation_contract_sha256=sha("evaluator"),
        comparison_policy_sha256=sha("comparison"),
        promotion_policy_sha256=sha("promotion"),
        paired_regret_tolerance=1e-8,
        d0_arm=arm("D0"),
        r0_arm=arm("R0"),
        expected_units=units,
        source_commit=git("source"),
    )


def evaluation(
    held_out,
    arm_id,
    *,
    status="ROUTE_SAFE",
    regret=0.0,
    temporal=0,
    frequency=0,
):
    route = status.startswith("ROUTE_")
    return CompleteRouteEvaluationV1(
        arm_id=arm_id,
        exact_evidence_sha256=held_out.exact_evidence_sha256,
        exact_panel_sha256=held_out.exact_panel_sha256,
        exact_preflight_sha256=held_out.exact_preflight_sha256,
        policy_sha256=sha("policy"),
        exact_oracle_decision_sha256=held_out.exact_oracle_decision_sha256,
        submission_sha256=sha(f"submission-{held_out.work_id}-{arm_id}-{status}"),
        status=status,
        critical_false_safe_violations=(),
        incomplete_secondary_violations=(),
        exact_oracle_objective=(
            None if status == "ORACLE_UNAVAILABLE" else 1.0
        ),
        submitted_exact_objective=(1.0 + regret if status == "ROUTE_SAFE" else None),
        selection_regret=(regret if status == "ROUTE_SAFE" else None),
        temporal_switches=(temporal if route else None),
        frequency_switches=(frequency if route else None),
        reason=f"{arm_id} {status}",
    )


def record(p, held_out, arm_id, **kwargs):
    return WorkArmEvaluationV1(
        unit=held_out,
        arm_run_sha256=(p.d0_arm.sha256 if arm_id == "D0" else p.r0_arm.sha256),
        dataset_sha256=p.dataset_sha256,
        split_sha256=p.split_sha256,
        test_subset_sha256=p.test_subset_sha256,
        candidate_panel_sha256=p.candidate_panel_sha256,
        feature_contract_sha256=p.feature_contract_sha256,
        metric_contract_sha256=p.metric_contract_sha256,
        preregistration_sha256=p.sha256,
        evaluation=evaluation(held_out, arm_id, **kwargs),
    )


def report_and_geometry():
    p = prereg()
    by_name = {row.work_id: row for row in p.expected_units}
    records = [
        record(p, by_name["one"], "D0", temporal=2, frequency=1, regret=0.1),
        record(p, by_name["one"], "R0", temporal=1, frequency=2, regret=0.05),
        record(p, by_name["two"], "D0", status="ABSTAIN"),
        record(p, by_name["two"], "R0", temporal=3, frequency=4, regret=0.2),
    ]
    report = GroupedComparisonReportV1.build(
        p, records, verifier_commit=git("report-verifier")
    )
    geometry = GeometryManifestV1(
        preregistration_sha256=p.sha256,
        units=tuple(
            sorted(
                (
                    UnitRoutingGeometryV1(
                        unit_sha256=by_name["one"].sha256,
                        spectral_grid_sha256=sha("grid-one"),
                        partition_certificate_sha256=sha("partition-one"),
                        time_cell_count=3,
                        band_count=2,
                    ),
                    UnitRoutingGeometryV1(
                        unit_sha256=by_name["two"].sha256,
                        spectral_grid_sha256=sha("grid-two"),
                        partition_certificate_sha256=sha("partition-two"),
                        time_cell_count=2,
                        band_count=3,
                    ),
                ),
                key=lambda row: row.unit_sha256,
            )
        ),
    )
    return p, report, geometry


def test_normalized_switching_and_status_dominance():
    p, report, geometry = report_and_geometry()
    result = build_grouped_diagnostics_v1(
        report, p, geometry, verifier_commit=git("diagnostics")
    )
    result.validate(report, p, geometry)
    assert result.d0_switching.safe_unit_count == 1
    assert result.d0_switching.temporal_switch_rate == pytest.approx(2 / 4)
    assert result.d0_switching.frequency_switch_rate == pytest.approx(1 / 3)
    assert result.r0_switching.safe_unit_count == 2
    assert result.r0_switching.temporal_switch_rate == pytest.approx(4 / 7)
    assert result.r0_switching.frequency_switch_rate == pytest.approx(6 / 7)
    assert result.status_dominance.r0_status_better_count == 1
    assert result.status_dominance.equal_status_count == 1
    payload = result.identity_dict(report, p, geometry)
    assert payload["promotion_decision"] is None


def test_geometry_manifest_must_exactly_cover_units():
    p, report, geometry = report_and_geometry()
    with pytest.raises(GroupedDiagnosticsError, match="exactly cover"):
        build_grouped_diagnostics_v1(
            report,
            p,
            dataclasses.replace(geometry, units=geometry.units[:-1]),
            verifier_commit=git("diagnostics"),
        )
    with pytest.raises(GroupedDiagnosticsError, match="duplicates"):
        build_grouped_diagnostics_v1(
            report,
            p,
            dataclasses.replace(
                geometry, units=(geometry.units[0], geometry.units[0])
            ),
            verifier_commit=git("diagnostics"),
        )


def test_switch_count_cannot_exceed_possible_boundaries():
    p, report, geometry = report_and_geometry()
    records = list(report.work_evaluations)
    index = next(
        i for i, row in enumerate(records)
        if row.unit.work_id == "one" and row.arm_id == "D0"
    )
    records[index] = dataclasses.replace(
        records[index],
        evaluation=dataclasses.replace(
            records[index].evaluation, temporal_switches=5
        ),
    )
    broken = GroupedComparisonReportV1.build(
        p, records, verifier_commit=git("report-verifier")
    )
    with pytest.raises(GroupedDiagnosticsError, match="exceed unit boundaries"):
        build_grouped_diagnostics_v1(
            broken, p, geometry, verifier_commit=git("diagnostics")
        )


def test_diagnostics_tampering_is_detected():
    p, report, geometry = report_and_geometry()
    result = build_grouped_diagnostics_v1(
        report, p, geometry, verifier_commit=git("diagnostics")
    )
    broken = dataclasses.replace(
        result,
        r0_switching=dataclasses.replace(
            result.r0_switching, temporal_switch_total=0
        ),
    )
    with pytest.raises(GroupedDiagnosticsError, match="differ from"):
        broken.validate(report, p, geometry)


def test_oracle_unavailable_units_are_excluded_from_status_order():
    p, report, geometry = report_and_geometry()
    records = []
    for row in report.work_evaluations:
        if row.unit.work_id == "two":
            records.append(
                dataclasses.replace(
                    row,
                    evaluation=evaluation(
                        row.unit, row.arm_id, status="ORACLE_UNAVAILABLE"
                    ),
                )
            )
        else:
            records.append(row)
    changed = GroupedComparisonReportV1.build(
        p, records, verifier_commit=git("report-verifier")
    )
    result = build_grouped_diagnostics_v1(
        changed, p, geometry, verifier_commit=git("diagnostics")
    )
    assert result.status_dominance.oracle_unavailable_count == 1
    assert result.status_dominance.oracle_available_count == 1
    assert result.status_dominance.equal_status_count == 1
