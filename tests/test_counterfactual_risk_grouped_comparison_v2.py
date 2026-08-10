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
from audio_extract.counterfactual_risk_grouped_comparison_v2 import (
    ArmRouteProvenanceV2,
    GroupedComparisonReportV2,
    GroupedComparisonV2Error,
    WorkArmEvaluationV2,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def unit(
    name: str,
    *,
    shared_exact: str | None = None,
) -> HeldOutUnitV1:
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


def prereg(
    *units: HeldOutUnitV1,
) -> GroupedComparisonPreregistrationV1:
    ordered = tuple(sorted(units, key=lambda row: row.sha256))
    return GroupedComparisonPreregistrationV1(
        experiment_id="D0_R0_HELD_OUT_V2",
        dataset_sha256=sha("dataset"),
        split_sha256=sha("split"),
        test_subset_sha256=sha("test-subset"),
        exact_evidence_manifest_sha256=sha("exact-manifest"),
        candidate_panel_sha256=sha("candidate-panel"),
        feature_contract_sha256=sha("feature-contract"),
        metric_contract_sha256=sha("metric-contract"),
        routing_policy_sha256=sha("routing-policy"),
        complete_route_evaluation_contract_sha256=sha("evaluator-v1"),
        comparison_policy_sha256=sha("comparison-v2"),
        promotion_policy_sha256=sha("promotion-separate"),
        paired_regret_tolerance=1e-8,
        d0_arm=arm("D0"),
        r0_arm=arm("R0"),
        expected_units=ordered,
        source_commit=git("source"),
    )


def submission(
    arm_id: str,
    held_out: HeldOutUnitV1,
    *,
    status: str = "ROUTE",
) -> ArmRouteSubmissionV1:
    return ArmRouteSubmissionV1(
        arm_id=arm_id,
        route_evidence_sha256=sha(
            f"route-artifact-{held_out.work_id}-{arm_id}"
        ),
        source_status=(
            "ROUTE" if status == "ROUTE" else "ABSTAIN_UNCERTAIN"
        ),
        status=status,
        labels=(
            np.asarray([[0]], dtype=np.int32)
            if status == "ROUTE"
            else None
        ),
        reason=f"{arm_id} {status}",
    )


def provenance(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    *,
    submitted: ArmRouteSubmissionV1 | None = None,
) -> ArmRouteProvenanceV2:
    frozen = p.d0_arm if arm_id == "D0" else p.r0_arm
    submitted = submitted or submission(arm_id, held_out)
    return ArmRouteProvenanceV2(
        unit_sha256=held_out.sha256,
        group_family_sha256=held_out.group_family_sha256,
        query_condition_sha256=held_out.query_condition_sha256,
        query_quality_stratum_sha256=(
            held_out.query_quality_stratum_sha256
        ),
        arm_id=arm_id,
        arm_run_sha256=frozen.sha256,
        artifact_manifest_sha256=frozen.artifact_manifest_sha256,
        compute_budget_sha256=frozen.compute_budget_sha256,
        checkpoint_rule_sha256=frozen.checkpoint_rule_sha256,
        inference_policy_sha256=frozen.inference_policy_sha256,
        source_commit=frozen.source_commit,
        route_artifact_sha256=submitted.route_evidence_sha256,
        verifier_commit=git(f"verify-{arm_id}"),
        submission=submitted,
    )


def critical_violation() -> SelectedRiskViolationV1:
    return SelectedRiskViolationV1(
        0,
        0,
        0,
        "voice",
        "critical_above_threshold",
        1.0,
        1.2,
    )


def incomplete_violation() -> SelectedRiskViolationV1:
    return SelectedRiskViolationV1(
        0,
        0,
        0,
        "artifact",
        "required_secondary_missing",
        None,
        None,
    )


def evaluation(
    held_out: HeldOutUnitV1,
    route: ArmRouteSubmissionV1,
    *,
    status: str = "ROUTE_SAFE",
    regret: float = 0.0,
    temporal: int = 0,
    frequency: int = 0,
    oracle_objective: float | None | object = ...,
) -> CompleteRouteEvaluationV1:
    if oracle_objective is ...:
        oracle_objective = (
            None if status == "ORACLE_UNAVAILABLE" else 1.0
        )
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
        policy_sha256=sha("routing-policy"),
        exact_oracle_decision_sha256=(
            held_out.exact_oracle_decision_sha256
        ),
        submission_sha256=route.sha256,
        status=status,
        critical_false_safe_violations=critical,
        incomplete_secondary_violations=incomplete,
        exact_oracle_objective=oracle_objective,
        submitted_exact_objective=(1.0 + regret if safe else None),
        selection_regret=(regret if safe else None),
        temporal_switches=(temporal if route_status else None),
        frequency_switches=(frequency if route_status else None),
        reason=f"{route.arm_id} {status}",
    )


def record(
    p: GroupedComparisonPreregistrationV1,
    held_out: HeldOutUnitV1,
    arm_id: str,
    *,
    status: str = "ROUTE_SAFE",
    regret: float = 0.0,
    temporal: int = 0,
    frequency: int = 0,
    oracle_objective: float | None | object = ...,
) -> WorkArmEvaluationV2:
    route_status = "ABSTAIN" if status == "ABSTAIN" else "ROUTE"
    route = submission(arm_id, held_out, status=route_status)
    return WorkArmEvaluationV2(
        unit=held_out,
        route_provenance=provenance(
            p,
            held_out,
            arm_id,
            submitted=route,
        ),
        dataset_sha256=p.dataset_sha256,
        split_sha256=p.split_sha256,
        test_subset_sha256=p.test_subset_sha256,
        candidate_panel_sha256=p.candidate_panel_sha256,
        feature_contract_sha256=p.feature_contract_sha256,
        metric_contract_sha256=p.metric_contract_sha256,
        preregistration_sha256=p.sha256,
        evaluation=evaluation(
            held_out,
            route,
            status=status,
            regret=regret,
            temporal=temporal,
            frequency=frequency,
            oracle_objective=oracle_objective,
        ),
    )


def matrix(
    p: GroupedComparisonPreregistrationV1,
    *,
    statuses=None,
    regrets=None,
) -> list[WorkArmEvaluationV2]:
    statuses = statuses or {}
    regrets = regrets or {}
    rows = []
    for held_out in p.expected_units:
        for arm_id in ("D0", "R0"):
            key = (held_out.work_id, arm_id)
            rows.append(
                record(
                    p,
                    held_out,
                    arm_id,
                    status=statuses.get(key, "ROUTE_SAFE"),
                    regret=regrets.get(key, 0.0),
                )
            )
    return rows


def test_v2_builds_complete_nonpromoting_report():
    p = prereg(unit("one"), unit("two"), unit("three"))
    statuses = {
        ("one", "D0"): "ROUTE_SAFE",
        ("one", "R0"): "ROUTE_SAFE",
        ("two", "D0"): "ABSTAIN",
        ("two", "R0"): "ROUTE_SAFE",
        ("three", "D0"): "ROUTE_CATASTROPHIC_FALSE_SAFE",
        ("three", "R0"): "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE",
    }
    regrets = {
        ("one", "D0"): 0.3,
        ("one", "R0"): 0.1,
        ("two", "R0"): 0.2,
    }
    report = GroupedComparisonReportV2.build(
        p,
        matrix(p, statuses=statuses, regrets=regrets),
        verifier_commit=git("report-verifier"),
    )
    report.validate(p)
    payload = report.identity_dict(p)
    assert payload["promotion_decision"] is None
    assert report.d0_summary.safe_route_count == 1
    assert report.d0_summary.catastrophic_false_safe_count == 1
    assert report.r0_summary.safe_route_count == 2
    assert report.r0_summary.incomplete_evidence_count == 1
    assert report.pairwise_summary.both_safe_count == 1
    assert report.pairwise_summary.mean_r0_minus_d0_regret == pytest.approx(
        -0.2
    )


def test_unit_swap_fails_even_when_exact_evidence_is_shared():
    first = unit("first", shared_exact="shared")
    second = unit("second", shared_exact="shared")
    p = prereg(first, second)
    value = record(p, first, "D0")
    moved = dataclasses.replace(value, unit=second)
    with pytest.raises(
        GroupedComparisonV2Error,
        match="another held-out unit",
    ):
        moved.validate(p)


def test_forged_arm_run_and_route_artifact_are_refused():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(p, held_out, "D0")
    forged = dataclasses.replace(
        value.route_provenance,
        arm_run_sha256=p.r0_arm.sha256,
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="another frozen arm run",
    ):
        dataclasses.replace(
            value,
            route_provenance=forged,
        ).validate(p)
    wrong_route = dataclasses.replace(
        value.route_provenance,
        route_artifact_sha256=sha("other-route"),
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="another route artifact",
    ):
        dataclasses.replace(
            value,
            route_provenance=wrong_route,
        ).validate(p)


def test_complete_evaluation_must_name_the_bound_submission():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(p, held_out, "R0")
    broken = dataclasses.replace(
        value.evaluation,
        submission_sha256=sha("unregistered-submission"),
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="another route submission",
    ):
        dataclasses.replace(value, evaluation=broken).validate(p)


@pytest.mark.parametrize(
    "status",
    [
        "ABSTAIN",
        "ROUTE_CATASTROPHIC_FALSE_SAFE",
        "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE",
    ],
)
def test_oracle_available_status_requires_finite_objective(status):
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(
        p,
        held_out,
        "D0",
        status=status,
        oracle_objective=None,
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="lacks a finite objective",
    ):
        value.validate(p)


def test_oracle_unavailable_must_not_carry_objective():
    held_out = unit("one")
    p = prereg(held_out, unit("two"))
    value = record(
        p,
        held_out,
        "D0",
        status="ORACLE_UNAVAILABLE",
        oracle_objective=1.0,
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="carries an oracle objective",
    ):
        value.validate(p)


def test_summary_type_tampering_is_rejected():
    p = prereg(unit("one"), unit("two"))
    report = GroupedComparisonReportV2.build(
        p,
        matrix(p),
        verifier_commit=git("verify"),
    )
    broken_count = dataclasses.replace(
        report.d0_summary,
        unit_count=2.0,
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="must be a non-negative integer",
    ):
        broken_count.validate()
    broken_bool = dataclasses.replace(
        report.d0_summary,
        safe_route_count=True,
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="must be a non-negative integer",
    ):
        broken_bool.validate()
    cross = list(report.pairwise_summary.status_cross_tab)
    left, right, _ = cross[0]
    cross[0] = (left, right, True)
    broken_cross = dataclasses.replace(
        report.pairwise_summary,
        status_cross_tab=tuple(cross),
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="positive integer",
    ):
        broken_cross.validate()


def test_cross_arm_oracle_objective_mismatch_is_refused():
    p = prereg(unit("one"), unit("two"))
    values = matrix(p)
    index = next(
        i
        for i, row in enumerate(values)
        if row.unit.work_id == "one" and row.arm_id == "R0"
    )
    values[index] = dataclasses.replace(
        values[index],
        evaluation=dataclasses.replace(
            values[index].evaluation,
            exact_oracle_objective=2.0,
        ),
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="exact evidence differs",
    ):
        GroupedComparisonReportV2.build(
            p,
            values,
            verifier_commit=git("verify"),
        )


def test_missing_duplicate_and_extra_matrix_cells_are_refused():
    p = prereg(unit("one"), unit("two"))
    values = matrix(p)
    with pytest.raises(
        GroupedComparisonV2Error,
        match="exactly cover",
    ):
        GroupedComparisonReportV2.build(
            p,
            values[:-1],
            verifier_commit=git("verify"),
        )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="duplicate",
    ):
        GroupedComparisonReportV2.build(
            p,
            values + [values[0]],
            verifier_commit=git("verify"),
        )


def test_all_oracle_unavailable_has_none_coverages_not_zero():
    p = prereg(unit("one"), unit("two"))
    statuses = {
        (held_out.work_id, arm_id): "ORACLE_UNAVAILABLE"
        for held_out in p.expected_units
        for arm_id in ("D0", "R0")
    }
    report = GroupedComparisonReportV2.build(
        p,
        matrix(p, statuses=statuses),
        verifier_commit=git("verify"),
    )
    assert report.d0_summary.oracle_available_count == 0
    assert report.d0_summary.route_submission_coverage is None
    assert report.d0_summary.safe_route_coverage is None
    assert report.d0_summary.abstention_rate is None
    assert report.pairwise_summary.oracle_available_count == 0
    assert report.pairwise_summary.both_safe_coverage is None


def test_summary_tampering_is_recomputed_and_rejected():
    p = prereg(unit("one"), unit("two"))
    report = GroupedComparisonReportV2.build(
        p,
        matrix(p),
        verifier_commit=git("verify"),
    )
    broken = dataclasses.replace(
        report,
        pairwise_summary=dataclasses.replace(
            report.pairwise_summary,
            r0_lower_regret_count=1,
            regret_tie_count=1,
        ),
    )
    with pytest.raises(
        GroupedComparisonV2Error,
        match="pairwise summary differs",
    ):
        broken.validate(p)
