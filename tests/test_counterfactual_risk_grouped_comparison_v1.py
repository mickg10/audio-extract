import dataclasses
import hashlib

import pytest

from audio_extract.counterfactual_risk_complete_route_evaluation_v1 import (
    CompleteRouteEvaluationV1,
    SelectedRiskViolationV1,
)
from audio_extract.counterfactual_risk_grouped_comparison_v1 import (
    ComparisonArmV1,
    GroupedComparisonError,
    GroupedComparisonPreregistrationV1,
    GroupedComparisonReportV1,
    HeldOutUnitV1,
    WorkArmEvaluationV1,
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
        query_quality_stratum_sha256=sha(f"query-quality-{name}"),
        exact_evidence_sha256=sha(f"exact-{name}"),
        exact_panel_sha256=sha(f"panel-exact-{name}"),
        exact_preflight_sha256=sha(f"preflight-exact-{name}"),
        exact_oracle_decision_sha256=sha(f"oracle-exact-{name}"),
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


def prereg(*names: str) -> GroupedComparisonPreregistrationV1:
    units = tuple(sorted((unit(name) for name in names), key=lambda row: row.sha256))
    return GroupedComparisonPreregistrationV1(
        experiment_id="D0_R0_HELD_OUT_V1",
        dataset_sha256=sha("dataset"),
        split_sha256=sha("split"),
        test_subset_sha256=sha("test-subset"),
        exact_evidence_manifest_sha256=sha("exact-evidence-manifest"),
        candidate_panel_sha256=sha("candidate-panel"),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        routing_policy_sha256=sha("routing-policy"),
        complete_route_evaluation_contract_sha256=sha("evaluator"),
        comparison_policy_sha256=sha("comparison-policy"),
        promotion_policy_sha256=sha("promotion-policy"),
        paired_regret_tolerance=1e-8,
        d0_arm=arm("D0"),
        r0_arm=arm("R0"),
        expected_units=units,
        source_commit=git("source"),
    )


def violation(kind: str = "critical_above_threshold"):
    if kind == "required_secondary_missing":
        return SelectedRiskViolationV1(
            0, 0, 0, "artifact", kind, None, None
        )
    return SelectedRiskViolationV1(
        0, 0, 0, "voice", kind, 1.0, 1.2
    )


def evaluation(
    arm_id: str,
    held_out: HeldOutUnitV1,
    *,
    status: str = "ROUTE_SAFE",
    regret: float = 0.0,
    temporal: int = 0,
    frequency: int = 0,
    exact_override=None,
    policy: str = "routing-policy",
):
    critical = ()
    incomplete = ()
    submitted = 1.0
    selection = regret
    if status == "ROUTE_CATASTROPHIC_FALSE_SAFE":
        critical = (violation(),)
        submitted = None
        selection = None
    elif status == "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE":
        incomplete = (violation("required_secondary_missing"),)
        submitted = None
        selection = None
    elif status in {"ABSTAIN", "ORACLE_UNAVAILABLE"}:
        submitted = None
        selection = None
        temporal = None
        frequency = None
    exact = exact_override or {
        "exact_evidence_sha256": held_out.exact_evidence_sha256,
        "exact_panel_sha256": held_out.exact_panel_sha256,
        "exact_preflight_sha256": held_out.exact_preflight_sha256,
        "exact_oracle_decision_sha256": (
            held_out.exact_oracle_decision_sha256
        ),
    }
    return CompleteRouteEvaluationV1(
        arm_id=arm_id,
        exact_evidence_sha256=exact["exact_evidence_sha256"],
        exact_panel_sha256=exact["exact_panel_sha256"],
        exact_preflight_sha256=exact["exact_preflight_sha256"],
        policy_sha256=sha(policy),
        exact_oracle_decision_sha256=(
            exact["exact_oracle_decision_sha256"]
        ),
        submission_sha256=sha(
            f"submission-{arm_id}-{held_out.work_id}-{status}"
        ),
        status=status,
        critical_false_safe_violations=critical,
        incomplete_secondary_violations=incomplete,
        exact_oracle_objective=(
            None if status == "ORACLE_UNAVAILABLE" else 1.0
        ),
        submitted_exact_objective=submitted,
        selection_regret=selection,
        temporal_switches=temporal,
        frequency_switches=frequency,
        reason=f"{arm_id} {status}",
    )


def record(preregistration, held_out, arm_id, **kwargs):
    return WorkArmEvaluationV1(
        unit=held_out,
        arm_run_sha256=(
            preregistration.d0_arm.sha256
            if arm_id == "D0"
            else preregistration.r0_arm.sha256
        ),
        dataset_sha256=preregistration.dataset_sha256,
        split_sha256=preregistration.split_sha256,
        test_subset_sha256=preregistration.test_subset_sha256,
        candidate_panel_sha256=preregistration.candidate_panel_sha256,
        feature_contract_sha256=preregistration.feature_contract_sha256,
        metric_contract_sha256=preregistration.metric_contract_sha256,
        preregistration_sha256=preregistration.sha256,
        evaluation=evaluation(arm_id, held_out, **kwargs),
    )


def matrix(preregistration, statuses=None, regrets=None, switches=None):
    statuses = statuses or {}
    regrets = regrets or {}
    switches = switches or {}
    rows = []
    for held_out in preregistration.expected_units:
        for arm_id in ("D0", "R0"):
            key = (held_out.work_id, arm_id)
            temporal, frequency = switches.get(key, (0, 0))
            rows.append(
                record(
                    preregistration,
                    held_out,
                    arm_id,
                    status=statuses.get(key, "ROUTE_SAFE"),
                    regret=regrets.get(key, 0.0),
                    temporal=temporal,
                    frequency=frequency,
                )
            )
    return rows


def test_complete_report_aggregates_safety_regret_and_switches():
    p = prereg("verdi", "puccini", "donizetti")
    statuses = {
        ("verdi", "D0"): "ROUTE_SAFE",
        ("verdi", "R0"): "ROUTE_SAFE",
        ("puccini", "D0"): "ABSTAIN",
        ("puccini", "R0"): "ROUTE_SAFE",
        ("donizetti", "D0"): "ROUTE_CATASTROPHIC_FALSE_SAFE",
        ("donizetti", "R0"): "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE",
    }
    regrets = {
        ("verdi", "D0"): 0.3,
        ("verdi", "R0"): 0.1,
        ("puccini", "R0"): 0.2,
    }
    switches = {
        ("verdi", "D0"): (4, 5),
        ("verdi", "R0"): (2, 3),
        ("puccini", "R0"): (1, 1),
    }
    report = GroupedComparisonReportV1.build(
        p, matrix(p, statuses, regrets, switches), verifier_commit=git("verify")
    )
    report.validate(p)
    assert report.status == "COMPLETE_NO_PROMOTION_DECISION"
    assert report.d0_summary.safe_route_count == 1
    assert report.d0_summary.catastrophic_false_safe_count == 1
    assert report.d0_summary.abstention_count == 1
    assert report.r0_summary.safe_route_count == 2
    assert report.r0_summary.incomplete_evidence_count == 1
    assert report.r0_summary.mean_safe_regret == pytest.approx(0.15)
    assert report.pairwise_summary.both_safe_count == 1
    assert report.pairwise_summary.r0_lower_regret_count == 1
    assert report.pairwise_summary.mean_r0_minus_d0_regret == pytest.approx(-0.2)
    assert report.identity_dict(p)["promotion_decision"] is None


def test_input_order_does_not_change_built_report():
    p = prereg("one", "two")
    values = matrix(p)
    first = GroupedComparisonReportV1.build(
        p, values, verifier_commit=git("verify")
    )
    second = GroupedComparisonReportV1.build(
        p, tuple(reversed(values)), verifier_commit=git("verify")
    )
    assert first == second
    assert first.sha256(p) == second.sha256(p)


def test_missing_or_duplicate_unit_arm_is_refused():
    p = prereg("one", "two")
    values = matrix(p)
    with pytest.raises(GroupedComparisonError, match="exactly cover"):
        GroupedComparisonReportV1.build(
            p, values[:-1], verifier_commit=git("verify")
        )
    with pytest.raises(GroupedComparisonError, match="duplicate"):
        GroupedComparisonReportV1.build(
            p, values + [values[0]], verifier_commit=git("verify")
        )


def test_cross_arm_exact_evidence_mismatch_is_refused():
    p = prereg("one", "two")
    values = matrix(p)
    target = next(
        index for index, row in enumerate(values)
        if row.unit.work_id == "one" and row.arm_id == "R0"
    )
    changed = list(values)
    changed[target] = dataclasses.replace(
        changed[target],
        evaluation=dataclasses.replace(
            changed[target].evaluation,
            exact_oracle_objective=2.0,
        ),
    )
    with pytest.raises(GroupedComparisonError, match="exact evidence differs"):
        GroupedComparisonReportV1.build(
            p, changed, verifier_commit=git("verify")
        )


def test_cross_unit_evaluation_swap_is_refused_for_both_arms():
    p = prereg("one", "two")
    values = matrix(p)
    first = next(row for row in p.expected_units if row.work_id == "one")
    second = next(row for row in p.expected_units if row.work_id == "two")
    exact = {
        "exact_evidence_sha256": second.exact_evidence_sha256,
        "exact_panel_sha256": second.exact_panel_sha256,
        "exact_preflight_sha256": second.exact_preflight_sha256,
        "exact_oracle_decision_sha256": second.exact_oracle_decision_sha256,
    }
    changed = [
        dataclasses.replace(
            row,
            evaluation=evaluation(row.arm_id, row.unit, exact_override=exact),
        )
        if row.unit == first
        else row
        for row in values
    ]
    with pytest.raises(GroupedComparisonError, match="held-out unit"):
        GroupedComparisonReportV1.build(
            p, changed, verifier_commit=git("verify")
        )


def test_catastrophic_and_incomplete_are_not_zero_regret():
    p = prereg("one", "two")
    statuses = {
        ("one", "D0"): "ROUTE_CATASTROPHIC_FALSE_SAFE",
        ("one", "R0"): "ROUTE_SAFE",
        ("two", "D0"): "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE",
        ("two", "R0"): "ABSTAIN",
    }
    report = GroupedComparisonReportV1.build(
        p,
        matrix(p, statuses, {("one", "R0"): 0.25}),
        verifier_commit=git("verify"),
    )
    assert report.d0_summary.safe_route_count == 0
    assert report.d0_summary.mean_safe_regret is None
    assert report.r0_summary.mean_safe_regret == pytest.approx(0.25)
    assert report.pairwise_summary.both_safe_count == 0
    assert report.pairwise_summary.mean_r0_minus_d0_regret is None


def test_oracle_unavailable_status_must_agree_between_arms():
    p = prereg("one", "two")
    values = matrix(
        p,
        {("one", "D0"): "ORACLE_UNAVAILABLE"},
    )
    r0_index = next(
        index for index, row in enumerate(values)
        if row.unit.work_id == "one" and row.arm_id == "R0"
    )
    values[r0_index] = dataclasses.replace(
        values[r0_index],
        evaluation=dataclasses.replace(
            values[r0_index].evaluation,
            exact_oracle_objective=None,
        ),
    )
    with pytest.raises(GroupedComparisonError, match="availability differs"):
        GroupedComparisonReportV1.build(
            p, values, verifier_commit=git("verify")
        )


def test_equal_compute_canonical_order_and_unique_families_are_binding():
    p = prereg("one", "two")
    with pytest.raises(GroupedComparisonError, match="compute budgets differ"):
        dataclasses.replace(
            p,
            r0_arm=dataclasses.replace(
                p.r0_arm, compute_budget_sha256=sha("other")
            ),
        ).validate()
    with pytest.raises(GroupedComparisonError, match="canonical SHA order"):
        dataclasses.replace(
            p, expected_units=tuple(reversed(p.expected_units))
        ).validate()
    duplicate = dataclasses.replace(
        p.expected_units[1],
        source_family_sha256=p.expected_units[0].source_family_sha256,
    )
    duplicate_units = tuple(
        sorted((p.expected_units[0], duplicate), key=lambda row: row.sha256)
    )
    with pytest.raises(GroupedComparisonError, match="only once"):
        dataclasses.replace(p, expected_units=duplicate_units).validate()


def test_unit_identity_binds_query_quality_and_exact_evidence():
    original = unit("one")
    assert original.sha256 != dataclasses.replace(
        original, query_quality_stratum_sha256=sha("other-quality")
    ).sha256
    assert original.sha256 != dataclasses.replace(
        original, exact_panel_sha256=sha("other-panel")
    ).sha256
    assert original.group_family_sha256 == dataclasses.replace(
        original, exact_panel_sha256=sha("other-panel")
    ).group_family_sha256


def test_summary_tampering_is_detected():
    p = prereg("one", "two")
    report = GroupedComparisonReportV1.build(
        p, matrix(p), verifier_commit=git("verify")
    )
    broken = dataclasses.replace(
        report,
        d0_summary=dataclasses.replace(
            report.d0_summary, critical_violation_count=1
        ),
    )
    with pytest.raises(GroupedComparisonError, match="D0 summary differs"):
        broken.validate(p)


def test_paired_regret_tolerance_changes_only_tie_classification():
    p = dataclasses.replace(
        prereg("one", "two"), paired_regret_tolerance=0.05
    )
    regrets = {
        ("one", "D0"): 0.10,
        ("one", "R0"): 0.14,
        ("two", "D0"): 0.20,
        ("two", "R0"): 0.10,
    }
    report = GroupedComparisonReportV1.build(
        p, matrix(p, regrets=regrets), verifier_commit=git("verify")
    )
    assert report.pairwise_summary.regret_tie_count == 1
    assert report.pairwise_summary.r0_lower_regret_count == 1
    assert report.pairwise_summary.mean_r0_minus_d0_regret == pytest.approx(-0.03)


def test_preregistration_binds_evaluator_evidence_and_promotion_policies():
    p = prereg("one", "two")
    identities = {
        p.sha256,
        dataclasses.replace(
            p, exact_evidence_manifest_sha256=sha("other-evidence")
        ).sha256,
        dataclasses.replace(
            p, complete_route_evaluation_contract_sha256=sha("other-evaluator")
        ).sha256,
        dataclasses.replace(
            p, promotion_policy_sha256=sha("other-promotion")
        ).sha256,
    }
    assert len(identities) == 4
