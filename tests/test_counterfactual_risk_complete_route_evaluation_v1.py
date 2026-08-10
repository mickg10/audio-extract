import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import RationalMeasure
from audio_extract.counterfactual_risk_complete_route_evaluation_v1 import (
    ArmRouteSubmissionV1,
    CompleteRouteEvaluationError,
    CompleteRouteEvaluationV1,
    SelectedRiskViolationV1,
    evaluate_complete_route_submission,
)
from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    GroupFamilyIdentity,
)
from audio_extract.counterfactual_risk_exhaustive_decoder_v1 import (
    decode_exhaustive_small_grid,
)
from audio_extract.counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedUpperRiskPanelV1,
)
from audio_extract.counterfactual_risk_routing_preflight_v1 import (
    FrozenRoutingPolicyV1,
    preflight_partitioned_upper_risks,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def policy(**kwargs) -> FrozenRoutingPolicyV1:
    values = dict(
        critical_thresholds=(("voice", 1.0), ("hole", 1.0)),
        secondary_weights=(("artifact", 0.1),),
        critical_slack_weight=1.0,
        temporal_switch_penalty=0.05,
        frequency_switch_penalty=0.04,
        feasibility_tolerance=0.0,
        objective_tolerance=1e-10,
        whole_track_abstention=True,
        require_secondary_evidence=True,
    )
    values.update(kwargs)
    return FrozenRoutingPolicyV1(**values)


def panel(
    routing_policy=None,
    *,
    upper=None,
    available=None,
    allowed=(True, True),
) -> PartitionedUpperRiskPanelV1:
    routing_policy = routing_policy or policy()
    group = GroupFamilyIdentity(
        work_id="work",
        recording_session_id="session",
        target_singer_id="singer",
        source_family_sha256=sha("source-family"),
        query_condition_sha256=sha("query"),
    )
    risk = np.asarray(
        upper
        if upper is not None
        else [
            [
                [[0.5, 0.2, 3.0], [0.2, 1.0, 1.0]],
                [[1.1, 0.2, 0.5], [0.2, 0.3, 0.2]],
            ]
        ],
        dtype=np.float64,
    )
    mask = np.asarray(
        available
        if available is not None
        else np.ones_like(risk, dtype=bool),
        dtype=bool,
    )
    return PartitionedUpperRiskPanelV1(
        group_family_sha256=group.sha256,
        work_id=group.work_id,
        recording_session_id=group.recording_session_id,
        target_singer_id=group.target_singer_id,
        source_family_sha256=group.source_family_sha256,
        query_condition_sha256=group.query_condition_sha256,
        mixture_pcm_sha256=sha("mixture"),
        spectral_grid_sha256=sha("grid"),
        resolution_ms=500,
        time_cell_count=1,
        band_count=2,
        partition_certificate_sha256=sha("partition"),
        prediction_manifest_sha256=sha("exact-panel"),
        model_sha256=sha("exact-oracle"),
        model_input_sha256=sha("exact-input"),
        prediction_policy_sha256=sha("exact-policy"),
        source_commit=git("source"),
        candidate_panel_sha256=sha("candidate-panel"),
        candidate_slot_sha256s=(sha("slot-a"), sha("slot-b")),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=routing_policy.sha256,
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        inference_row_sha256s=(sha("row-0"), sha("row-1")),
        cell_sha256s=(sha("cell-0"), sha("cell-1")),
        row_prediction_allowed=tuple(allowed),
        measures=(RationalMeasure(1), RationalMeasure(2)),
        upper=risk,
        available=mask,
    )


def submission(labels, *, arm="R0", evidence="route-evidence"):
    return ArmRouteSubmissionV1(
        arm_id=arm,
        route_evidence_sha256=sha(evidence),
        source_status="ROUTE",
        status="ROUTE",
        labels=np.asarray(labels, dtype=np.int32),
        reason="synthetic submitted route",
    )


def evaluate(value, routing_policy, submitted, *, maximum_cells=12):
    exact_preflight = preflight_partitioned_upper_risks(
        value, routing_policy
    )
    return evaluate_complete_route_submission(
        submitted,
        value,
        exact_preflight,
        routing_policy,
        exact_evidence_sha256=sha("exact-evidence"),
        maximum_cells=maximum_cells,
    )


def test_exact_optimum_submission_has_zero_complete_route_regret():
    routing_policy = policy()
    value = panel(routing_policy)
    preflight = preflight_partitioned_upper_risks(value, routing_policy)
    oracle = decode_exhaustive_small_grid(
        value, preflight, routing_policy
    )
    submitted = ArmRouteSubmissionV1.from_exhaustive_decision(
        "R0", oracle
    )
    result = evaluate(value, routing_policy, submitted)
    assert result.status == "ROUTE_SAFE"
    assert result.exact_oracle_objective == pytest.approx(0.50)
    assert result.submitted_exact_objective == pytest.approx(0.50)
    assert result.selection_regret == pytest.approx(0.0)
    assert result.critical_false_safe_violations == ()
    assert result.incomplete_secondary_violations == ()
    assert result.temporal_switches == 0
    assert result.frequency_switches == 1


def test_feasible_suboptimal_complete_route_has_positive_regret():
    routing_policy = policy()
    value = panel(routing_policy)
    result = evaluate(
        value,
        routing_policy,
        submission([[1, 1]], arm="D0"),
    )
    assert result.status == "ROUTE_SAFE"
    assert result.submitted_exact_objective == pytest.approx(0.58)
    assert result.selection_regret == pytest.approx(0.08)
    assert result.frequency_switches == 0


def test_exact_critical_violation_is_a_catastrophic_false_safe():
    routing_policy = policy()
    value = panel(routing_policy)
    result = evaluate(
        value,
        routing_policy,
        submission([[0, 0]], arm="R0"),
    )
    assert result.status == "ROUTE_CATASTROPHIC_FALSE_SAFE"
    assert result.submitted_exact_objective is None
    assert result.selection_regret is None
    assert len(result.critical_false_safe_violations) == 1
    violation = result.critical_false_safe_violations[0]
    assert violation.time_index == 0
    assert violation.band_index == 1
    assert violation.candidate_index == 0
    assert violation.metric_name == "voice"
    assert violation.kind == "critical_above_threshold"
    assert violation.threshold == pytest.approx(1.0)
    assert violation.value == pytest.approx(1.1)


def test_missing_exact_critical_evidence_is_catastrophic_false_safe():
    routing_policy = policy()
    value = panel(routing_policy)
    upper = np.asarray(value.upper).copy()
    available = np.asarray(value.available).copy()
    upper[0, 0, 1, 1] = 0.0
    available[0, 0, 1, 1] = False
    changed = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "upper": upper, "available": available}
    )
    result = evaluate(
        changed,
        routing_policy,
        submission([[1, 1]]),
    )
    assert result.status == "ROUTE_CATASTROPHIC_FALSE_SAFE"
    assert any(
        violation.kind == "critical_missing"
        and violation.metric_name == "hole"
        for violation in result.critical_false_safe_violations
    )


def test_missing_required_secondary_evidence_is_incomplete_not_safe():
    routing_policy = policy()
    value = panel(routing_policy)
    upper = np.asarray(value.upper).copy()
    available = np.asarray(value.available).copy()
    upper[0, 0, 1, 2] = 0.0
    available[0, 0, 1, 2] = False
    changed = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "upper": upper, "available": available}
    )
    result = evaluate(
        changed,
        routing_policy,
        submission([[1, 1]]),
    )
    assert result.status == "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE"
    assert result.critical_false_safe_violations == ()
    assert len(result.incomplete_secondary_violations) == 1
    violation = result.incomplete_secondary_violations[0]
    assert violation.kind == "required_secondary_missing"
    assert violation.metric_name == "artifact"
    assert result.selection_regret is None


def test_arm_abstention_is_preserved_and_not_scored_as_zero_regret():
    routing_policy = policy()
    value = panel(routing_policy)
    submitted = ArmRouteSubmissionV1(
        arm_id="R0",
        route_evidence_sha256=sha("abstention"),
        source_status="ABSTAIN_NONUNIQUE_OPTIMUM",
        status="ABSTAIN",
        labels=None,
        reason="student abstained",
    )
    result = evaluate(value, routing_policy, submitted)
    assert result.status == "ABSTAIN"
    assert result.exact_oracle_objective == pytest.approx(0.50)
    assert result.submitted_exact_objective is None
    assert result.selection_regret is None
    assert result.temporal_switches is None
    assert result.frequency_switches is None


def test_nonunique_exact_oracle_blocks_evaluation_of_every_arm():
    routing_policy = policy(
        secondary_weights=(),
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
    )
    value = panel(
        routing_policy,
        upper=np.full((1, 2, 2, 3), 0.2, dtype=np.float64),
    )
    result = evaluate(
        value,
        routing_policy,
        submission([[0, 0]]),
    )
    assert result.status == "ORACLE_UNAVAILABLE"
    assert result.exact_oracle_objective is None
    assert result.selection_regret is None
    assert result.critical_false_safe_violations == ()


def test_threshold_equality_is_exact_safe_evidence():
    routing_policy = policy()
    value = panel(routing_policy)
    result = evaluate(
        value,
        routing_policy,
        submission([[1, 1]]),
    )
    assert result.status == "ROUTE_SAFE"
    # Candidate 1's first-cell hole value equals the 1.0 threshold.
    assert value.upper[0, 0, 1, 1] == 1.0
    assert not result.critical_false_safe_violations


def test_submitted_shape_and_candidate_range_are_refused():
    routing_policy = policy()
    value = panel(routing_policy)
    with pytest.raises(
        CompleteRouteEvaluationError,
        match="label grid differs",
    ):
        evaluate(value, routing_policy, submission([[0]]))
    with pytest.raises(
        CompleteRouteEvaluationError,
        match="outside the exact panel",
    ):
        evaluate(value, routing_policy, submission([[0, 2]]))


def test_submission_and_evaluation_identity_bind_arm_route_evidence_and_exact_facts():
    routing_policy = policy()
    value = panel(routing_policy)
    first = evaluate(
        value,
        routing_policy,
        submission([[1, 1]], arm="D0", evidence="first-route"),
    )
    second = evaluate(
        value,
        routing_policy,
        submission([[1, 1]], arm="R0", evidence="second-route"),
    )
    changed_upper = np.asarray(value.upper).copy()
    changed_upper[0, 0, 0, 2] += 0.5
    changed_panel = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "upper": changed_upper}
    )
    third = evaluate(
        changed_panel,
        routing_policy,
        submission([[1, 1]], arm="D0", evidence="first-route"),
    )
    assert len({first.sha256, second.sha256, third.sha256}) == 3


def test_violation_and_status_semantics_are_fail_closed():
    critical = SelectedRiskViolationV1(
        0,
        1,
        0,
        "voice",
        "critical_above_threshold",
        1.0,
        1.1,
    )
    critical.validate()
    with pytest.raises(
        CompleteRouteEvaluationError,
        match="above-threshold violation",
    ):
        SelectedRiskViolationV1(
            0,
            1,
            0,
            "voice",
            "critical_above_threshold",
            1.0,
            1.0,
        ).validate()

    routing_policy = policy()
    value = panel(routing_policy)
    safe = evaluate(
        value,
        routing_policy,
        submission([[1, 1]]),
    )
    with pytest.raises(
        CompleteRouteEvaluationError,
        match="safe route contains exact violations",
    ):
        CompleteRouteEvaluationV1(
            **{
                **safe.__dict__,
                "critical_false_safe_violations": (critical,),
            }
        ).validate()


def test_maximum_cell_limit_is_inherited_from_exact_oracle():
    routing_policy = policy()
    value = panel(routing_policy)
    exact_preflight = preflight_partitioned_upper_risks(
        value, routing_policy
    )
    with pytest.raises(Exception, match="above maximum_cells"):
        evaluate_complete_route_submission(
            submission([[0, 1]]),
            value,
            exact_preflight,
            routing_policy,
            exact_evidence_sha256=sha("exact-evidence"),
            maximum_cells=1,
        )
