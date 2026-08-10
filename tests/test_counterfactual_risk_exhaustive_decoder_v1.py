import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import RationalMeasure
from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    GroupFamilyIdentity,
)
from audio_extract.counterfactual_risk_exhaustive_decoder_v1 import (
    ExhaustiveDecoderError,
    ExhaustiveRouteDecisionV1,
    decode_exhaustive_small_grid,
    recompute_route_objective,
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
    measures=(1, 2),
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
        prediction_manifest_sha256=sha("prediction"),
        model_sha256=sha("model"),
        model_input_sha256=sha("model-input"),
        prediction_policy_sha256=sha("prediction-policy"),
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
        measures=tuple(RationalMeasure(value) for value in measures),
        upper=risk,
        available=mask,
    )


def solve(value, routing_policy, *, maximum_cells=12):
    preflight = preflight_partitioned_upper_risks(value, routing_policy)
    return preflight, decode_exhaustive_small_grid(
        value,
        preflight,
        routing_policy,
        maximum_cells=maximum_cells,
    )


def test_complementary_unique_route_uses_physical_measure_and_switch_cost():
    routing_policy = policy()
    value = panel(routing_policy)
    preflight, result = solve(value, routing_policy)
    assert result.status == "ROUTE"
    assert result.labels is not None
    assert result.labels.tolist() == [[0, 1]]
    assert result.data_objective == pytest.approx(0.48)
    assert result.objective == pytest.approx(0.50)
    assert result.temporal_switches == 0
    assert result.frequency_switches == 1
    assert result.selection_counts == (1, 1)
    assert result.optimum_count == 1
    result.validate_against(value, preflight, routing_policy)
    assert not result.labels.flags.writeable


def test_large_switch_penalty_prefers_one_global_candidate():
    routing_policy = policy(frequency_switch_penalty=0.3)
    value = panel(routing_policy)
    _, result = solve(value, routing_policy)
    assert result.status == "ROUTE"
    assert result.labels is not None
    assert result.labels.tolist() == [[1, 1]]
    assert result.objective == pytest.approx(0.58)
    assert result.frequency_switches == 0
    assert result.selection_counts == (0, 2)


def test_nonunique_optimum_abstains_instead_of_using_enumeration_order():
    routing_policy = policy(
        secondary_weights=(),
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
    )
    upper = np.full((1, 2, 2, 3), 0.2, dtype=np.float64)
    value = panel(routing_policy, upper=upper)
    _, result = solve(value, routing_policy)
    assert result.status == "ABSTAIN_NONUNIQUE_OPTIMUM"
    assert result.labels is None
    assert result.objective is None
    assert result.optimum_count == 4
    assert result.selection_counts == (0, 0)


def test_no_confidently_feasible_cell_returns_no_route_or_fallback_labels():
    routing_policy = policy()
    value = panel(routing_policy, allowed=(False, True))
    upper = np.asarray(value.upper).copy()
    available = np.asarray(value.available).copy()
    upper[0, 0] = 0.0
    available[0, 0] = False
    blocked = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "upper": upper, "available": available}
    )
    preflight, result = solve(blocked, routing_policy)
    assert preflight.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert result.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert result.labels is None
    assert result.objective is None
    assert result.optimum_count == 0


def test_exhaustive_decoder_refuses_grid_above_frozen_limit():
    routing_policy = policy()
    value = panel(routing_policy)
    preflight = preflight_partitioned_upper_risks(value, routing_policy)
    with pytest.raises(
        ExhaustiveDecoderError,
        match="above maximum_cells",
    ):
        decode_exhaustive_small_grid(
            value,
            preflight,
            routing_policy,
            maximum_cells=1,
        )


def weighted_choice_panel(measures):
    routing_policy = policy(
        secondary_weights=(),
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=2.0,
    )
    upper = np.asarray(
        [
            [
                [[0.1, 0.1, 0.0], [0.9, 0.9, 0.0]],
                [[0.9, 0.9, 0.0], [0.1, 0.1, 0.0]],
            ]
        ],
        dtype=np.float64,
    )
    return routing_policy, panel(
        routing_policy, upper=upper, measures=measures
    )


def test_physical_measure_can_change_the_unique_whole_route_choice():
    first_policy, first_panel = weighted_choice_panel((9, 1))
    _, first = solve(first_panel, first_policy)
    assert first.status == "ROUTE"
    assert first.labels is not None
    assert first.labels.tolist() == [[0, 0]]
    assert first.data_objective == pytest.approx(0.18)

    second_policy, second_panel = weighted_choice_panel((1, 9))
    _, second = solve(second_panel, second_policy)
    assert second.status == "ROUTE"
    assert second.labels is not None
    assert second.labels.tolist() == [[1, 1]]
    assert second.data_objective == pytest.approx(0.18)


def test_equal_measure_whole_routes_abstain_when_both_are_optimal():
    routing_policy, value = weighted_choice_panel((1, 1))
    _, result = solve(value, routing_policy)
    assert result.status == "ABSTAIN_NONUNIQUE_OPTIMUM"
    assert result.optimum_count == 2


def test_independent_objective_recomputation_refuses_infeasible_or_wrong_labels():
    routing_policy = policy()
    value = panel(routing_policy)
    preflight, result = solve(value, routing_policy)
    assert result.labels is not None
    total, data, temporal, frequency = recompute_route_objective(
        result.labels, value, preflight, routing_policy
    )
    assert total == pytest.approx(result.objective)
    assert data == pytest.approx(result.data_objective)
    assert temporal == result.temporal_switches
    assert frequency == result.frequency_switches

    infeasible = np.asarray([[0, 0]], dtype=np.int32)
    with pytest.raises(
        ExhaustiveDecoderError,
        match="failed hard feasibility",
    ):
        recompute_route_objective(
            infeasible, value, preflight, routing_policy
        )


def test_decision_semantic_validation_rejects_objective_switch_and_input_substitution():
    routing_policy = policy()
    value = panel(routing_policy)
    preflight, result = solve(value, routing_policy)
    with pytest.raises(
        ExhaustiveDecoderError,
        match="objective differs",
    ):
        ExhaustiveRouteDecisionV1(
            **{**result.__dict__, "objective": result.objective + 1.0}
        ).validate_against(value, preflight, routing_policy)
    with pytest.raises(
        ExhaustiveDecoderError,
        match="switch counts differ",
    ):
        ExhaustiveRouteDecisionV1(
            **{
                **result.__dict__,
                "frequency_switches": result.frequency_switches + 1,
            }
        ).validate_against(value, preflight, routing_policy)
    with pytest.raises(
        ExhaustiveDecoderError,
        match="differs from decoder inputs",
    ):
        ExhaustiveRouteDecisionV1(
            **{**result.__dict__, "panel_sha256": sha("other-panel")}
        ).validate_against(value, preflight, routing_policy)


def test_decoder_input_identity_mismatch_is_refused_before_enumeration():
    routing_policy = policy()
    value = panel(routing_policy)
    preflight = preflight_partitioned_upper_risks(value, routing_policy)
    changed_policy = policy(frequency_switch_penalty=0.05)
    with pytest.raises(
        ExhaustiveDecoderError,
        match="inconsistent identities",
    ):
        decode_exhaustive_small_grid(
            value, preflight, changed_policy
        )


def test_decision_identity_binds_labels_objective_policy_and_inputs():
    routing_policy = policy()
    value = panel(routing_policy)
    _, first = solve(value, routing_policy)

    changed_policy = policy(frequency_switch_penalty=0.3)
    changed_panel = panel(changed_policy)
    _, second = solve(changed_panel, changed_policy)

    tie_policy = policy(
        secondary_weights=(),
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
    )
    tie_panel = panel(
        tie_policy,
        upper=np.full((1, 2, 2, 3), 0.2, dtype=np.float64),
    )
    _, third = solve(tie_panel, tie_policy)
    assert len({first.sha256, second.sha256, third.sha256}) == 3


def test_objective_tolerance_is_strictly_positive_and_identity_bearing():
    with pytest.raises(
        Exception,
        match="objective_tolerance must be strictly positive",
    ):
        policy(objective_tolerance=0.0).validate()
    first = policy(objective_tolerance=1e-10)
    second = policy(objective_tolerance=1e-8)
    assert first.sha256 != second.sha256
