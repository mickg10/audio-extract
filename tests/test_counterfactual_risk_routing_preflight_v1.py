import hashlib
import itertools

import numpy as np
import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import RationalMeasure
from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    GroupFamilyIdentity,
)
from audio_extract.counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedUpperRiskPanelV1,
)
from audio_extract.counterfactual_risk_routing_preflight_v1 import (
    FrozenRoutingPolicyV1,
    RoutingPreflightError,
    RoutingPreflightV1,
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
        whole_track_abstention=True,
        require_secondary_evidence=True,
    )
    values.update(kwargs)
    return FrozenRoutingPolicyV1(**values)


def panel(
    *,
    upper=None,
    available=None,
    route_policy_sha256=None,
    allowed=(True, True),
) -> PartitionedUpperRiskPanelV1:
    routing_policy = policy()
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
        route_policy_sha256=(
            route_policy_sha256 or routing_policy.sha256
        ),
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


def test_ready_preflight_computes_expected_cost_and_hard_feasibility():
    value = panel()
    result = preflight_partitioned_upper_risks(value, policy())
    assert result.status == "READY_FOR_CERTIFIED_DECODER"
    assert result.no_feasible_candidate_cells == ()
    assert result.cost.shape == (1, 2, 2)
    assert result.feasible.shape == result.cost.shape
    assert result.cost[0, 0, 0] == pytest.approx(0.8)
    assert result.cost[0, 0, 1] == pytest.approx(1.1)
    assert result.cost[0, 1, 0] == pytest.approx(1.15)
    assert result.cost[0, 1, 1] == pytest.approx(0.32)
    assert result.feasible.tolist() == [[[True, True], [False, True]]]
    assert not result.cost.flags.writeable
    assert not result.feasible.flags.writeable


def test_threshold_equality_is_feasible_and_above_threshold_is_not():
    value = panel()
    result = preflight_partitioned_upper_risks(value, policy())
    assert result.feasible[0, 0, 1]
    changed_upper = np.asarray(value.upper).copy()
    changed_upper[0, 0, 1, 1] = np.nextafter(1.0, np.inf)
    changed = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "upper": changed_upper}
    )
    result = preflight_partitioned_upper_risks(changed, policy())
    assert not result.feasible[0, 0, 1]


def test_missing_secondary_evidence_is_fail_closed_by_default():
    value = panel()
    available = np.asarray(value.available).copy()
    available[0, 1, 1, 2] = False
    upper = np.asarray(value.upper).copy()
    upper[0, 1, 1, 2] = 0.0
    changed = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "available": available, "upper": upper}
    )
    result = preflight_partitioned_upper_risks(changed, policy())
    assert result.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert result.no_feasible_candidate_cells == ((0, 1),)
    assert not result.feasible[0, 1].any()


def test_optional_secondary_evidence_can_be_ignored_only_by_bound_policy():
    optional_policy = policy(require_secondary_evidence=False)
    value = panel(route_policy_sha256=optional_policy.sha256)
    available = np.asarray(value.available).copy()
    available[0, 1, 1, 2] = False
    upper = np.asarray(value.upper).copy()
    upper[0, 1, 1, 2] = 0.0
    changed = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "available": available, "upper": upper}
    )
    result = preflight_partitioned_upper_risks(
        changed, optional_policy
    )
    assert result.status == "READY_FOR_CERTIFIED_DECODER"
    assert result.feasible[0, 1, 1]
    assert result.cost[0, 1, 1] == pytest.approx(0.3)


def test_feature_blocked_cell_forces_whole_track_abstention():
    value = panel(allowed=(False, True))
    available = np.asarray(value.available).copy()
    available[0, 0] = False
    upper = np.asarray(value.upper).copy()
    upper[0, 0] = 0.0
    changed = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "available": available, "upper": upper}
    )
    result = preflight_partitioned_upper_risks(changed, policy())
    assert result.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert result.no_feasible_candidate_cells == ((0, 0),)
    assert not result.feasible[0, 0].any()


def test_policy_identity_must_equal_partitioned_panel_policy():
    value = panel()
    changed_policy = policy(temporal_switch_penalty=0.06)
    with pytest.raises(
        RoutingPreflightError,
        match="policy identity differs",
    ):
        preflight_partitioned_upper_risks(value, changed_policy)


def test_policy_rejects_unknown_duplicate_overlap_zero_and_relaxed_abstention():
    metric_names = ("voice", "hole", "artifact")
    cases = (
        (
            policy(critical_thresholds=(("unknown", 1.0),)),
            "unknown metrics",
        ),
        (
            policy(
                critical_thresholds=(("voice", 1.0), ("voice", 2.0))
            ),
            "unique",
        ),
        (
            policy(secondary_weights=(("voice", 0.1),)),
            "both critical and secondary",
        ),
        (
            policy(critical_thresholds=(("voice", 0.0),)),
            "positive",
        ),
        (
            policy(secondary_weights=(("artifact", 0.0),)),
            "strictly positive",
        ),
        (
            policy(whole_track_abstention=False),
            "whole-track abstention",
        ),
    )
    for value, message in cases:
        with pytest.raises(RoutingPreflightError, match=message):
            value.validate(metric_names)


def test_unavailable_risks_never_enter_cost_arithmetic():
    value = panel()
    available = np.asarray(value.available).copy()
    available[0, 0, 0, 0] = False
    upper = np.asarray(value.upper).copy()
    upper[0, 0, 0, 0] = 0.0
    changed = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "available": available, "upper": upper}
    )
    result = preflight_partitioned_upper_risks(changed, policy())
    assert np.isfinite(result.cost).all()
    assert not result.feasible[0, 0, 0]


def test_preflight_manifest_rejects_status_cell_list_and_mask_contradictions():
    result = preflight_partitioned_upper_risks(panel(), policy())
    with pytest.raises(RoutingPreflightError, match="status differs"):
        RoutingPreflightV1(
            **{**result.__dict__, "status": "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"}
        ).validate()
    with pytest.raises(RoutingPreflightError, match="cell list differs"):
        RoutingPreflightV1(
            **{
                **result.__dict__,
                "no_feasible_candidate_cells": ((0, 0),),
            }
        ).validate()

    feasible = np.asarray(result.feasible).copy()
    feasible[0, 0] = False
    with pytest.raises(RoutingPreflightError, match="cell list differs"):
        RoutingPreflightV1(
            **{**result.__dict__, "feasible": feasible}
        ).validate()


def test_preflight_identity_binds_policy_cost_feasibility_and_panel():
    first = preflight_partitioned_upper_risks(panel(), policy())
    changed_policy = policy(frequency_switch_penalty=0.06)
    changed_panel = panel(route_policy_sha256=changed_policy.sha256)
    second = preflight_partitioned_upper_risks(
        changed_panel, changed_policy
    )
    changed_upper = np.asarray(changed_panel.upper).copy()
    changed_upper[0, 0, 0, 2] += 1.0
    third = preflight_partitioned_upper_risks(
        PartitionedUpperRiskPanelV1(
            **{**changed_panel.__dict__, "upper": changed_upper}
        ),
        changed_policy,
    )
    assert len({first.sha256, second.sha256, third.sha256}) == 3


def test_exhaustive_one_cell_critical_states_match_fail_closed_rule():
    routing_policy = policy(secondary_weights=())
    checked = 0
    for risk_bits in itertools.product((0.5, 1.5), repeat=4):
        risks = np.asarray(risk_bits, dtype=np.float64).reshape(1, 1, 2, 2)
        upper = np.concatenate(
            (risks, np.zeros((1, 1, 2, 1), dtype=np.float64)),
            axis=-1,
        )
        for mask_bits in itertools.product((False, True), repeat=4):
            critical_mask = np.asarray(mask_bits, dtype=bool).reshape(
                1, 1, 2, 2
            )
            available = np.concatenate(
                (
                    critical_mask,
                    np.ones((1, 1, 2, 1), dtype=bool),
                ),
                axis=-1,
            )
            stored = np.where(available, upper, 0.0)
            value = panel(
                upper=np.repeat(stored, 2, axis=1),
                available=np.repeat(available, 2, axis=1),
                route_policy_sha256=routing_policy.sha256,
                allowed=(True, True),
            )
            # The helper panel has two bands; use the same exhaustive state in
            # both so the expected feasibility is identical for both cells.
            result = preflight_partitioned_upper_risks(
                value, routing_policy
            )
            expected = np.all(
                critical_mask & (risks <= 1.0), axis=-1
            )[0, 0]
            assert np.array_equal(result.feasible[0, 0], expected)
            assert np.array_equal(result.feasible[0, 1], expected)
            checked += 1
    assert checked == 2 ** 8
