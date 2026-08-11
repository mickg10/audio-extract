import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import RationalMeasure
from audio_extract.counterfactual_risk_d0_structured_teacher_v1 import (
    D0StructuredTeacherError,
    D0StructuredTeacherV1,
    build_d0_structured_teacher,
)
from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    GroupFamilyIdentity,
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


_DEFAULT_EVIDENCE = "sha256:" + hashlib.sha256(b"exact-evidence").hexdigest()


def policy(**kwargs) -> FrozenRoutingPolicyV1:
    values = {
        "critical_thresholds": (("voice", 1.0), ("hole", 1.0)),
        "secondary_weights": (("artifact", 0.1),),
        "critical_slack_weight": 1.0,
        "temporal_switch_penalty": 0.05,
        "frequency_switch_penalty": 0.04,
        "feasibility_tolerance": 0.0,
        "objective_tolerance": 1e-10,
        "whole_track_abstention": True,
        "require_secondary_evidence": True,
    }
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
        prediction_manifest_sha256=sha("exact-risk-panel"),
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
        measures=tuple(RationalMeasure(value) for value in measures),
        upper=risk,
        available=mask,
    )


def teacher(
    value,
    routing_policy,
    *,
    minimum_margin=0.1,
    maximum_cells=12,
    exact_evidence=_DEFAULT_EVIDENCE,
):
    preflight = preflight_partitioned_upper_risks(value, routing_policy)
    result = build_d0_structured_teacher(
        value,
        preflight,
        routing_policy,
        exact_evidence_sha256=exact_evidence,
        minimum_structured_margin=minimum_margin,
        maximum_cells=maximum_cells,
    )
    return preflight, result


def test_d0_targets_use_global_route_and_behavioral_margins():
    routing_policy = policy()
    value = panel(routing_policy)
    _, result = teacher(value, routing_policy, minimum_margin=0.1)
    assert result.status == "READY_FOR_D0_TRAINING"
    assert result.labels is not None
    # Global structured route: cell 0 keeps candidate 0, cell 1 candidate 1.
    assert result.labels.tolist() == [[0, 1]]
    assert result.cell_available.tolist() == [[True, True]]
    assert result.cell_feasible.tolist() == [[True, True]]
    # Behavioral margin = min(threshold - exact_risk) over the critical
    # constraints for the selected candidate.
    # cell (0,0) candidate 0: min(1.0-0.5, 1.0-0.2) = 0.5
    # cell (0,1) candidate 1: min(1.0-0.2, 1.0-0.3) = 0.7
    assert result.behavioral_margin[0, 0] == pytest.approx(0.5)
    assert result.behavioral_margin[0, 1] == pytest.approx(0.7)
    assert result.target_available.tolist() == [[True, True]]
    assert result.training_rows() == (
        {
            "offline_inference_row_sha256": sha("row-0"),
            "candidate_index": 0,
            "behavioral_margin": pytest.approx(0.5),
        },
        {
            "offline_inference_row_sha256": sha("row-1"),
            "candidate_index": 1,
            "behavioral_margin": pytest.approx(0.7),
        },
    )


def test_larger_frozen_margin_admits_only_the_high_margin_cell():
    routing_policy = policy()
    value = panel(routing_policy)
    _, result = teacher(value, routing_policy, minimum_margin=0.6)
    # 0.5 < 0.6 <= 0.7 -> only the second cell is a training target.
    assert result.status == "READY_FOR_D0_TRAINING"
    assert result.target_available.tolist() == [[False, True]]
    assert len(result.training_rows()) == 1
    assert result.training_rows()[0]["offline_inference_row_sha256"] == sha(
        "row-1"
    )


def test_structured_teacher_can_disagree_with_local_unary_argmin():
    routing_policy = policy(frequency_switch_penalty=0.3)
    value = panel(routing_policy)
    preflight, result = teacher(value, routing_policy, minimum_margin=0.1)
    # Candidate 0 is locally cheaper in the first cell, but the global route
    # selects candidate 1 after switching cost and physical measure.
    assert preflight.cost[0, 0, 0] < preflight.cost[0, 0, 1]
    assert result.labels.tolist() == [[1, 1]]
    assert result.labels[0, 0] == 1
    # Candidate 1 in cell 0 sits exactly on the hole threshold -> margin 0.0.
    assert result.behavioral_margin[0, 0] == pytest.approx(0.0)
    assert result.cell_feasible[0, 0]
    assert not result.target_available[0, 0]


def test_sole_feasible_candidate_is_forced_by_feasibility():
    routing_policy = policy()
    value = panel(routing_policy)
    upper = np.asarray(value.upper).copy()
    # Make candidate 1 in cell 0 critical-infeasible; candidate 0 is forced.
    upper[0, 0, 1, 0] = 1.2
    changed = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "upper": upper}
    )
    _, result = teacher(changed, routing_policy, minimum_margin=0.1)
    assert result.labels.tolist() == [[0, 1]]
    assert result.cell_available.tolist() == [[True, True]]
    assert result.cell_feasible.tolist() == [[True, True]]
    assert result.behavioral_margin[0, 0] == pytest.approx(0.5)
    assert result.target_available[0, 0]
    assert result.training_rows()[0]["candidate_index"] == 0


def test_nonunique_route_produces_no_d0_labels_or_margins():
    routing_policy = policy(
        secondary_weights=(),
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
    )
    value = panel(
        routing_policy,
        upper=np.full((1, 2, 2, 3), 0.2, dtype=np.float64),
    )
    _, result = teacher(value, routing_policy)
    assert result.status == "UNAVAILABLE_NONUNIQUE_ROUTE"
    assert result.labels is None
    assert result.cell_available is None
    assert result.behavioral_margin is None
    assert result.target_available is None
    assert result.training_rows() == ()


def test_missing_required_risk_cell_fails_closed():
    routing_policy = policy()
    value = panel(routing_policy, allowed=(False, True))
    upper = np.asarray(value.upper).copy()
    available = np.asarray(value.available).copy()
    # The first cell has no confidently feasible candidate (evidence missing).
    upper[0, 0] = 0.0
    available[0, 0] = False
    blocked = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "upper": upper, "available": available}
    )
    _, result = teacher(blocked, routing_policy)
    assert result.status == "UNAVAILABLE_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert result.labels is None
    assert result.behavioral_margin is None
    assert result.training_rows() == ()


def test_unique_route_with_no_high_margin_cells_is_a_nontraining_result():
    routing_policy = policy()
    value = panel(routing_policy)
    _, result = teacher(value, routing_policy, minimum_margin=1.0)
    # Every behavioral margin (0.5, 0.7) is below the frozen minimum of 1.0.
    assert result.status == "UNAVAILABLE_NO_HIGH_MARGIN_CELLS"
    assert result.labels.tolist() == [[0, 1]]
    assert not result.target_available.any()
    assert result.training_rows() == ()


def test_teacher_arrays_are_read_only_and_identity_binds_evidence_margin_and_route():
    routing_policy = policy()
    value = panel(routing_policy)
    _, first = teacher(value, routing_policy, minimum_margin=0.1)
    _, changed_margin = teacher(value, routing_policy, minimum_margin=0.6)
    _, changed_evidence = teacher(
        value,
        routing_policy,
        minimum_margin=0.1,
        exact_evidence=sha("other-evidence"),
    )
    assert first.labels is not None
    for array in (
        first.labels,
        first.cell_available,
        first.cell_feasible,
        first.behavioral_margin,
        first.target_available,
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        first.labels[0, 0] = 1
    assert len(
        {
            first.sha256(candidate_count=2),
            changed_margin.sha256(candidate_count=2),
            changed_evidence.sha256(candidate_count=2),
        }
    ) == 3


def test_teacher_validation_rejects_target_and_feasibility_corruption():
    routing_policy = policy()
    value = panel(routing_policy)
    _, result = teacher(value, routing_policy, minimum_margin=0.1)

    target = np.asarray(result.target_available).copy()
    target[0, 0] = False
    with pytest.raises(
        D0StructuredTeacherError,
        match="target availability differs",
    ):
        D0StructuredTeacherV1(
            **{**result.__dict__, "target_available": target}
        ).validate(candidate_count=2)

    margin = np.asarray(result.behavioral_margin).copy()
    # A negative margin contradicts the stored feasibility of the cell.
    margin[0, 0] = -0.5
    with pytest.raises(
        D0StructuredTeacherError,
        match="feasibility differs",
    ):
        D0StructuredTeacherV1(
            **{**result.__dict__, "behavioral_margin": margin}
        ).validate(candidate_count=2)


def test_invalid_evidence_margin_and_grid_limit_are_refused():
    routing_policy = policy()
    value = panel(routing_policy)
    preflight = preflight_partitioned_upper_risks(value, routing_policy)
    with pytest.raises(D0StructuredTeacherError, match="canonical sha256"):
        build_d0_structured_teacher(
            value,
            preflight,
            routing_policy,
            exact_evidence_sha256="bad",
            minimum_structured_margin=0.1,
        )
    with pytest.raises(
        D0StructuredTeacherError,
        match="strictly positive",
    ):
        build_d0_structured_teacher(
            value,
            preflight,
            routing_policy,
            exact_evidence_sha256=sha("exact"),
            minimum_structured_margin=0.0,
        )
    with pytest.raises(Exception, match="above maximum_cells"):
        build_d0_structured_teacher(
            value,
            preflight,
            routing_policy,
            exact_evidence_sha256=sha("exact"),
            minimum_structured_margin=0.1,
            maximum_cells=1,
        )
