import numpy as np
import pytest

from audio_extract.counterfactual_risk_router import (
    RiskPanel,
    RiskPanelIdentity,
    RiskRouterConfig,
    build_exact_teacher_targets,
    solve_risk_route,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def identity(order=("a", "b", "c")):
    digits = {"a": "1", "b": "2", "c": "3"}
    return RiskPanelIdentity(
        candidate_ids=tuple(sha(digits[name]) for name in order),
        metric_names=("voice", "hole", "artifact", "secondary"),
        feature_contract_sha256=sha("4"),
        risk_model_sha256=sha("5"),
        query_encoder_sha256=sha("6"),
    )


def config(**kwargs):
    values = dict(
        critical_thresholds=(
            ("voice", 1.0),
            ("hole", 1.0),
            ("artifact", 1.0),
        ),
        secondary_weights=(("secondary", 0.1),),
        critical_slack_weight=1.0,
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
        conservative_index=0,
        teacher_near_tie_margin=0.05,
        milp_time_limit_seconds=30.0,
        milp_relative_gap=0.0,
    )
    values.update(kwargs)
    return RiskRouterConfig(**values)


def panel(values, available=None, order=("a", "b", "c")):
    values = np.asarray(values, dtype=np.float64)
    if available is None:
        available = np.ones_like(values, dtype=bool)
    return RiskPanel(identity(order), values, np.asarray(available, dtype=bool))


def test_panel_binds_candidate_and_metric_identity():
    values = np.zeros((1, 1, 3, 4))
    row = panel(values)
    assert row.validate() == (1, 1, 3, 4)
    assert row.identity.sha256.startswith("sha256:")
    with pytest.raises(ValueError, match="candidate axis"):
        RiskPanel(
            identity(), values[:, :, :2],
            np.ones_like(values[:, :, :2], bool)
        ).validate()
    with pytest.raises(ValueError, match="unique"):
        RiskPanelIdentity(
            candidate_ids=(sha("1"), sha("1")),
            metric_names=("voice",),
            feature_contract_sha256=sha("2"),
            risk_model_sha256=sha("3"),
        ).validate()


def test_exact_teacher_masks_near_ties_and_keeps_strong_margin():
    values = np.asarray([
        [[
            [0.50, 0.50, 0.50, 0.0],
            [0.51, 0.50, 0.50, 0.0],
            [0.90, 0.90, 0.90, 0.0],
        ]],
        [[
            [0.80, 0.80, 0.80, 0.0],
            [0.20, 0.20, 0.20, 0.0],
            [0.90, 0.90, 0.90, 0.0],
        ]],
    ])
    teacher = build_exact_teacher_targets(panel(values), config())
    assert teacher.labels[:, 0].tolist() == [0, 1]
    assert teacher.available[:, 0].tolist() == [False, True]
    assert teacher.margins[0, 0] == pytest.approx(0.01)
    assert teacher.margins[1, 0] == pytest.approx(0.60)


def test_global_route_selects_complementary_candidates():
    values = np.asarray([
        [[
            [0.10, 0.10, 0.10, 0.0],
            [0.80, 0.80, 0.80, 0.0],
            [0.90, 0.90, 0.90, 0.0],
        ]],
        [[
            [0.80, 0.80, 0.80, 0.0],
            [0.10, 0.10, 0.10, 0.0],
            [0.90, 0.90, 0.90, 0.0],
        ]],
    ])
    result = solve_risk_route(panel(values), config())
    assert result.status == "ROUTE"
    assert result.labels.shape == (2, 1)
    assert result.labels[:, 0].tolist() == [0, 1]
    assert result.temporal_switches == 1
    assert sum(result.selection_counts) == 2


def test_switch_penalty_can_prefer_one_whole_track_candidate():
    values = np.asarray([
        [[
            [0.10, 0.10, 0.10, 0.0],
            [0.12, 0.12, 0.12, 0.0],
            [0.90, 0.90, 0.90, 0.0],
        ]],
        [[
            [0.12, 0.12, 0.12, 0.0],
            [0.10, 0.10, 0.10, 0.0],
            [0.90, 0.90, 0.90, 0.0],
        ]],
    ])
    result = solve_risk_route(
        panel(values), config(temporal_switch_penalty=1.0)
    )
    assert result.status == "ROUTE"
    assert result.temporal_switches == 0
    assert len(set(result.labels.ravel().tolist())) == 1


def test_no_confidently_feasible_candidate_abstains_to_parent():
    values = np.asarray([
        [[
            [1.2, 0.1, 0.1, 0.0],
            [0.1, 1.2, 0.1, 0.0],
            [0.1, 0.1, 1.2, 0.0],
        ]]
    ])
    result = solve_risk_route(panel(values), config())
    assert result.status == "ABSTAIN_USE_CONSERVATIVE_WHOLE_TRACK"
    assert result.labels.tolist() == [[0]]
    assert result.infeasible_cells == ((0, 0),)
    assert result.objective is None


def test_missing_critical_prediction_cannot_be_treated_as_safe():
    values = np.asarray([
        [[
            [0.1, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.0],
            [0.3, 0.3, 0.3, 0.0],
        ]]
    ])
    available = np.ones_like(values, dtype=bool)
    available[0, 0, :, 0] = False
    result = solve_risk_route(panel(values, available), config())
    assert result.status.startswith("ABSTAIN")
    assert result.infeasible_cells == ((0, 0),)


def test_plan_has_no_channel_axis_and_identity_is_deterministic():
    values = np.asarray([
        [[
            [0.1, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.0],
            [0.3, 0.3, 0.3, 0.0],
        ]]
    ])
    first = solve_risk_route(panel(values), config())
    second = solve_risk_route(panel(values), config())
    assert first.labels.ndim == 2
    assert first.routing_plan_sha256 == second.routing_plan_sha256
    reordered = values[:, :, [1, 0, 2], :]
    third = solve_risk_route(
        panel(reordered, order=("b", "a", "c")),
        config(conservative_index=1),
    )
    assert third.routing_plan_sha256 != first.routing_plan_sha256


def test_upper_bound_crossing_threshold_changes_feasibility():
    values = np.asarray([
        [[
            [0.4, 0.4, 0.4, 0.0],
            [0.5, 0.5, 0.5, 0.0],
            [0.8, 0.8, 0.8, 0.0],
        ]]
    ])
    baseline = solve_risk_route(panel(values), config())
    assert baseline.labels.tolist() == [[0]]
    values[0, 0, 0, 0] = 1.01
    changed = solve_risk_route(panel(values), config())
    assert changed.labels.tolist() != [[0]]


def test_teacher_does_not_label_no_feasible_cell():
    values = np.asarray([
        [[
            [1.1, 1.1, 1.1, 0.0],
            [1.2, 1.2, 1.2, 0.0],
            [1.3, 1.3, 1.3, 0.0],
        ]]
    ])
    teacher = build_exact_teacher_targets(panel(values), config())
    assert teacher.labels.tolist() == [[-1]]
    assert teacher.available.tolist() == [[False]]
    assert teacher.feasible_counts.tolist() == [[0]]


def test_nonfinite_or_negative_available_risks_are_refused():
    values = np.zeros((1, 1, 3, 4))
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        panel(values).validate()
    values[0, 0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        panel(values).validate()
