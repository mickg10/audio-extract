from itertools import product

import numpy as np
import pytest

from audio_extract.counterfactual_risk_router import (
    RiskPanel,
    RiskPanelIdentity,
    RiskRouterConfig,
    _candidate_costs,
    build_local_counterfactual_teacher_targets,
    solve_risk_route,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def identity(candidates: int) -> RiskPanelIdentity:
    return RiskPanelIdentity(
        source_pcm_sha256=sha("0"),
        candidate_ids=tuple(
            "sha256:" + f"{index + 1:064x}" for index in range(candidates)
        ),
        metric_names=("voice", "hole", "artifact"),
        feature_contract_sha256=sha("a"),
        risk_model_sha256=sha("b"),
    )


def config(**kwargs) -> RiskRouterConfig:
    values = dict(
        critical_thresholds=(
            ("voice", 1.0),
            ("hole", 1.0),
            ("artifact", 1.0),
        ),
        critical_slack_weight=1.0,
        temporal_switch_penalty=0.07,
        frequency_switch_penalty=0.11,
        conservative_index=0,
        milp_time_limit_seconds=30.0,
        milp_relative_gap=0.0,
    )
    values.update(kwargs)
    return RiskRouterConfig(**values)


def risk_panel(values: np.ndarray, available=None) -> RiskPanel:
    values = np.asarray(values, dtype=np.float64)
    if available is None:
        available = np.ones_like(values, dtype=bool)
    return RiskPanel(identity(values.shape[2]), values, available)


def exhaustive_objective(panel: RiskPanel, cfg: RiskRouterConfig):
    cost, feasible = _candidate_costs(panel, cfg)
    time, bands, candidates = cost.shape
    cells = time * bands
    best = None
    best_labels = []
    for flat in product(range(candidates), repeat=cells):
        labels = np.asarray(flat, dtype=np.int32).reshape(time, bands)
        selected = np.take_along_axis(
            feasible, labels[..., None], axis=-1
        )[..., 0]
        if not np.all(selected):
            continue
        data = float(np.take_along_axis(
            cost, labels[..., None], axis=-1
        )[..., 0].mean())
        temporal = int(np.count_nonzero(labels[1:] != labels[:-1]))
        frequency = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
        total = (
            data
            + cfg.temporal_switch_penalty * temporal / cells
            + cfg.frequency_switch_penalty * frequency / cells
        )
        if best is None or total < best - 1e-12:
            best = total
            best_labels = [labels]
        elif abs(total - best) <= 1e-12:
            best_labels.append(labels)
    return best, best_labels


@pytest.mark.parametrize("seed", range(8))
def test_milp_matches_exhaustive_small_grid(seed):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.05, 1.25, size=(2, 2, 3, 3))
    values[:, :, 0, :] = rng.uniform(0.05, 0.8, size=(2, 2, 3))
    panel = risk_panel(values)
    cfg = config()
    result = solve_risk_route(panel, cfg)
    best, labels = exhaustive_objective(panel, cfg)
    assert result.status == "ROUTE"
    assert result.objective == pytest.approx(best, abs=1e-9)
    assert any(np.array_equal(result.labels, expected) for expected in labels)


@pytest.mark.parametrize("seed", range(12))
def test_teacher_label_and_margin_match_direct_enumeration(seed):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.01, 1.4, size=(3, 2, 4, 3))
    panel = risk_panel(values)
    cfg = config(teacher_near_tie_margin=0.08)
    cost, feasible = _candidate_costs(panel, cfg)
    teacher = build_local_counterfactual_teacher_targets(panel, cfg)

    for time in range(values.shape[0]):
        for band in range(values.shape[1]):
            allowed = np.flatnonzero(feasible[time, band])
            if allowed.size == 0:
                assert teacher.labels[time, band] == -1
                assert not teacher.available[time, band]
                continue
            order = allowed[
                np.argsort(cost[time, band, allowed], kind="stable")
            ]
            assert teacher.labels[time, band] == order[0]
            if len(order) == 1:
                assert teacher.available[time, band]
                assert np.isfinite(teacher.margins[time, band])
            else:
                margin = cost[time, band, order[1]] - cost[
                    time, band, order[0]
                ]
                assert teacher.margins[time, band] == pytest.approx(margin)
                assert teacher.available[time, band] == (margin >= 0.08)


def test_unavailable_values_do_not_change_panel_or_plan_hash():
    values = np.asarray([[[[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]]])
    available = np.ones_like(values, dtype=bool)
    available[0, 0, 1, 2] = False

    first = values.copy()
    first[0, 0, 1, 2] = np.nan
    second = values.copy()
    second[0, 0, 1, 2] = 1e200

    panel_a = risk_panel(first, available)
    panel_b = risk_panel(second, available)
    assert panel_a.sha256 == panel_b.sha256
    result_a = solve_risk_route(panel_a, config())
    result_b = solve_risk_route(panel_b, config())
    assert result_a.routing_plan_sha256 == result_b.routing_plan_sha256


def test_feasibility_tolerance_is_explicit_and_identity_bearing():
    values = np.asarray([[[[1.0001, 0.2, 0.2], [0.9, 0.9, 0.9]]]])
    panel = risk_panel(values)
    strict = solve_risk_route(
        panel, config(feasibility_tolerance=0.0, critical_slack_weight=0.0)
    )
    tolerant_cfg = config(
        feasibility_tolerance=0.001, critical_slack_weight=0.0
    )
    tolerant = solve_risk_route(panel, tolerant_cfg)
    assert strict.labels.tolist() == [[1]]
    assert tolerant.labels.tolist() == [[0]]
    assert strict.routing_plan_sha256 != tolerant.routing_plan_sha256


def test_every_selected_cell_satisfies_every_critical_upper_bound():
    rng = np.random.default_rng(99)
    values = rng.uniform(0.0, 1.5, size=(3, 2, 4, 3))
    values[:, :, 0, :] = 0.5
    panel = risk_panel(values)
    cfg = config()
    result = solve_risk_route(panel, cfg)
    assert result.status == "ROUTE"
    for time in range(3):
        for band in range(2):
            selected = result.labels[time, band]
            assert np.all(values[time, band, selected] <= 1.0)


def test_local_hard_feasibility_is_intentionally_strict():
    values = np.full((2, 1, 2, 3), 0.4)
    values[0, 0, 0, 0] = 1.2
    values[1, 0, 1, 0] = 1.2
    route = solve_risk_route(risk_panel(values), config())
    assert route.status == "ROUTE"
    assert route.labels[:, 0].tolist() == [1, 0]

    values[0, 0, 1, 1] = 1.2
    abstain = solve_risk_route(risk_panel(values), config())
    assert abstain.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert abstain.infeasible_cells == ((0, 0),)
