"""Solver-independent exhaustive checks for the counterfactual-risk router."""

from itertools import product

import numpy as np
import pytest

from audio_extract.counterfactual_risk_router import (
    RiskPanel,
    RiskPanelIdentity,
    RiskRouterConfig,
    solve_risk_route,
)


def _sha(digit: str) -> str:
    return "sha256:" + digit * 64


def _identity(candidates: int) -> RiskPanelIdentity:
    return RiskPanelIdentity(
        source_pcm_sha256=_sha("0"),
        candidate_ids=tuple(
            _sha(str(index + 1)) for index in range(candidates)
        ),
        metric_names=("voice", "hole", "artifact", "secondary"),
        feature_contract_sha256=_sha("7"),
        risk_model_sha256=_sha("8"),
    )


def _config(
    *,
    temporal: float,
    frequency: float,
    conservative: int = 0,
) -> RiskRouterConfig:
    return RiskRouterConfig(
        critical_thresholds=(
            ("voice", 1.0),
            ("hole", 1.0),
            ("artifact", 1.0),
        ),
        secondary_weights=(("secondary", 0.2),),
        critical_slack_weight=1.0,
        temporal_switch_penalty=temporal,
        frequency_switch_penalty=frequency,
        conservative_index=conservative,
        milp_time_limit_seconds=30.0,
        milp_relative_gap=0.0,
        integrality_tolerance=1e-7,
        objective_tolerance=1e-8,
    )


def _unary_and_feasible(panel: RiskPanel, config: RiskRouterConfig):
    """Independent transcription of the public v0 decoder policy."""

    upper = np.asarray(panel.upper, dtype=np.float64)
    available = np.asarray(panel.available, dtype=bool)
    metric = {
        name: index
        for index, name in enumerate(panel.identity.metric_names)
    }
    normalized = []
    feasible = np.ones(upper.shape[:3], dtype=bool)
    for name, threshold in config.critical_thresholds:
        index = metric[name]
        present = available[..., index]
        value = np.where(present, upper[..., index], 0.0)
        feasible &= present
        feasible &= value <= threshold + config.feasibility_tolerance
        normalized.append(value / threshold)
    cost = config.critical_slack_weight * np.max(
        np.stack(normalized, axis=-1), axis=-1
    )
    for name, weight in config.secondary_weights:
        index = metric[name]
        present = available[..., index]
        value = np.where(present, upper[..., index], 0.0)
        feasible &= present
        cost += weight * value
    return cost, feasible


def _objective(labels, cost, config):
    labels = np.asarray(labels, dtype=np.int32)
    selected = np.take_along_axis(
        cost, labels[..., None], axis=-1
    )[..., 0]
    cells = labels.size
    temporal = int(np.count_nonzero(labels[1:] != labels[:-1]))
    frequency = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    return (
        float(selected.mean())
        + config.temporal_switch_penalty * temporal / cells
        + config.frequency_switch_penalty * frequency / cells
    )


def _brute_force(panel: RiskPanel, config: RiskRouterConfig):
    cost, feasible = _unary_and_feasible(panel, config)
    time, bands, candidates = feasible.shape
    if np.any(feasible.sum(axis=-1) == 0):
        return None
    best_value = np.inf
    best_labels = []
    for flattened in product(range(candidates), repeat=time * bands):
        labels = np.asarray(flattened, dtype=np.int32).reshape(time, bands)
        allowed = np.take_along_axis(
            feasible, labels[..., None], axis=-1
        )[..., 0]
        if not np.all(allowed):
            continue
        value = _objective(labels, cost, config)
        if value < best_value - 1e-12:
            best_value = value
            best_labels = [labels.copy()]
        elif abs(value - best_value) <= 1e-12:
            best_labels.append(labels.copy())
    assert best_labels
    return best_value, best_labels


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("shape", ((1, 1), (2, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("penalties", ((0.0, 0.0), (0.1, 0.2), (1.0, 0.5)))
def test_milp_matches_exhaustive_global_optimum(seed, shape, penalties):
    rng = np.random.default_rng(seed)
    time, bands = shape
    candidates = 2
    values = rng.uniform(0.02, 1.25, size=(time, bands, candidates, 4))
    available = rng.random(values.shape) > 0.08
    # Ensure at least one fully observed, critical-feasible candidate per cell so
    # this test exercises the route rather than the abstention branch.
    for t in range(time):
        for b in range(bands):
            chosen = (seed + t + b) % candidates
            values[t, b, chosen, :3] = rng.uniform(0.05, 0.8, size=3)
            available[t, b, chosen, :] = True
    panel = RiskPanel(_identity(candidates), values, available)
    config = _config(
        temporal=penalties[0], frequency=penalties[1]
    )
    reference = _brute_force(panel, config)
    assert reference is not None
    optimum, labels = reference
    result = solve_risk_route(panel, config)
    assert result.status == "ROUTE"
    assert result.objective == pytest.approx(optimum, abs=1e-8, rel=1e-8)
    assert any(np.array_equal(result.labels, item) for item in labels)


@pytest.mark.parametrize("location", ((0, 0), (1, 1)))
def test_any_locally_infeasible_cell_forces_explicit_whole_track_abstention(
    location,
):
    values = np.full((2, 2, 2, 4), 0.2, dtype=np.float64)
    available = np.ones_like(values, dtype=bool)
    t, b = location
    values[t, b, :, 0] = 1.5
    panel = RiskPanel(_identity(2), values, available)
    config = _config(temporal=0.1, frequency=0.1)
    result = solve_risk_route(panel, config)
    assert result.status == "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE"
    assert result.infeasible_cells == (location,)
    assert np.all(result.labels == config.conservative_index)


def test_route_hash_is_sensitive_to_availability_even_when_labels_and_values_match():
    values = np.asarray(
        [[[[0.1, 0.1, 0.1, 0.0], [0.2, 0.2, 0.2, 0.0]]]],
        dtype=np.float64,
    )
    first_available = np.ones_like(values, dtype=bool)
    second_available = first_available.copy()
    # Candidate 1 remains irrelevant to the selected route, but changing its
    # evidence availability must still change the bound plan identity.
    second_available[0, 0, 1, 3] = False
    config = _config(temporal=0.0, frequency=0.0)
    first = solve_risk_route(
        RiskPanel(_identity(2), values, first_available), config
    )
    second = solve_risk_route(
        RiskPanel(_identity(2), values, second_available), config
    )
    assert first.labels.tolist() == second.labels.tolist()
    assert first.routing_plan_sha256 != second.routing_plan_sha256
