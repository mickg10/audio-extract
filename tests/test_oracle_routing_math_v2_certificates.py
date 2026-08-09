import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

from audio_extract.oracle_routing_math_v2 import (
    CellQuadratic,
    QuadraticGrid,
    RoutingMathError,
    _gradient,
    _lipschitz_bound,
    best_whole_track,
    build_exact_cell_quadratic,
    one_hot,
    project_simplex,
    route_objective,
    solve_o2_global,
    solve_o3_convex,
    stack_cells,
    unary_costs,
)


def simple_cell(costs):
    costs = np.asarray(costs, dtype=np.float64)
    return CellQuadratic(
        Q=np.diag(costs),
        c=np.zeros_like(costs),
        constant=0.0,
        mode="silent_direct_fallback",
        accompaniment_energy=0.0,
        vocal_energy=0.0,
        condition_number=None,
    )


def source_cell(alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    accompaniment = np.asarray([1.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 1.0], dtype=np.complex128)
    candidates = alpha[:, None] * accompaniment[None]
    return build_exact_cell_quadratic(
        candidates,
        accompaniment,
        vocal,
        voice_weight=0.0,
        artifact_weight=0.0,
    )


def test_positive_cell_measure_changes_o1_in_the_expected_direction():
    first = simple_cell([0.0, 1.0])
    second = simple_cell([2.0, 0.0])
    equal = stack_cells([[first], [second]])
    weighted = stack_cells([[first], [second]], measure=[[100.0], [1.0]])
    assert best_whole_track(equal)[0] == 1
    assert best_whole_track(weighted)[0] == 0


def test_globally_scaling_measure_changes_no_objective_or_solution():
    grid_a = stack_cells(
        [[source_cell([1.0, 0.7])], [source_cell([0.6, 1.0])]],
        measure=[[3.0], [11.0]],
    )
    grid_b = QuadraticGrid(
        grid_a.Q,
        grid_a.c,
        grid_a.constant,
        grid_a.modes,
        np.asarray(grid_a.measure) * 37.0,
    )
    weights = np.asarray([[[0.3, 0.7]], [[0.8, 0.2]]])
    kwargs = dict(temporal_smoothness=0.13, frequency_smoothness=0.0)
    assert route_objective(weights, grid_a, **kwargs) == pytest.approx(
        route_objective(weights, grid_b, **kwargs), abs=1e-12
    )
    index_a, costs_a = best_whole_track(grid_a)
    index_b, costs_b = best_whole_track(grid_b)
    assert index_a == index_b
    assert costs_a == pytest.approx(costs_b, abs=1e-12)
    o2a = solve_o2_global(
        grid_a, temporal_switch_penalty=0.07, frequency_switch_penalty=0.0
    )
    o2b = solve_o2_global(
        grid_b, temporal_switch_penalty=0.07, frequency_switch_penalty=0.0
    )
    assert np.array_equal(o2a.labels, o2b.labels)
    assert o2a.objective == pytest.approx(o2b.objective, abs=1e-12)


def test_zero_negative_or_nonfinite_measure_is_refused():
    cell = simple_cell([0.0, 1.0])
    for measure in ([[0.0]], [[-1.0]], [[float("nan")]], [[float("inf")]]):
        with pytest.raises(ValueError, match="strictly positive"):
            stack_cells([[cell]], measure=measure)


def test_reference_floor_must_be_strictly_positive():
    accompaniment = np.asarray([1.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 1.0], dtype=np.complex128)
    candidates = np.stack((accompaniment, vocal))
    with pytest.raises(ValueError, match="reference_floor"):
        build_exact_cell_quadratic(
            candidates, accompaniment, vocal, reference_floor=0.0
        )


def test_weighted_o2_milp_matches_bruteforce_exactly():
    cells = [
        source_cell([1.0, 0.7, 1.2]),
        source_cell([0.7, 1.0, 1.2]),
        source_cell([1.2, 0.7, 1.0]),
        source_cell([1.0, 1.2, 0.7]),
    ]
    measure = np.asarray([[17.0, 3.0], [5.0, 29.0]])
    grid = stack_cells(
        [[cells[0], cells[1]], [cells[2], cells[3]]],
        measure=measure,
    )
    temporal_penalty = 0.07
    frequency_penalty = 0.11
    result = solve_o2_global(
        grid,
        temporal_switch_penalty=temporal_penalty,
        frequency_switch_penalty=frequency_penalty,
    )
    unary = unary_costs(grid)
    mu = measure / measure.sum()
    best = float("inf")
    labels_at_best = []
    for flat in itertools.product(range(3), repeat=4):
        labels = np.asarray(flat, dtype=np.int64).reshape(2, 2)
        chosen = np.take_along_axis(
            unary, labels[..., None], axis=-1
        )[..., 0]
        objective = (
            float(np.sum(mu * chosen))
            + temporal_penalty
            * np.count_nonzero(labels[1:] != labels[:-1]) / 4
            + frequency_penalty
            * np.count_nonzero(labels[:, 1:] != labels[:, :-1]) / 4
        )
        if objective < best - 1e-12:
            best = objective
            labels_at_best = [labels]
        elif abs(objective - best) <= 1e-12:
            labels_at_best.append(labels)
    assert result.objective == pytest.approx(best, abs=1e-10)
    assert any(np.array_equal(result.labels, row) for row in labels_at_best)


def test_o3_returned_mapping_certificate_is_recomputed_at_returned_iterate():
    grid = stack_cells(
        [
            [source_cell([0.8, 1.2, 1.05]), source_cell([1.15, 0.75, 1.0])],
            [source_cell([0.9, 1.1, 0.7]), source_cell([1.3, 0.85, 1.0])],
        ],
        measure=[[13.0, 2.0], [7.0, 19.0]],
    )
    o1, _ = best_whole_track(grid)
    result = solve_o3_convex(
        grid,
        temporal_smoothness=0.03,
        frequency_smoothness=0.04,
        tolerance=1e-8,
        max_iterations=50_000,
        o1_index=o1,
        starts=("uniform", "O1", "independent"),
    )
    step = 1.0 / _lipschitz_bound(
        grid, temporal_smoothness=0.03, frequency_smoothness=0.04
    )
    gradient = _gradient(
        result.weights,
        grid,
        temporal_smoothness=0.03,
        frequency_smoothness=0.04,
    )
    projected = project_simplex(result.weights - step * gradient)
    independent = float(
        np.max(np.abs((result.weights - projected) / step), initial=0.0)
    )
    assert result.projected_gradient_norm == pytest.approx(
        independent, rel=1e-12, abs=1e-12
    )
    assert independent <= 1e-8
    assert set(result.final_objectives) == {"uniform", "O1", "independent"}
    assert result.objective_spread <= max(1e-10, 10e-8)


def test_o3_requires_every_declared_start_to_converge():
    grid = stack_cells([[source_cell([0.6, 1.2, 1.6])]])
    with pytest.raises(RoutingMathError, match="every declared start"):
        solve_o3_convex(
            grid,
            temporal_smoothness=0.0,
            frequency_smoothness=0.0,
            tolerance=1e-14,
            max_iterations=1,
            starts=("uniform", "O1"),
        )


def test_o3_matches_independent_slsqp_on_small_weighted_grid():
    grid = stack_cells(
        [
            [source_cell([0.75, 1.2, 1.05]), source_cell([1.1, 0.8, 1.3])],
            [source_cell([0.9, 1.15, 0.7]), source_cell([1.25, 0.85, 1.0])],
        ],
        measure=[[11.0, 2.0], [5.0, 17.0]],
    )
    temporal = 0.05
    frequency = 0.03
    o1, _ = best_whole_track(grid)
    result = solve_o3_convex(
        grid,
        temporal_smoothness=temporal,
        frequency_smoothness=frequency,
        tolerance=2e-8,
        max_iterations=100_000,
        o1_index=o1,
        starts=("uniform", "O1", "independent"),
    )
    t, b, k = grid.validate()

    def unpack(flat):
        return np.asarray(flat, dtype=np.float64).reshape(t, b, k)

    measure = grid.normalized_measure()
    count = float(t * b)

    def objective(flat):
        value = unpack(flat)
        cells = (
            np.einsum("tbk,tbkl,tbl->tb", value, grid.Q, value)
            - 2.0 * np.einsum("tbk,tbk->tb", grid.c, value)
            + grid.constant
        )
        return float(
            np.sum(measure * cells)
            + temporal * np.square(value[1:] - value[:-1]).sum() / count
            + frequency * np.square(value[:, 1:] - value[:, :-1]).sum() / count
        )

    def jacobian(flat):
        return _gradient(
            unpack(flat),
            grid,
            temporal_smoothness=temporal,
            frequency_smoothness=frequency,
        ).reshape(-1)

    constraints = [
        {
            "type": "eq",
            "fun": lambda flat, index=index: (
                unpack(flat).reshape(-1, k)[index].sum() - 1.0
            ),
        }
        for index in range(t * b)
    ]
    independent = minimize(
        objective,
        np.full(t * b * k, 1.0 / k),
        method="SLSQP",
        jac=jacobian,
        bounds=[(0.0, 1.0)] * (t * b * k),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 20_000},
    )
    assert independent.success, independent.message
    assert result.objective == pytest.approx(
        float(independent.fun), rel=2e-7, abs=2e-8
    )
