import numpy as np
import pytest

from audio_extract.oracle_routing_math_v2 import (
    RoutingMathError,
    best_whole_track,
    build_exact_cell_quadratic,
    project_simplex,
    route_objective,
    solve_o2_global,
    solve_o3_convex,
    stack_cells,
    unary_costs,
)


def basis(alpha, beta=None, residual=None):
    alpha = np.asarray(alpha, dtype=np.complex128)
    beta = np.zeros_like(alpha) if beta is None else np.asarray(
        beta, dtype=np.complex128
    )
    accompaniment = np.asarray([1.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 1.0], dtype=np.complex128)
    if residual is None:
        residual = np.zeros((len(alpha), 2), dtype=np.complex128)
    candidates = (
        alpha[:, None] * accompaniment[None]
        + beta[:, None] * vocal[None]
        + np.asarray(residual, dtype=np.complex128)
    )
    return candidates, accompaniment, vocal


def test_source_coordinate_cost_rejects_accompaniment_amplification():
    candidates, accompaniment, vocal = basis([1.0, 2.0], beta=[0.2, 0.0])
    cell = build_exact_cell_quadratic(candidates, accompaniment, vocal)
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "source_coordinates"
    assert costs[0] < costs[1]


def test_source_coordinate_cost_penalizes_phase_inversion():
    candidates, accompaniment, vocal = basis([1.0, -1.0])
    cell = build_exact_cell_quadratic(candidates, accompaniment, vocal)
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] == pytest.approx(4.0, rel=1e-8)


def test_no_vocal_cell_uses_direct_fallback():
    accompaniment = np.asarray([1.0, 0.5], dtype=np.complex128)
    vocal = np.zeros_like(accompaniment)
    candidates = np.stack((accompaniment, 0.5 * accompaniment))
    cell = build_exact_cell_quadratic(candidates, accompaniment, vocal)
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "no_vocal_direct_fallback"
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] > 0.0


def test_vocal_only_cell_uses_direct_fallback():
    accompaniment = np.zeros(2, dtype=np.complex128)
    vocal = np.asarray([1.0, 0.25], dtype=np.complex128)
    candidates = np.stack((np.zeros_like(vocal), vocal))
    cell = build_exact_cell_quadratic(candidates, accompaniment, vocal)
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "vocal_only_direct_fallback"
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] > 0.0


def test_simplex_projection_is_nonnegative_and_normalized():
    result = project_simplex(
        np.asarray([[-1.0, 2.0, 0.0], [0.2, 0.3, 0.7]])
    )
    assert np.all(result >= 0.0)
    assert np.allclose(result.sum(-1), 1.0)


def test_o3_finds_exact_interpolation_between_two_imperfect_vertices():
    candidates, accompaniment, vocal = basis([0.8, 1.2])
    cell = build_exact_cell_quadratic(
        candidates, accompaniment, vocal,
        voice_weight=0.0, artifact_weight=0.0,
    )
    grid = stack_cells([[cell]])
    o1_index, _ = best_whole_track(grid)
    result = solve_o3_convex(
        grid, temporal_smoothness=0.0, frequency_smoothness=0.0,
        tolerance=1e-9, max_iterations=10_000, o1_index=o1_index,
        starts=("uniform", "O1", "independent"),
    )
    assert result.weights[0, 0, 0] == pytest.approx(0.5, abs=1e-5)
    assert result.weights[0, 0, 1] == pytest.approx(0.5, abs=1e-5)
    assert result.objective < unary_costs(grid).min() - 1e-5
    assert result.projected_gradient_norm <= 1e-9


def test_o2_globally_routes_complementary_time_cells():
    candidates1, accompaniment1, vocal1 = basis([1.0, 0.5])
    candidates2, accompaniment2, vocal2 = basis([0.5, 1.0])
    grid = stack_cells([[
        build_exact_cell_quadratic(candidates1, accompaniment1, vocal1)
    ], [
        build_exact_cell_quadratic(candidates2, accompaniment2, vocal2)
    ]])
    result = solve_o2_global(
        grid, temporal_switch_penalty=0.0, frequency_switch_penalty=0.0
    )
    assert result.labels.tolist() == [[0], [1]]
    assert result.temporal_switches == 1


def test_all_derived_quadratics_are_psd_and_finite():
    candidates, accompaniment, vocal = basis(
        [0.9, 1.1], beta=[0.2, -0.1],
        residual=np.asarray([[0.0, 0.1], [0.1, 0.0]]),
    )
    cell = build_exact_cell_quadratic(candidates, accompaniment, vocal)
    assert float(np.linalg.eigvalsh(cell.Q).min()) >= -1e-10
    grid = stack_cells([[cell]])
    weights = np.full((1, 1, 2), 0.5)
    assert np.isfinite(route_objective(weights, grid)[0])


def test_whole_track_selection_includes_no_vocal_fallback_cell():
    candidates1, accompaniment1, vocal1 = basis([1.0, 0.8])
    first = build_exact_cell_quadratic(
        candidates1, accompaniment1, vocal1
    )
    accompaniment2 = np.asarray([1.0, 0.2], dtype=np.complex128)
    vocal2 = np.zeros_like(accompaniment2)
    candidates2 = np.stack((np.zeros_like(accompaniment2), accompaniment2))
    second = build_exact_cell_quadratic(
        candidates2, accompaniment2, vocal2
    )
    grid = stack_cells([[first], [second]])
    index, costs = best_whole_track(grid)
    assert second.mode == "no_vocal_direct_fallback"
    assert index == 1
    assert costs[1] < costs[0]


def test_o3_refuses_an_uncertified_iteration_limit():
    candidates, accompaniment, vocal = basis([0.8, 1.2])
    grid = stack_cells([[
        build_exact_cell_quadratic(
            candidates, accompaniment, vocal,
            voice_weight=0.0, artifact_weight=0.0,
        )
    ]])
    with pytest.raises(RoutingMathError, match="did not converge"):
        solve_o3_convex(
            grid, temporal_smoothness=0.0, frequency_smoothness=0.0,
            tolerance=1e-20, max_iterations=1,
            starts=("O1",),
        )
