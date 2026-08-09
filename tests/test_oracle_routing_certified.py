import numpy as np
import pytest

from audio_extract.oracle_routing_certified import (
    CertifiedRoutingError,
    CertifiedRoutingConfig,
    best_whole_track,
    build_cell_quadratic,
    project_simplex,
    route_objective,
    solve_convex_certified,
    solve_discrete_global,
    stack_cell_grid,
    unary_costs,
)


def _basis(alpha, beta=None, residual=None):
    alpha = np.asarray(alpha, dtype=np.complex128)
    beta = np.zeros_like(alpha) if beta is None else np.asarray(
        beta, dtype=np.complex128
    )
    candidates = len(alpha)
    accompaniment = np.asarray([1.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 1.0], dtype=np.complex128)
    if residual is None:
        residual = np.zeros((candidates, 2), dtype=np.complex128)
    mixture_candidates = (
        alpha[:, None] * accompaniment[None]
        + beta[:, None] * vocal[None]
        + np.asarray(residual, dtype=np.complex128)
    )
    return mixture_candidates, accompaniment, vocal


def _fast_config(**kwargs):
    values = dict(
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
        temporal_weight_smoothness=0.0,
        frequency_weight_smoothness=0.0,
        projected_gradient_tolerance=1e-9,
        max_projected_gradient_iterations=10_000,
    )
    values.update(kwargs)
    return CertifiedRoutingConfig(**values)


def test_source_coordinate_cost_rejects_amplification_gaming():
    candidates, accompaniment, vocal = _basis(
        [1.0, 2.0], beta=[0.2, 0.0]
    )
    cell = build_cell_quadratic(
        candidates, accompaniment, vocal, _fast_config()
    )
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "source_coordinates"
    assert costs[0] < costs[1]


def test_source_coordinate_cost_penalizes_phase_inversion():
    candidates, accompaniment, vocal = _basis([1.0, -1.0])
    cell = build_cell_quadratic(
        candidates, accompaniment, vocal, _fast_config(
            voice_weight=0.0, residual_weight=0.0
        )
    )
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] == pytest.approx(4.0, rel=1e-8)


def test_no_vocal_cell_uses_direct_fallback_and_preserves_orchestra():
    accompaniment = np.asarray([1.0, 0.5], dtype=np.complex128)
    vocal = np.zeros_like(accompaniment)
    candidates = np.stack((accompaniment, 0.5 * accompaniment))
    cell = build_cell_quadratic(
        candidates, accompaniment, vocal, _fast_config()
    )
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "no_vocal_direct_fallback"
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] > 0.0


def test_vocal_only_cell_uses_direct_fallback_and_prefers_silence():
    accompaniment = np.zeros(2, dtype=np.complex128)
    vocal = np.asarray([1.0, 0.25], dtype=np.complex128)
    candidates = np.stack((np.zeros_like(vocal), vocal))
    cell = build_cell_quadratic(
        candidates, accompaniment, vocal, _fast_config()
    )
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "vocal_only_direct_fallback"
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] > 0.0


def test_simplex_projection_is_nonnegative_and_normalized():
    result = project_simplex(
        np.asarray([[-1.0, 2.0, 0.0], [0.2, 0.3, 0.7]])
    )
    assert np.all(result >= 0)
    assert np.allclose(result.sum(axis=-1), 1.0)


def test_certified_o3_finds_exact_interpolation():
    candidates, accompaniment, vocal = _basis([0.8, 1.2])
    cell = build_cell_quadratic(
        candidates, accompaniment, vocal,
        _fast_config(voice_weight=0.0, residual_weight=0.0),
    )
    grid = stack_cell_grid([[cell]])
    o1, _ = best_whole_track(grid)
    result = solve_convex_certified(
        grid,
        _fast_config(voice_weight=0.0, residual_weight=0.0),
        o1_index=o1,
        starts=("uniform", "O1", "independent"),
    )
    assert result.converged
    assert result.projected_gradient_norm <= 1e-9
    assert result.weights[0, 0, 0] == pytest.approx(0.5, abs=1e-5)
    assert result.weights[0, 0, 1] == pytest.approx(0.5, abs=1e-5)
    assert result.objective < unary_costs(grid).min() - 1e-5


def test_certified_o3_requires_every_declared_start_to_converge():
    candidates, accompaniment, vocal = _basis([0.8, 1.2])
    config = _fast_config(
        voice_weight=0.0,
        residual_weight=0.0,
        max_projected_gradient_iterations=1,
    )
    grid = stack_cell_grid([[
        build_cell_quadratic(candidates, accompaniment, vocal, config)
    ]])
    with pytest.raises(CertifiedRoutingError, match="every declared start"):
        solve_convex_certified(
            grid, config, o1_index=0, starts=("uniform", "O1")
        )


def test_global_o2_routes_complementary_time_cells():
    candidates_a, accompaniment_a, vocal_a = _basis([1.0, 0.5])
    candidates_b, accompaniment_b, vocal_b = _basis([0.5, 1.0])
    cell_a = build_cell_quadratic(
        candidates_a, accompaniment_a, vocal_a, _fast_config()
    )
    cell_b = build_cell_quadratic(
        candidates_b, accompaniment_b, vocal_b, _fast_config()
    )
    grid = stack_cell_grid([[cell_a], [cell_b]])
    result = solve_discrete_global(grid, _fast_config())
    assert result.labels.tolist() == [[0], [1]]
    assert result.temporal_switches == 1
    assert result.mip_gap in (None, pytest.approx(0.0))


def test_derived_quadratic_is_psd_and_objective_is_finite():
    candidates, accompaniment, vocal = _basis(
        [0.9, 1.1],
        beta=[0.2, -0.1],
        residual=np.asarray([[0.0, 0.1], [0.1, 0.0]]),
    )
    cell = build_cell_quadratic(
        candidates, accompaniment, vocal, _fast_config()
    )
    assert np.linalg.eigvalsh(cell.Q).min() >= -1e-10
    grid = stack_cell_grid([[cell]])
    weights = np.full((1, 1, 2), 0.5)
    assert np.isfinite(route_objective(weights, grid)[0])


def test_whole_track_selection_includes_direct_fallback_cells():
    candidates_a, accompaniment_a, vocal_a = _basis([1.0, 0.8])
    cell_a = build_cell_quadratic(
        candidates_a, accompaniment_a, vocal_a, _fast_config()
    )

    accompaniment_b = np.asarray([1.0, 0.2], dtype=np.complex128)
    vocal_b = np.zeros_like(accompaniment_b)
    candidates_b = np.stack((np.zeros_like(accompaniment_b), accompaniment_b))
    cell_b = build_cell_quadratic(
        candidates_b, accompaniment_b, vocal_b, _fast_config()
    )
    grid = stack_cell_grid([[cell_a], [cell_b]])
    index, costs = best_whole_track(grid)
    assert cell_b.mode == "no_vocal_direct_fallback"
    assert index == 1
    assert costs[1] < costs[0]
