import itertools

import numpy as np
import pytest

from audio_extract.oracle_routing_certified_v2 import (
    CellQuadratic,
    CertifiedRoutingConfig,
    CertifiedRoutingError,
    best_whole_track,
    build_cell_quadratic,
    build_spectral_quadratic_grid,
    cell_measure_weights,
    one_hot_weights,
    project_simplex,
    route_objective,
    solve_convex_certified,
    solve_discrete_global,
    stabilize_psd,
    stack_cell_grid,
    unary_costs,
)


def _basis(alpha, beta=None, residual=None):
    alpha = np.asarray(alpha, dtype=np.complex128)
    beta = np.zeros_like(alpha) if beta is None else np.asarray(beta, dtype=np.complex128)
    accompaniment = np.asarray([1.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 1.0], dtype=np.complex128)
    if residual is None:
        residual = np.zeros((len(alpha), 2), dtype=np.complex128)
    estimates = (
        alpha[:, None] * accompaniment[None]
        + beta[:, None] * vocal[None]
        + np.asarray(residual, dtype=np.complex128)
    )
    return estimates, accompaniment, vocal


def _config(**kwargs):
    values = dict(
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
        temporal_weight_smoothness=0.0,
        frequency_weight_smoothness=0.0,
        projected_gradient_tolerance=1e-9,
        max_projected_gradient_iterations=20_000,
    )
    values.update(kwargs)
    return CertifiedRoutingConfig(**values)


def _simplex(rng, candidates):
    value = rng.random(candidates)
    return value / value.sum()


def _unary_cell(values, *, tolerance=1e-10):
    values = np.asarray(values, dtype=np.float64)
    return CellQuadratic(
        Q=np.diag(values), c=np.zeros_like(values), constant=0.0,
        mode="synthetic_direct", accompaniment_energy=1.0,
        vocal_energy=0.0, condition_number=None,
        psd_relative_tolerance=tolerance,
    )


def _unary_grid(unary, *, cell_weights=None):
    unary = np.asarray(unary, dtype=np.float64)
    cells = [
        [_unary_cell(unary[t, b]) for b in range(unary.shape[1])]
        for t in range(unary.shape[0])
    ]
    return stack_cell_grid(cells, cell_weights=cell_weights)


def test_config_refuses_zero_reference_floor():
    with pytest.raises(ValueError, match="reference_floor"):
        CertifiedRoutingConfig(
            reference_floor=0.0, reference_floor_relative=0.0
        ).validate()


def test_config_identity_materializes_exact_float_text():
    identity = CertifiedRoutingConfig().identity_dict()
    assert identity["ridge_relative"] == "1e-08"
    assert identity["max_projected_gradient_iterations"] == 20_000


def test_scale_relative_psd_repairs_roundoff_and_refuses_material_negative():
    scale = 1e12
    tiny_negative = np.diag([scale, -0.5e-10 * scale])
    repaired, facts = stabilize_psd(
        tiny_negative, relative_tolerance=1e-10
    )
    assert facts["diagonal_shift"] > 0.0
    assert np.linalg.eigvalsh(repaired).min() >= -1e-5
    material = np.diag([scale, -2.0e-10 * scale])
    with pytest.raises(CertifiedRoutingError, match="materially non-PSD"):
        stabilize_psd(material, relative_tolerance=1e-10)


def test_source_coordinate_cost_rejects_amplification_gaming():
    estimates, accompaniment, vocal = _basis([1.0, 2.0], beta=[0.2, 0.0])
    cell = build_cell_quadratic(estimates, accompaniment, vocal, _config())
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "source_coordinates"
    assert costs[0] < costs[1]


def test_source_coordinate_cost_penalizes_phase_inversion():
    estimates, accompaniment, vocal = _basis([1.0, -1.0])
    cell = build_cell_quadratic(
        estimates, accompaniment, vocal,
        _config(voice_weight=0.0, residual_weight=0.0),
    )
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] == pytest.approx(4.0, rel=1e-8)


def test_no_vocal_cell_uses_direct_fallback_and_preserves_orchestra():
    accompaniment = np.asarray([1.0, 0.5], dtype=np.complex128)
    vocal = np.zeros_like(accompaniment)
    estimates = np.stack((accompaniment, 0.5 * accompaniment))
    cell = build_cell_quadratic(estimates, accompaniment, vocal, _config())
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "no_vocal_direct_fallback"
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] > 0.0


def test_vocal_only_cell_uses_direct_fallback_and_prefers_silence():
    accompaniment = np.zeros(2, dtype=np.complex128)
    vocal = np.asarray([1.0, 0.25], dtype=np.complex128)
    estimates = np.stack((np.zeros_like(vocal), vocal))
    cell = build_cell_quadratic(estimates, accompaniment, vocal, _config())
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "vocal_only_direct_fallback"
    assert costs[0] == pytest.approx(0.0, abs=1e-10)
    assert costs[1] > 0.0


def test_silent_cell_penalizes_routed_noise():
    accompaniment = np.zeros(4, dtype=np.complex128)
    vocal = np.zeros_like(accompaniment)
    estimates = np.stack((
        np.zeros_like(accompaniment),
        np.full_like(accompaniment, 0.1),
    ))
    cell = build_cell_quadratic(
        estimates, accompaniment, vocal, _config(reference_floor=1e-4)
    )
    costs = np.diag(cell.Q) - 2.0 * cell.c + cell.constant
    assert cell.mode == "silent_direct_fallback"
    assert costs[0] == pytest.approx(0.0, abs=1e-12)
    assert costs[1] > 0.0


def test_measure_weights_are_time_by_frequency_extent():
    result = cell_measure_weights([2.0, 0.5], [10.0, 30.0])
    assert result.tolist() == [[20.0, 60.0], [5.0, 15.0]]


def test_spectral_adapter_uses_physical_cell_measures():
    rng = np.random.default_rng(88)
    accompaniment = (
        rng.normal(size=(2, 5, 4)) + 1j * rng.normal(size=(2, 5, 4))
    ).astype(np.complex64)
    vocal = (
        rng.normal(size=(2, 5, 4)) + 1j * rng.normal(size=(2, 5, 4))
    ).astype(np.complex64)
    candidates = np.stack((accompaniment, accompaniment + 0.2 * vocal))
    grid = build_spectral_quadratic_grid(
        candidates,
        accompaniment,
        vocal,
        time_ranges=((0, 1), (1, 4)),
        frequency_ranges=((0, 2), (2, 5)),
        sample_rate_hz=8,
        n_fft=8,
        hop_length=2,
        config=_config(),
    )
    assert grid.cell_weights.tolist() == [[0.5, 0.75], [1.5, 2.25]]


def test_track_relative_floors_preserve_quadratics_under_gain_scaling():
    estimates, accompaniment, vocal = _basis(
        [0.8, 1.2], beta=[0.15, -0.1]
    )
    config = _config(
        accompaniment_floor=0.0,
        vocal_floor=0.0,
        reference_floor=0.0,
        min_source_power_relative=1e-6,
        reference_floor_relative=1e-8,
    )
    power_a = float(np.mean(np.abs(accompaniment) ** 2))
    power_v = float(np.mean(np.abs(vocal) ** 2))
    base = build_cell_quadratic(
        estimates,
        accompaniment,
        vocal,
        config,
        global_accompaniment_power=power_a,
        global_vocal_power=power_v,
    )
    gain = 1e-4
    scaled = build_cell_quadratic(
        gain * estimates,
        gain * accompaniment,
        gain * vocal,
        config,
        global_accompaniment_power=gain**2 * power_a,
        global_vocal_power=gain**2 * power_v,
    )
    assert scaled.mode == base.mode
    assert np.allclose(scaled.Q, base.Q, rtol=2e-8, atol=2e-8)
    assert np.allclose(scaled.c, base.c, rtol=2e-8, atol=2e-8)
    assert scaled.constant == pytest.approx(base.constant)


def test_grid_refuses_nonpositive_weights_and_empty_modes():
    cell = _unary_cell([0.0, 1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        stack_cell_grid([[cell]], cell_weights=np.asarray([[0.0]]))
    bad = CellQuadratic(
        Q=cell.Q, c=cell.c, constant=cell.constant, mode="",
        accompaniment_energy=1.0, vocal_energy=0.0,
        condition_number=None, psd_relative_tolerance=1e-10,
    )
    with pytest.raises(ValueError, match="non-empty mode"):
        bad.validate()


def test_project_simplex_is_nonnegative_and_normalized():
    result = project_simplex(
        np.asarray([[-1.0, 2.0, 0.0], [0.2, 0.3, 0.7]])
    )
    assert np.all(result >= 0)
    assert np.allclose(result.sum(axis=-1), 1.0)


def test_weighted_whole_track_choice_changes_only_with_declared_weights():
    grid_equal = _unary_grid(
        np.asarray([[[0.0, 0.1]], [[1.0, 0.0]]]),
        cell_weights=np.asarray([[1.0], [1.0]]),
    )
    assert best_whole_track(grid_equal)[0] == 1
    grid_weighted = _unary_grid(
        np.asarray([[[0.0, 0.1]], [[1.0, 0.0]]]),
        cell_weights=np.asarray([[20.0], [1.0]]),
    )
    assert best_whole_track(grid_weighted)[0] == 0


def test_scaling_cell_weights_is_invariant_for_o1_o2_o3():
    unary = np.asarray([
        [[0.0, 0.9], [0.8, 0.1]],
        [[0.6, 0.0], [0.2, 0.7]],
    ])
    base_weights = np.asarray([[3.0, 1.0], [2.0, 4.0]])
    config = _config(
        temporal_switch_penalty=0.2,
        frequency_switch_penalty=0.3,
        temporal_weight_smoothness=0.17,
        frequency_weight_smoothness=0.11,
        projected_gradient_tolerance=2e-8,
    )
    grid_a = _unary_grid(unary, cell_weights=base_weights)
    grid_b = _unary_grid(unary, cell_weights=37.0 * base_weights)
    o1_a, costs_a = best_whole_track(grid_a)
    o1_b, costs_b = best_whole_track(grid_b)
    assert o1_a == o1_b
    assert np.allclose(costs_a, costs_b, atol=1e-12)
    o2_a = solve_discrete_global(grid_a, config)
    o2_b = solve_discrete_global(grid_b, config)
    assert np.array_equal(o2_a.labels, o2_b.labels)
    assert o2_a.objective == pytest.approx(o2_b.objective, abs=1e-10)
    o3_a = solve_convex_certified(
        grid_a, config, o1_index=o1_a, o2_labels=o2_a.labels
    )
    o3_b = solve_convex_certified(
        grid_b, config, o1_index=o1_b, o2_labels=o2_b.labels
    )
    assert np.allclose(o3_a.weights, o3_b.weights, atol=1e-7)
    assert o3_a.objective == pytest.approx(o3_b.objective, abs=1e-9)


def test_certified_o3_finds_exact_interpolation():
    estimates, accompaniment, vocal = _basis([0.8, 1.2])
    config = _config(voice_weight=0.0, residual_weight=0.0)
    cell = build_cell_quadratic(estimates, accompaniment, vocal, config)
    grid = stack_cell_grid([[cell]])
    o1, _ = best_whole_track(grid)
    result = solve_convex_certified(
        grid, config, o1_index=o1,
        starts=("uniform", "O1", "independent"),
    )
    assert result.converged
    assert result.projected_gradient_norm <= config.projected_gradient_tolerance
    assert result.weights[0, 0, 0] == pytest.approx(0.5, abs=1e-5)
    assert result.weights[0, 0, 1] == pytest.approx(0.5, abs=1e-5)
    assert result.objective < unary_costs(grid).min() - 1e-5


def test_o3_rejects_when_only_a_subset_of_requested_starts_certifies():
    matrix = np.diag([1.0, 3.0, 5.0])
    optimum = np.full(3, 1.0 / 3.0)
    cell = CellQuadratic(
        Q=matrix,
        c=matrix @ optimum,
        constant=0.0,
        mode="synthetic",
        accompaniment_energy=1.0,
        vocal_energy=1.0,
        condition_number=1.0,
        psd_relative_tolerance=1e-10,
    )
    grid = stack_cell_grid([[cell]])
    config = _config(
        projected_gradient_tolerance=1e-12,
        max_projected_gradient_iterations=1,
    )
    with pytest.raises(
        CertifiedRoutingError,
        match="every requested start",
    ):
        solve_convex_certified(
            grid,
            config,
            o1_index=0,
            starts=("uniform", "O1"),
        )


def test_global_o2_routes_complementary_time_cells():
    estimates_a, accompaniment_a, vocal_a = _basis([1.0, 0.5])
    estimates_b, accompaniment_b, vocal_b = _basis([0.5, 1.0])
    cell_a = build_cell_quadratic(
        estimates_a, accompaniment_a, vocal_a, _config()
    )
    cell_b = build_cell_quadratic(
        estimates_b, accompaniment_b, vocal_b, _config()
    )
    grid = stack_cell_grid(
        [[cell_a], [cell_b]], cell_weights=np.asarray([[2.0], [1.0]])
    )
    result = solve_discrete_global(grid, _config())
    assert result.labels.tolist() == [[0], [1]]
    assert result.temporal_switches == 1


def test_random_complex_source_quadratic_matches_direct_definition():
    rng = np.random.default_rng(83021)
    config = _config(alpha_weight=1.7, voice_weight=0.8, residual_weight=0.35)
    for _ in range(20):
        samples, candidates = 32, 4
        accompaniment = rng.normal(size=samples) + 1j * rng.normal(size=samples)
        vocal = rng.normal(size=samples) + 1j * rng.normal(size=samples)
        vocal -= accompaniment * (
            np.vdot(accompaniment, vocal) / np.vdot(accompaniment, accompaniment)
        )
        alpha = rng.normal(size=candidates) + 1j * rng.normal(size=candidates)
        beta = rng.normal(size=candidates) + 1j * rng.normal(size=candidates)
        residual = (
            rng.normal(size=(candidates, samples))
            + 1j * rng.normal(size=(candidates, samples))
        )
        for index in range(candidates):
            residual[index] -= accompaniment * (
                np.vdot(accompaniment, residual[index])
                / np.vdot(accompaniment, accompaniment)
            )
            residual[index] -= vocal * (
                np.vdot(vocal, residual[index]) / np.vdot(vocal, vocal)
            )
        estimates = (
            alpha[:, None] * accompaniment
            + beta[:, None] * vocal
            + 0.05 * residual
        )
        cell = build_cell_quadratic(estimates, accompaniment, vocal, config)
        basis = np.column_stack((accompaniment, vocal))
        gram = basis.conj().T @ basis
        ridge = config.ridge_relative * max(
            float(np.trace(gram).real) / 2.0, np.finfo(float).tiny
        )
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(2), basis.conj().T @ estimates.T
        )
        fitted_alpha, fitted_beta = coefficients
        fitted_residual = (
            estimates - fitted_alpha[:, None] * accompaniment
            - fitted_beta[:, None] * vocal
        )
        accompaniment_energy = float(np.vdot(accompaniment, accompaniment).real)
        vocal_energy = float(np.vdot(vocal, vocal).real)
        scale = max(accompaniment_energy, config.reference_floor)
        for _ in range(8):
            weights = _simplex(rng, candidates)
            quadratic = float(
                weights @ cell.Q @ weights - 2.0 * cell.c @ weights
                + cell.constant
            )
            alpha_weighted = weights @ fitted_alpha
            beta_weighted = weights @ fitted_beta
            residual_weighted = weights @ fitted_residual
            direct = (
                config.alpha_weight * abs(alpha_weighted - 1.0) ** 2
                + config.voice_weight * abs(beta_weighted) ** 2
                * vocal_energy / scale
                + config.residual_weight
                * float(np.vdot(residual_weighted, residual_weighted).real)
                / scale
            )
            assert quadratic == pytest.approx(direct, rel=3e-9, abs=3e-9)


def test_random_direct_fallback_matches_exact_target_error():
    rng = np.random.default_rng(1941)
    config = _config(direct_weight=1.3)
    for mode in ("no_vocal", "vocal_only", "silent"):
        for _ in range(12):
            samples, candidates = 24, 3
            accompaniment = rng.normal(size=samples) + 1j * rng.normal(size=samples)
            vocal = rng.normal(size=samples) + 1j * rng.normal(size=samples)
            if mode == "no_vocal":
                vocal[:] = 0
            elif mode == "vocal_only":
                accompaniment[:] = 0
            else:
                accompaniment[:] = 0; vocal[:] = 0
            estimates = 0.1 * (
                rng.normal(size=(candidates, samples))
                + 1j * rng.normal(size=(candidates, samples))
            )
            cell = build_cell_quadratic(estimates, accompaniment, vocal, config)
            scale = max(
                float(np.vdot(accompaniment, accompaniment).real)
                + float(np.vdot(vocal, vocal).real),
                config.reference_floor,
            )
            for _ in range(5):
                weights = _simplex(rng, candidates)
                quadratic = float(
                    weights @ cell.Q @ weights - 2.0 * cell.c @ weights
                    + cell.constant
                )
                error = weights @ estimates - accompaniment
                direct = (
                    config.direct_weight * float(np.vdot(error, error).real)
                    / scale
                )
                assert quadratic == pytest.approx(direct, rel=3e-9, abs=3e-9)


def _brute_o2(grid, config):
    time, bands, candidates = grid.validate()
    temporal_edge = 0.5 * (grid.cell_weights[1:] + grid.cell_weights[:-1])
    frequency_edge = 0.5 * (
        grid.cell_weights[:, 1:] + grid.cell_weights[:, :-1]
    )
    best = None
    for flat in itertools.product(range(candidates), repeat=time * bands):
        labels = np.asarray(flat, dtype=np.int64).reshape(time, bands)
        route = one_hot_weights(labels, candidates)
        data = route_objective(route, grid)[1]
        temporal = float(np.sum(temporal_edge * (labels[1:] != labels[:-1])))
        frequency = float(np.sum(
            frequency_edge * (labels[:, 1:] != labels[:, :-1])
        ))
        value = (
            data + config.temporal_switch_penalty
            * temporal / grid.total_cell_weight
            + config.frequency_switch_penalty
            * frequency / grid.total_cell_weight
        )
        if best is None or value < best[0] - 1e-12:
            best = (float(value), labels.copy())
    return best


def test_o2_milp_equals_exhaustive_weighted_global_optimum():
    rng = np.random.default_rng(227)
    for _ in range(8):
        grid = _unary_grid(
            rng.uniform(0.0, 3.0, size=(2, 2, 3)),
            cell_weights=rng.uniform(0.2, 4.0, size=(2, 2)),
        )
        config = _config(
            temporal_switch_penalty=float(rng.uniform(0.0, 0.8)),
            frequency_switch_penalty=float(rng.uniform(0.0, 0.8)),
        )
        result = solve_discrete_global(grid, config)
        brute_value, _ = _brute_o2(grid, config)
        assert result.objective == pytest.approx(brute_value, abs=2e-9)


def test_o3_matches_independent_slsqp_on_weighted_convex_grid():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(982)
    time, bands, candidates = 2, 2, 3
    c = rng.normal(scale=0.15, size=(time, bands, candidates))
    cells = []
    for time_index in range(time):
        row = []
        for band_index in range(bands):
            factor = rng.normal(size=(candidates, candidates))
            matrix = factor.T @ factor + 0.2 * np.eye(candidates)
            row.append(CellQuadratic(
                Q=matrix, c=c[time_index, band_index], constant=0.0,
                mode="synthetic", accompaniment_energy=1.0,
                vocal_energy=1.0, condition_number=1.0,
                psd_relative_tolerance=1e-10,
            ))
        cells.append(row)
    grid = stack_cell_grid(
        cells, cell_weights=np.asarray([[0.5, 4.0], [2.0, 1.0]])
    )
    config = _config(
        temporal_weight_smoothness=0.17,
        frequency_weight_smoothness=0.11,
        projected_gradient_tolerance=2e-8,
        max_projected_gradient_iterations=40_000,
    )
    result = solve_convex_certified(
        grid, config, starts=("uniform", "O1", "independent")
    )
    assert result.converged

    def objective(flat):
        return route_objective(
            flat.reshape(time, bands, candidates), grid,
            temporal_smoothness=config.temporal_weight_smoothness,
            frequency_smoothness=config.frequency_weight_smoothness,
        )[0]

    constraints = [
        {
            "type": "eq",
            "fun": lambda flat, cell=cell: (
                flat.reshape(time * bands, candidates)[cell].sum() - 1.0
            ),
        }
        for cell in range(time * bands)
    ]
    independent = scipy_optimize.minimize(
        objective,
        np.full(time * bands * candidates, 1.0 / candidates),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (time * bands * candidates),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    assert independent.success, independent.message
    assert result.objective == pytest.approx(independent.fun, abs=3e-7)
