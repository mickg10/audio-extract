import itertools

import numpy as np
import pytest

from audio_extract.oracle_routing_certified import (
    CellQuadratic,
    CertifiedRoutingConfig,
    QuadraticGrid,
    build_cell_quadratic,
    one_hot_weights,
    route_objective,
    solve_convex_certified,
    solve_discrete_global,
    stack_cell_grid,
)


def _simplex(rng, candidates):
    value = rng.random(candidates)
    return value / value.sum()


def test_random_complex_source_quadratic_matches_direct_definition():
    rng = np.random.default_rng(83021)
    config = CertifiedRoutingConfig(
        alpha_weight=1.7,
        voice_weight=0.8,
        residual_weight=0.35,
        temporal_weight_smoothness=0.0,
        frequency_weight_smoothness=0.0,
    )
    for _ in range(30):
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
        cell = build_cell_quadratic(
            estimates, accompaniment, vocal, config
        )
        assert cell.mode == "source_coordinates"

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
            estimates
            - fitted_alpha[:, None] * accompaniment
            - fitted_beta[:, None] * vocal
        )
        accompaniment_energy = float(np.vdot(accompaniment, accompaniment).real)
        vocal_energy = float(np.vdot(vocal, vocal).real)
        scale = max(accompaniment_energy, config.reference_floor)

        for _ in range(10):
            weights = _simplex(rng, candidates)
            quadratic = float(
                weights @ cell.Q @ weights
                - 2.0 * cell.c @ weights
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
            assert quadratic == pytest.approx(
                direct, rel=2e-9, abs=2e-9
            )


def test_random_direct_fallback_quadratic_matches_exact_target_error():
    rng = np.random.default_rng(1941)
    config = CertifiedRoutingConfig(direct_weight=1.3)
    for mode in ("no_vocal", "vocal_only", "silent"):
        for _ in range(20):
            samples, candidates = 24, 3
            accompaniment = (
                rng.normal(size=samples) + 1j * rng.normal(size=samples)
            )
            vocal = rng.normal(size=samples) + 1j * rng.normal(size=samples)
            if mode == "no_vocal":
                vocal[:] = 0
            elif mode == "vocal_only":
                accompaniment[:] = 0
            else:
                accompaniment[:] = 0
                vocal[:] = 0
            estimates = 0.1 * (
                rng.normal(size=(candidates, samples))
                + 1j * rng.normal(size=(candidates, samples))
            )
            cell = build_cell_quadratic(
                estimates, accompaniment, vocal, config
            )
            assert cell.mode.endswith("direct_fallback")
            scale = max(
                float(np.vdot(accompaniment, accompaniment).real)
                + float(np.vdot(vocal, vocal).real),
                config.reference_floor,
            )
            for _ in range(8):
                weights = _simplex(rng, candidates)
                quadratic = float(
                    weights @ cell.Q @ weights
                    - 2.0 * cell.c @ weights
                    + cell.constant
                )
                error = weights @ estimates - accompaniment
                direct = (
                    config.direct_weight
                    * float(np.vdot(error, error).real) / scale
                )
                assert quadratic == pytest.approx(
                    direct, rel=2e-9, abs=2e-9
                )


def _unary_grid(unary):
    unary = np.asarray(unary, dtype=np.float64)
    cells = []
    for time in range(unary.shape[0]):
        row = []
        for band in range(unary.shape[1]):
            values = unary[time, band]
            row.append(CellQuadratic(
                Q=np.diag(values),
                c=np.zeros(len(values)),
                constant=0.0,
                mode="direct_test",
                accompaniment_energy=1.0,
                vocal_energy=0.0,
                condition_number=None,
            ))
        cells.append(row)
    return stack_cell_grid(cells)


def _brute_o2(grid, temporal_penalty, frequency_penalty):
    time, bands, candidates = grid.validate()
    best = None
    for flat in itertools.product(
        range(candidates), repeat=time * bands
    ):
        labels = np.asarray(flat, dtype=np.int64).reshape(time, bands)
        weights = one_hot_weights(labels, candidates)
        data = route_objective(weights, grid)[1]
        temporal = np.count_nonzero(labels[1:] != labels[:-1])
        frequency = np.count_nonzero(labels[:, 1:] != labels[:, :-1])
        value = (
            data
            + temporal_penalty * temporal / (time * bands)
            + frequency_penalty * frequency / (time * bands)
        )
        if best is None or value < best[0] - 1e-12:
            best = (float(value), labels.copy())
    return best


def test_o2_milp_equals_exhaustive_global_optimum_on_seeded_small_grids():
    rng = np.random.default_rng(227)
    for _ in range(12):
        grid = _unary_grid(rng.uniform(0.0, 3.0, size=(2, 2, 3)))
        temporal_penalty = float(rng.uniform(0.0, 0.8))
        frequency_penalty = float(rng.uniform(0.0, 0.8))
        config = CertifiedRoutingConfig(
            temporal_switch_penalty=temporal_penalty,
            frequency_switch_penalty=frequency_penalty,
            temporal_weight_smoothness=0.0,
            frequency_weight_smoothness=0.0,
        )
        result = solve_discrete_global(grid, config)
        brute_value, _ = _brute_o2(
            grid, temporal_penalty, frequency_penalty
        )
        assert result.objective == pytest.approx(brute_value, abs=1e-9)

        returned = one_hot_weights(result.labels, 3)
        returned_data = route_objective(returned, grid)[1]
        returned_value = (
            returned_data
            + temporal_penalty * result.temporal_switches / 4.0
            + frequency_penalty * result.frequency_switches / 4.0
        )
        assert returned_value == pytest.approx(brute_value, abs=1e-9)


def test_o3_matches_independent_slsqp_on_seeded_convex_grid():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(982)
    time, bands, candidates = 2, 2, 3
    q = np.empty((time, bands, candidates, candidates))
    c = rng.normal(scale=0.15, size=(time, bands, candidates))
    for time_index in range(time):
        for band_index in range(bands):
            factor = rng.normal(size=(candidates, candidates))
            q[time_index, band_index] = (
                factor.T @ factor + 0.2 * np.eye(candidates)
            )
    grid = QuadraticGrid(
        Q=q,
        c=c,
        constant=np.zeros((time, bands)),
        modes=np.full((time, bands), "synthetic", dtype=object),
    )
    config = CertifiedRoutingConfig(
        temporal_weight_smoothness=0.17,
        frequency_weight_smoothness=0.11,
        projected_gradient_tolerance=2e-8,
        max_projected_gradient_iterations=30_000,
    )
    result = solve_convex_certified(
        grid, config, starts=("uniform", "O1", "independent")
    )
    assert result.converged

    def objective(flat):
        return route_objective(
            flat.reshape(time, bands, candidates),
            grid,
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
    assert result.objective == pytest.approx(independent.fun, abs=2e-7)
