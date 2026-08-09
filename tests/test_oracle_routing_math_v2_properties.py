import itertools

import numpy as np
import pytest

from audio_extract.oracle_routing_math_v2 import (
    best_whole_track,
    build_exact_cell_quadratic,
    one_hot,
    route_objective,
    solve_o2_global,
    solve_o3_convex,
    stack_cells,
    unary_costs,
)


def test_source_coordinate_quadratic_matches_declared_formula():
    # A, V, and residual directions are exactly orthogonal.
    accompaniment = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 2.0, 0.0, 0.0], dtype=np.complex128)
    alpha = np.asarray([0.8 + 0.1j, 1.15 - 0.05j])
    beta = np.asarray([0.2 - 0.1j, -0.15 + 0.05j])
    residual = np.asarray([
        [0.0, 0.0, 0.3 + 0.1j, 0.0],
        [0.0, 0.0, 0.0, -0.2 + 0.15j],
    ])
    candidates = (
        alpha[:, None] * accompaniment
        + beta[:, None] * vocal
        + residual
    )
    cell = build_exact_cell_quadratic(
        candidates, accompaniment, vocal,
        alpha_weight=2.0, voice_weight=3.0, artifact_weight=4.0,
        ridge_relative=0.0, reference_floor=1e-12,
    )
    weights = np.asarray([0.35, 0.65])
    quadratic = float(
        weights @ cell.Q @ weights - 2.0 * cell.c @ weights + cell.constant
    )
    alpha_w = weights @ alpha
    beta_w = weights @ beta
    residual_w = weights @ residual
    declared = float(
        2.0 * abs(alpha_w - 1.0) ** 2
        + 3.0 * abs(beta_w) ** 2
        * np.vdot(vocal, vocal).real / np.vdot(accompaniment, accompaniment).real
        + 4.0 * np.vdot(residual_w, residual_w).real
        / np.vdot(accompaniment, accompaniment).real
    )
    assert quadratic == pytest.approx(declared, rel=1e-9, abs=1e-10)


def test_direct_fallback_quadratic_matches_weighted_exact_target_error():
    accompaniment = np.asarray([1.0, -0.25, 0.5], dtype=np.complex128)
    vocal = np.zeros_like(accompaniment)
    candidates = np.stack((
        accompaniment,
        0.4 * accompaniment,
        accompaniment + np.asarray([0.0, 0.1, -0.2]),
    ))
    cell = build_exact_cell_quadratic(
        candidates, accompaniment, vocal,
        direct_weight=2.5, reference_floor=1e-12,
    )
    weights = np.asarray([0.2, 0.3, 0.5])
    quadratic = float(weights @ cell.Q @ weights)
    error = weights @ candidates - accompaniment
    declared = float(
        2.5 * np.vdot(error, error).real
        / np.vdot(accompaniment, accompaniment).real
    )
    assert cell.mode == "no_vocal_direct_fallback"
    assert quadratic == pytest.approx(declared, rel=1e-10, abs=1e-12)


def test_o2_milp_matches_bruteforce_on_small_grid():
    accompaniment = np.asarray([1.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 1.0], dtype=np.complex128)
    # Different cells prefer different candidates.
    alpha_cells = (
        (1.0, 0.7, 1.2),
        (0.7, 1.0, 1.2),
        (1.2, 0.7, 1.0),
        (1.0, 1.2, 0.7),
    )
    cells = []
    for alpha in alpha_cells:
        candidates = np.asarray(alpha)[:, None] * accompaniment[None]
        cells.append(build_exact_cell_quadratic(
            candidates, accompaniment, vocal,
            voice_weight=0.0, artifact_weight=0.0,
        ))
    grid = stack_cells([[cells[0], cells[1]], [cells[2], cells[3]]])
    temporal_penalty = 0.07
    frequency_penalty = 0.11
    result = solve_o2_global(
        grid,
        temporal_switch_penalty=temporal_penalty,
        frequency_switch_penalty=frequency_penalty,
    )
    unary = unary_costs(grid)
    best = float("inf")
    best_labels = []
    for flat in itertools.product(range(3), repeat=4):
        labels = np.asarray(flat, dtype=np.int64).reshape(2, 2)
        data = float(np.take_along_axis(
            unary, labels[..., None], axis=-1
        )[..., 0].mean())
        objective = (
            data
            + temporal_penalty * np.count_nonzero(labels[1:] != labels[:-1]) / 4
            + frequency_penalty * np.count_nonzero(labels[:, 1:] != labels[:, :-1]) / 4
        )
        if objective < best - 1e-12:
            best = objective
            best_labels = [labels]
        elif abs(objective - best) <= 1e-12:
            best_labels.append(labels)
    assert result.objective == pytest.approx(best, abs=1e-10)
    assert any(np.array_equal(result.labels, labels) for labels in best_labels)


def test_complete_o3_objective_satisfies_jensen_inequality():
    accompaniment = np.asarray([1.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 1.0], dtype=np.complex128)
    rows = []
    for offset in (0.0, 0.1):
        row = []
        for alpha in ((0.8 + offset, 1.2, 1.0), (1.1, 0.75 + offset, 1.0)):
            candidates = np.asarray(alpha)[:, None] * accompaniment[None]
            row.append(build_exact_cell_quadratic(
                candidates, accompaniment, vocal,
                voice_weight=0.0, artifact_weight=0.0,
            ))
        rows.append(row)
    grid = stack_cells(rows)
    rng = np.random.default_rng(7)
    w1 = rng.dirichlet(np.ones(3), size=4).reshape(2, 2, 3)
    w2 = rng.dirichlet(np.ones(3), size=4).reshape(2, 2, 3)
    theta = 0.37
    mixed = theta * w1 + (1.0 - theta) * w2
    kwargs = dict(temporal_smoothness=0.13, frequency_smoothness=0.09)
    lhs = route_objective(mixed, grid, **kwargs)[0]
    rhs = (
        theta * route_objective(w1, grid, **kwargs)[0]
        + (1.0 - theta) * route_objective(w2, grid, **kwargs)[0]
    )
    assert lhs <= rhs + 1e-10


def test_certified_o3_is_no_worse_than_its_o1_and_o2_starts():
    accompaniment = np.asarray([1.0, 0.0], dtype=np.complex128)
    vocal = np.asarray([0.0, 1.0], dtype=np.complex128)
    rows = []
    for alpha in ((0.8, 1.2), (1.15, 0.85), (0.9, 1.1)):
        candidates = np.asarray(alpha)[:, None] * accompaniment[None]
        rows.append([build_exact_cell_quadratic(
            candidates, accompaniment, vocal,
            voice_weight=0.0, artifact_weight=0.0,
        )])
    grid = stack_cells(rows)
    o1_index, _ = best_whole_track(grid)
    o2 = solve_o2_global(
        grid, temporal_switch_penalty=0.02, frequency_switch_penalty=0.0
    )
    kwargs = dict(temporal_smoothness=0.02, frequency_smoothness=0.0)
    result = solve_o3_convex(
        grid, o1_index=o1_index, o2_labels=o2.labels,
        tolerance=1e-8, max_iterations=20_000, **kwargs,
    )
    o1 = one_hot(np.full((3, 1), o1_index), 2)
    o2_weights = one_hot(o2.labels, 2)
    assert result.objective <= route_objective(o1, grid, **kwargs)[0] + 1e-8
    assert result.objective <= route_objective(o2_weights, grid, **kwargs)[0] + 1e-8
