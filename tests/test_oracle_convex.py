import numpy as np
import pytest

from audio_extract.oracle_convex import (
    ConvexOracleConfig,
    ConvexOracleError,
    build_quadratic,
    objective,
    project_simplex,
    solve_independent_convex_oracle,
    solve_independent_discrete_oracle,
    solve_true_convex_oracle,
)
from audio_extract.oracle_routing import CellStatistics


def stats(alpha, beta=None, residual_gram=None, available=None):
    alpha = np.asarray(alpha, dtype=np.complex128)
    if alpha.ndim != 3:
        raise ValueError("alpha fixture must be Q,B,K")
    q, b, k = alpha.shape
    beta = np.zeros_like(alpha) if beta is None else np.asarray(beta, dtype=np.complex128)
    residual_gram = (
        np.zeros((q, b, k, k), dtype=np.complex128)
        if residual_gram is None else np.asarray(residual_gram, dtype=np.complex128)
    )
    available = (
        np.ones((q, b), dtype=bool)
        if available is None else np.asarray(available, dtype=bool)
    )
    return CellStatistics(
        alpha=alpha,
        beta=beta,
        residual_gram=residual_gram,
        accompaniment_energy=np.ones((q, b), dtype=np.float64),
        vocal_energy=np.ones((q, b), dtype=np.float64),
        condition_number=np.ones((q, b), dtype=np.float64),
        available=available,
        unary_risk=np.zeros((q, b, k), dtype=np.float64),
        time_frame_ranges=tuple((i, i + 1) for i in range(q)),
        frequency_bin_ranges=tuple((i, i + 1) for i in range(b)),
    )


def test_simplex_projection_is_exact_per_cell():
    value = np.array([
        [[-1.0, 2.0, 0.0], [0.2, 0.3, 0.7]],
        [[10.0, -4.0, 2.0], [1.0, 1.0, 1.0]],
    ])
    projected = project_simplex(value)
    assert np.all(projected >= 0)
    assert np.allclose(projected.sum(axis=-1), 1.0, atol=1e-12)


def test_true_convex_oracle_finds_interior_interpolation():
    # Candidate alpha values 0 and 2; alpha=1 is attained only by w=[0.5,0.5].
    source = stats(np.array([[[0.0, 2.0]]]))
    config = ConvexOracleConfig(
        transfer_weight=1.0,
        retained_voice_weight=0.0,
        orthogonal_artifact_weight=0.0,
        temporal_l2_weight=0.0,
        frequency_l2_weight=0.0,
        max_iterations=5000,
        gradient_mapping_tolerance=1e-8,
        objective_tolerance=1e-12,
        solution_agreement_tolerance=1e-8,
    )
    result = solve_true_convex_oracle(
        source, o1_index=0, o2_labels=np.array([[0]]), config=config
    )
    assert result.converged
    assert result.weights[0, 0, 0] == pytest.approx(0.5, abs=1e-5)
    assert result.weights[0, 0, 1] == pytest.approx(0.5, abs=1e-5)
    assert result.objective < result.best_vertex_objective - 0.9
    assert result.projected_gradient_mapping_inf <= config.gradient_mapping_tolerance


def test_independent_discrete_and_convex_envelopes_use_corrected_quadratic():
    source = stats(np.array([
        [[0.0, 2.0]],
        [[1.0, 3.0]],
    ]))
    config = ConvexOracleConfig(
        transfer_weight=1.0, retained_voice_weight=0.0,
        orthogonal_artifact_weight=0.0,
        temporal_l2_weight=9.0, frequency_l2_weight=7.0,
        max_iterations=5000, gradient_mapping_tolerance=1e-8,
        objective_tolerance=1e-12, solution_agreement_tolerance=1e-8,
    )
    discrete = solve_independent_discrete_oracle(
        source, fallback_index=0, config=config
    )
    assert np.array_equal(discrete.labels, np.array([[0], [0]]))
    assert discrete.objective == pytest.approx(0.5)
    convex, effective = solve_independent_convex_oracle(
        source, o1_index=0, o2_labels=discrete.labels, config=config
    )
    assert effective.temporal_l2_weight == 0.0
    assert effective.frequency_l2_weight == 0.0
    assert convex.weights[0, 0, 0] == pytest.approx(0.5, abs=1e-5)
    assert convex.weights[1, 0, 0] == pytest.approx(1.0, abs=1e-5)
    assert convex.objective < discrete.objective


def test_independent_envelopes_use_fallback_in_unavailable_cells():
    source = stats(
        np.array([[[1.0, 2.0]], [[0.0, 2.0]]]),
        available=np.array([[True], [False]]),
    )
    discrete = solve_independent_discrete_oracle(source, fallback_index=1)
    assert discrete.labels[1, 0] == 1
    convex, _ = solve_independent_convex_oracle(
        source, o1_index=1, o2_labels=discrete.labels
    )
    assert np.array_equal(convex.weights[1, 0], np.array([0.0, 1.0]))


def test_quadratic_penalizes_phase_and_overgain_not_only_holes():
    source = stats(np.array([[[1.0, -1.0, 2.0]]], dtype=np.complex128))
    config = ConvexOracleConfig(
        transfer_weight=1.0,
        retained_voice_weight=0.0,
        orthogonal_artifact_weight=0.0,
        temporal_l2_weight=0.0,
        frequency_l2_weight=0.0,
    )
    quadratic = build_quadratic(source, config)
    candidates = np.eye(3, dtype=np.float64).reshape(3, 1, 1, 3)
    values = [objective(candidate, quadratic, config)[0] for candidate in candidates]
    assert values[0] == pytest.approx(0.0, abs=1e-10)
    assert values[1] == pytest.approx(4.0, abs=1e-10)
    assert values[2] == pytest.approx(1.0, abs=1e-10)


def test_voice_term_uses_fixed_true_accompaniment_denominator():
    alpha = np.array([[[1.0, 100.0]]], dtype=np.complex128)
    beta = np.array([[[1.0, 1.0]]], dtype=np.complex128)
    source = stats(alpha, beta)
    config = ConvexOracleConfig(
        transfer_weight=0.0,
        retained_voice_weight=1.0,
        orthogonal_artifact_weight=0.0,
        temporal_l2_weight=0.0,
        frequency_l2_weight=0.0,
    )
    quadratic = build_quadratic(source, config)
    first = np.array([[[1.0, 0.0]]])
    second = np.array([[[0.0, 1.0]]])
    assert objective(first, quadratic, config)[0] == pytest.approx(
        objective(second, quadratic, config)[0], rel=1e-8
    )


def test_smoothness_couples_neighboring_cells():
    # Each cell has a different exact vertex; strong smoothing pulls the routes together.
    source = stats(np.array([
        [[1.0, 2.0]],
        [[2.0, 1.0]],
    ]))
    weak = ConvexOracleConfig(
        transfer_weight=1.0, retained_voice_weight=0.0,
        orthogonal_artifact_weight=0.0,
        temporal_l2_weight=0.0, frequency_l2_weight=0.0,
        max_iterations=5000, gradient_mapping_tolerance=1e-7,
        solution_agreement_tolerance=1e-6,
    )
    strong = ConvexOracleConfig(
        transfer_weight=1.0, retained_voice_weight=0.0,
        orthogonal_artifact_weight=0.0,
        temporal_l2_weight=10.0, frequency_l2_weight=0.0,
        max_iterations=10000, gradient_mapping_tolerance=1e-6,
        solution_agreement_tolerance=1e-5,
    )
    route = np.array([[0], [1]])
    weak_result = solve_true_convex_oracle(source, o1_index=0, o2_labels=route, config=weak)
    strong_result = solve_true_convex_oracle(source, o1_index=0, o2_labels=route, config=strong)
    weak_difference = np.abs(weak_result.weights[0] - weak_result.weights[1]).sum()
    strong_difference = np.abs(strong_result.weights[0] - strong_result.weights[1]).sum()
    assert strong_difference < weak_difference


def test_iteration_limit_is_rejected_instead_of_returned_as_oracle():
    source = stats(np.array([[[0.0, 2.0]]]))
    config = ConvexOracleConfig(
        transfer_weight=1.0,
        retained_voice_weight=0.0,
        orthogonal_artifact_weight=0.0,
        temporal_l2_weight=0.0,
        frequency_l2_weight=0.0,
        max_iterations=1,
        gradient_mapping_tolerance=1e-15,
        objective_tolerance=0.0,
        solution_agreement_tolerance=1e-12,
    )
    with pytest.raises(ConvexOracleError, match="did not converge"):
        solve_true_convex_oracle(
            source, o1_index=0, o2_labels=np.array([[0]]), config=config
        )


def test_non_psd_residual_gram_is_rejected():
    residual = np.array([[[[1.0, 2.0], [2.0, 1.0]]]], dtype=np.complex128)
    source = stats(np.array([[[1.0, 1.0]]]), residual_gram=residual)
    with pytest.raises(ConvexOracleError, match="not PSD"):
        build_quadratic(source, ConvexOracleConfig(psd_tolerance=1e-12))


def test_non_hermitian_residual_gram_is_rejected():
    residual = np.array([[[[1.0, 1.0j], [1.0j, 1.0]]]], dtype=np.complex128)
    source = stats(np.array([[[1.0, 1.0]]]), residual_gram=residual)
    with pytest.raises(ConvexOracleError, match="not Hermitian"):
        build_quadratic(source, ConvexOracleConfig(psd_tolerance=1e-12))


def test_unavailable_cells_have_only_smoothness_not_fake_zero_quality():
    source = stats(
        np.array([[[1.0, 2.0]], [[0.0, 2.0]]]),
        available=np.array([[True], [False]]),
    )
    config = ConvexOracleConfig(
        temporal_l2_weight=1.0, frequency_l2_weight=0.0,
        max_iterations=10000, gradient_mapping_tolerance=1e-6,
        solution_agreement_tolerance=1e-5,
    )
    result = solve_true_convex_oracle(
        source, o1_index=0, o2_labels=np.array([[0], [1]]), config=config
    )
    # The masked cell is inferred by smoothness; it is not counted as a data win.
    assert np.allclose(result.weights[0], result.weights[1], atol=1e-3)
