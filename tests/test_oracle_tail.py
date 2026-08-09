import numpy as np
import pytest

from audio_extract.oracle_convex import ConvexQuadratic
from audio_extract.oracle_tail import (
    ActiveSetConfig,
    TailOracleError,
    corrected_vertex_unary,
    solve_cell_active_set,
    solve_independent_convex,
    solve_independent_discrete,
)


def quadratic(gram, linear, constant=None, available=None):
    g = np.asarray(gram, dtype=np.float64)
    c = np.asarray(linear, dtype=np.float64)
    if g.ndim != 4 or c.shape != g.shape[:3]:
        raise ValueError("fixtures must be Q,B,K,K and Q,B,K")
    q, b, _, _ = g.shape
    if constant is None:
        constant = np.zeros((q, b), dtype=np.float64)
    if available is None:
        available = np.ones((q, b), dtype=bool)
    eig = max(float(np.linalg.eigvalsh(g[i, j]).max())
              for i in range(q) for j in range(b))
    return ConvexQuadratic(
        gram=g,
        linear=c,
        constant=np.asarray(constant, dtype=np.float64),
        available=np.asarray(available, dtype=bool),
        denominator=int(np.asarray(available, dtype=bool).sum()),
        max_local_eigenvalue=eig,
    )


def test_corrected_vertex_unary_penalizes_overgain_and_phase():
    # |alpha-1|^2 at alpha=[1,-1,2] gives [0,4,1].
    gram = np.zeros((1, 1, 3, 3), dtype=np.float64)
    alpha = np.array([1.0, -1.0, 2.0])
    gram[0, 0] = np.outer(alpha, alpha)
    linear = alpha.reshape(1, 1, 3)
    constant = np.ones((1, 1))
    unary = corrected_vertex_unary(quadratic(gram, linear, constant))
    assert unary[0, 0].tolist() == pytest.approx([0.0, 4.0, 1.0])


def test_o0d_selects_each_cell_independently_and_fills_masked_with_o1():
    gram = np.zeros((3, 1, 2, 2))
    linear = np.array([
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        [[10.0, -10.0]],
    ])
    available = np.array([[True], [True], [False]])
    result = solve_independent_discrete(
        quadratic(gram, linear, available=available), fallback_index=1
    )
    assert result.labels[:, 0].tolist() == [0, 1, 1]
    assert result.available_cells == 2
    assert result.unavailable_cells == 1
    assert np.allclose(result.weights.sum(axis=-1), 1.0)


def test_o0c_finds_interior_half_half_solution():
    # (2*w1 - 1)^2; optimum at w=[0.5,0.5].
    gram = np.array([[[[0.0, 0.0], [0.0, 4.0]]]])
    linear = np.array([[[0.0, 2.0]]])
    weight, facts = solve_cell_active_set(gram[0, 0], linear[0, 0], 1.0)
    assert weight.tolist() == pytest.approx([0.5, 0.5], abs=1e-8)
    assert facts["support_size"] == 2
    assert facts["interpolated"]
    assert facts["objective"] == pytest.approx(0.0, abs=1e-10)


def test_o0c_is_invariant_to_tiny_positive_objective_scaling():
    scale = 1e-13
    gram = scale * np.array([[0.0, 0.0], [0.0, 4.0]])
    linear = scale * np.array([0.0, 2.0])
    weight, facts = solve_cell_active_set(gram, linear, scale)
    assert weight.tolist() == pytest.approx([0.5, 0.5], abs=1e-8)
    assert facts["interpolated"]
    assert facts["objective"] == pytest.approx(0.0, abs=1e-24)
    assert facts["normalization_scale"] == pytest.approx(4e-13)


def test_o0c_equilibrates_high_scale_kkt_and_finds_uniform_optimum():
    gram = 1e8 * np.eye(3, dtype=np.float64)
    linear = np.zeros(3, dtype=np.float64)
    weight, facts = solve_cell_active_set(gram, linear, 0.0)
    assert weight.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)
    assert facts["support_size"] == 3
    assert facts["interpolated"]
    assert facts["normalized_stationarity_residual"] <= 1e-9
    assert facts["objective"] == pytest.approx(1e8 / 3, rel=1e-10)


def test_constant_shift_does_not_change_the_optimizer_or_tie_scale():
    gram = np.array([[0.0, 0.0], [0.0, 4.0]])
    linear = np.array([0.0, 2.0])
    first, _ = solve_cell_active_set(gram, linear, 0.0)
    shifted, facts = solve_cell_active_set(gram, linear, 1e200)
    assert shifted.tolist() == pytest.approx(first.tolist(), abs=1e-10)
    assert facts["normalization_scale"] == pytest.approx(4.0)


def test_o0c_boundary_vertex_is_optimal():
    # Minimize w1^2 + 2*w1 on the simplex: w1=0, candidate 0 is selected.
    gram = np.array([[0.0, 0.0], [0.0, 1.0]])
    linear = np.array([0.0, -1.0])
    weight, facts = solve_cell_active_set(gram, linear, 0.0)
    assert weight.tolist() == pytest.approx([1.0, 0.0])
    assert facts["support_size"] == 1
    assert not facts["interpolated"]


def test_singular_psd_multiple_optima_returns_deterministic_small_support():
    gram = np.zeros((3, 3), dtype=np.float64)
    linear = np.zeros(3, dtype=np.float64)
    weight, facts = solve_cell_active_set(gram, linear, 2.0)
    assert weight.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert facts["support"] == [0]
    assert facts["objective"] == pytest.approx(2.0)


def test_inactive_gradient_rejects_spurious_support():
    # Support {0} satisfies its equality constraint but candidate 1 has a negative
    # reduced gradient and must enter; global optimum is the candidate-1 vertex.
    gram = np.zeros((2, 2), dtype=np.float64)
    linear = np.array([0.0, 2.0], dtype=np.float64)
    weight, facts = solve_cell_active_set(gram, linear, 0.0)
    assert weight.tolist() == pytest.approx([0.0, 1.0])
    assert facts["support"] == [1]


def test_o0c_aggregate_excludes_unavailable_cell_from_data_objective():
    gram = np.zeros((2, 1, 2, 2), dtype=np.float64)
    linear = np.array([[[0.0, 2.0]], [[1000.0, -1000.0]]])
    constant = np.array([[1.0], [9999.0]])
    available = np.array([[True], [False]])
    result = solve_independent_convex(
        quadratic(gram, linear, constant, available), fallback_index=0
    )
    assert result.weights[1, 0].tolist() == pytest.approx([1.0, 0.0])
    assert result.available_cells == 1
    assert result.unavailable_cells == 1
    assert result.data_objective == pytest.approx(-3.0)
    assert result.interpolated_fraction == pytest.approx(0.0)
    assert result.cell_normalization_scale_min == pytest.approx(2.0)
    assert result.cell_normalization_scale_max == pytest.approx(2.0)


def test_non_psd_cell_is_refused():
    gram = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(TailOracleError, match="not PSD"):
        solve_cell_active_set(
            gram, np.zeros(2), 0.0,
            ActiveSetConfig(kkt_tolerance=1e-10),
        )
