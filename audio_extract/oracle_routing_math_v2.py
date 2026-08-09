"""Certified exact-reference routing mathematics for opera stem-separation research.

This module is deliberately independent of repository storage and audio I/O.  It
implements the mathematical core needed by an O1/O2/O3 oracle-envelope
diagnostic:

* every exact time/frequency cell contributes to the objective;
* identifiable A/V cells use a convex source-coordinate quadratic;
* A-only, V-only, silent, and ill-conditioned cells use exact direct-target
  error rather than disappearing as zero-cost cells;
* O2 is a global Potts-labeling MILP;
* O3 is a genuinely convex simplex problem with squared graph smoothness and a
  projected-gradient convergence certificate.

It consumes clean references and is therefore an oracle research tool, not a
deployable separator or judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

import numpy as np

_EPS = np.finfo(np.float64).tiny
_PSD_ATOL = 1e-12
_PSD_RTOL = 1e-10
_VALID_MODES = frozenset({
    "source_coordinates",
    "silent_direct_fallback",
    "no_vocal_direct_fallback",
    "vocal_only_direct_fallback",
    "ill_conditioned_direct_fallback",
})


def _psd_tolerance(eigenvalues: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), _EPS)
    return _PSD_ATOL + _PSD_RTOL * scale


def _validate_psd(matrix: np.ndarray, label: str) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum = float(eigenvalues.min(initial=0.0))
    tolerance = _psd_tolerance(eigenvalues)
    if minimum < -tolerance:
        raise ValueError(
            f"{label} is not positive semidefinite: "
            f"min_eigenvalue={minimum}, tolerance={tolerance}"
        )


class RoutingMathError(RuntimeError):
    """The routing input, optimization, or certificate is invalid."""


@dataclass(frozen=True)
class CellQuadratic:
    """One exact cell cost: w.T Q w - 2 c.T w + constant."""

    Q: np.ndarray
    c: np.ndarray
    constant: float
    mode: str
    accompaniment_energy: float
    vocal_energy: float
    condition_number: float | None

    def __post_init__(self) -> None:
        q = np.asarray(self.Q, dtype=np.float64)
        c = np.asarray(self.c, dtype=np.float64)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("Q must be a square matrix")
        if c.shape != (q.shape[0],):
            raise ValueError("c must match Q")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(c)):
            raise ValueError("quadratic coefficients must be finite")
        if not math.isfinite(float(self.constant)):
            raise ValueError("quadratic constant must be finite")
        symmetry_scale = max(float(np.max(np.abs(q), initial=0.0)), _EPS)
        if np.max(np.abs(q - q.T), initial=0.0) > (
            _PSD_ATOL + _PSD_RTOL * symmetry_scale
        ):
            raise ValueError("Q must be symmetric")
        _validate_psd(q, "Q")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"unknown cell mode: {self.mode!r}")
        for name, value in (
            ("accompaniment_energy", self.accompaniment_energy),
            ("vocal_energy", self.vocal_energy),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.condition_number is not None and (
            not math.isfinite(float(self.condition_number))
            or float(self.condition_number) < 1.0
        ):
            raise ValueError("condition_number must be None or finite and >= 1")


@dataclass(frozen=True)
class QuadraticGrid:
    """A complete time/band grid of exact convex costs.

    ``measure`` is a positive identity-bearing cell measure.  Objectives use
    its normalized form, so globally scaling every measure leaves every O1/O2/O3
    data term unchanged while short final tiles and narrow bands no longer have
    the same influence as large cells.
    """

    Q: np.ndarray          # (T, B, K, K)
    c: np.ndarray          # (T, B, K)
    constant: np.ndarray   # (T, B)
    modes: np.ndarray      # (T, B), object/string
    measure: np.ndarray | None = None  # (T, B), strictly positive

    def validate(self) -> tuple[int, int, int]:
        q = np.asarray(self.Q, dtype=np.float64)
        c = np.asarray(self.c, dtype=np.float64)
        k0 = np.asarray(self.constant, dtype=np.float64)
        if q.ndim != 4 or q.shape[-1] != q.shape[-2]:
            raise ValueError("Q must be (time, band, K, K)")
        if c.shape != q.shape[:-1] or k0.shape != q.shape[:2]:
            raise ValueError("c/constant shapes do not match Q")
        modes = np.asarray(self.modes, dtype=object)
        if modes.shape != q.shape[:2]:
            raise ValueError("mode grid does not match Q")
        if any(str(value) not in _VALID_MODES for value in modes.reshape(-1)):
            raise ValueError("mode grid contains an unknown cell mode")
        measure = (
            np.ones(q.shape[:2], dtype=np.float64)
            if self.measure is None
            else np.asarray(self.measure, dtype=np.float64)
        )
        if measure.shape != q.shape[:2]:
            raise ValueError("cell measure does not match Q")
        if not np.all(np.isfinite(measure)) or np.any(measure <= 0.0):
            raise ValueError("cell measure must be finite and strictly positive")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(c))
                and np.all(np.isfinite(k0))):
            raise ValueError("grid coefficients must be finite")
        symmetry = np.max(np.abs(q - np.swapaxes(q, -1, -2)), initial=0.0)
        symmetry_scale = max(float(np.max(np.abs(q), initial=0.0)), _EPS)
        if symmetry > _PSD_ATOL + _PSD_RTOL * symmetry_scale:
            raise ValueError(f"Q grid is not symmetric: {symmetry}")
        for index, matrix in enumerate(
            q.reshape(-1, q.shape[-1], q.shape[-1])
        ):
            _validate_psd(matrix, f"Q grid cell {index}")
        return q.shape[0], q.shape[1], q.shape[2]

    def normalized_measure(self) -> np.ndarray:
        self.validate()
        measure = (
            np.ones(self.Q.shape[:2], dtype=np.float64)
            if self.measure is None
            else np.asarray(self.measure, dtype=np.float64)
        )
        return measure / float(measure.sum())


@dataclass(frozen=True)
class O2Result:
    labels: np.ndarray
    objective: float
    data_objective: float
    temporal_switches: int
    frequency_switches: int
    solver_status: str
    mip_gap: float | None


@dataclass(frozen=True)
class O3Result:
    weights: np.ndarray
    objective: float
    data_objective: float
    temporal_smoothness: float
    frequency_smoothness: float
    iterations: int
    converged: bool
    projected_gradient_norm: float
    selected_start: str
    start_objectives: dict[str, float]
    final_objectives: dict[str, float]
    start_projected_gradient_norms: dict[str, float]
    start_iterations: dict[str, int]
    objective_spread: float


def _as_complex_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty (K,N) matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_complex_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128).reshape(-1)
    if arr.size < 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite non-empty vector")
    return arr


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize and remove only scale-relative numerical negativity."""

    value = np.asarray(matrix, dtype=np.float64)
    q = 0.5 * (value + value.T)
    eigenvalues = np.linalg.eigvalsh(q)
    minimum = float(eigenvalues.min(initial=0.0))
    tolerance = _psd_tolerance(eigenvalues)
    if minimum < -tolerance:
        raise RoutingMathError(
            "derived quadratic is materially non-PSD: "
            f"min_eigenvalue={minimum}, tolerance={tolerance}"
        )
    if minimum < 0.0:
        q = q + np.eye(q.shape[0], dtype=np.float64) * (
            -minimum + tolerance
        )
    return q


def build_exact_cell_quadratic(
    candidates: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    alpha_weight: float = 1.0,
    voice_weight: float = 1.0,
    artifact_weight: float = 1.0,
    direct_weight: float = 1.0,
    ridge_relative: float = 1e-8,
    max_condition: float = 1e6,
    accompaniment_floor: float = 1e-12,
    vocal_floor: float = 1e-12,
    reference_floor: float = 1e-12,
) -> CellQuadratic:
    """Build one convex exact-target cost for real simplex weights.

    For identifiable source coordinates,

        Y_i = alpha_i A + beta_i V + R_i

    the cost is

        lambda_a |sum_i w_i alpha_i - 1|^2
      + lambda_v |sum_i w_i beta_i|^2 E_V / (E_A + floor)
      + lambda_r ||sum_i w_i R_i||^2 / (E_A + floor).

    The denominators are independent of w, so the cost is a convex quadratic.

    If A/V coordinates are unavailable (A-only, V-only, nearly silent, or
    ill-conditioned), exact direct-target error is used:

        lambda_d ||sum_i w_i Y_i - A||^2 / max(E_A + E_V, floor).

    Thus no exact non-empty cell disappears from the routing objective.
    """

    ys = _as_complex_rows(candidates, name="candidates")
    a = _as_complex_vector(accompaniment, name="accompaniment")
    v = _as_complex_vector(vocal, name="vocal")
    if ys.shape[1] != a.size or v.size != a.size:
        raise ValueError("candidate and truth cell lengths differ")
    for name, value in (
        ("alpha_weight", alpha_weight),
        ("voice_weight", voice_weight),
        ("artifact_weight", artifact_weight),
        ("direct_weight", direct_weight),
        ("ridge_relative", ridge_relative),
        ("accompaniment_floor", accompaniment_floor),
        ("vocal_floor", vocal_floor),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(float(reference_floor)) or float(reference_floor) <= 0:
        raise ValueError("reference_floor must be finite and strictly positive")
    if not math.isfinite(float(max_condition)) or float(max_condition) <= 1.0:
        raise ValueError("max_condition must be finite and greater than one")
    if float(alpha_weight) <= 0.0:
        raise ValueError("alpha_weight must be strictly positive")
    if float(direct_weight) <= 0.0:
        raise ValueError("direct_weight must be strictly positive")

    ea = float(np.real(np.vdot(a, a)))
    ev = float(np.real(np.vdot(v, v)))
    x = np.column_stack((a, v))
    gram = x.conj().T @ x
    try:
        condition = float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        condition = math.inf

    identifiable = (
        ea > float(accompaniment_floor)
        and ev > float(vocal_floor)
        and math.isfinite(condition)
        and condition <= float(max_condition)
    )

    k = ys.shape[0]
    if identifiable:
        ridge = float(ridge_relative) * max(
            float(np.trace(gram).real) / 2.0, _EPS
        )
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(2, dtype=np.complex128),
            x.conj().T @ ys.T,
        )
        alpha = coefficients[0]
        beta = coefficients[1]
        residuals = ys - alpha[:, None] * a[None] - beta[:, None] * v[None]

        alpha_gram = np.real(np.outer(np.conj(alpha), alpha))
        beta_gram = np.real(np.outer(np.conj(beta), beta))
        residual_gram = np.real(residuals.conj() @ residuals.T)
        scale = max(ea, float(reference_floor))

        q = (
            float(alpha_weight) * alpha_gram
            + float(voice_weight) * (ev / scale) * beta_gram
            + float(artifact_weight) * (residual_gram / scale)
        )
        c = float(alpha_weight) * np.real(alpha)
        constant = float(alpha_weight)
        mode = "source_coordinates"
    else:
        errors = ys - a[None]
        scale = max(ea + ev, float(reference_floor))
        q = float(direct_weight) * np.real(errors.conj() @ errors.T) / scale
        c = np.zeros(k, dtype=np.float64)
        constant = 0.0
        if ea <= accompaniment_floor and ev <= vocal_floor:
            mode = "silent_direct_fallback"
        elif ev <= vocal_floor:
            mode = "no_vocal_direct_fallback"
        elif ea <= accompaniment_floor:
            mode = "vocal_only_direct_fallback"
        else:
            mode = "ill_conditioned_direct_fallback"

    q = _nearest_psd(q)
    return CellQuadratic(
        Q=q,
        c=np.asarray(c, dtype=np.float64),
        constant=float(constant),
        mode=mode,
        accompaniment_energy=ea,
        vocal_energy=ev,
        condition_number=(condition if math.isfinite(condition) else None),
    )


def stack_cells(
    cells: Sequence[Sequence[CellQuadratic]],
    *,
    measure: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> QuadraticGrid:
    """Convert a rectangular nested cell collection into a validated grid."""

    if not cells or not cells[0]:
        raise ValueError("at least one time/band cell is required")
    bands = len(cells[0])
    if any(len(row) != bands for row in cells):
        raise ValueError("cell grid must be rectangular")
    candidates = cells[0][0].Q.shape[0]
    if any(cell.Q.shape != (candidates, candidates)
           for row in cells for cell in row):
        raise ValueError("all cells must use the same candidate basis")

    q = np.stack([[cell.Q for cell in row] for row in cells])
    c = np.stack([[cell.c for cell in row] for row in cells])
    constant = np.asarray(
        [[cell.constant for cell in row] for row in cells], dtype=np.float64
    )
    modes = np.asarray(
        [[cell.mode for cell in row] for row in cells], dtype=object
    )
    cell_measure = (
        np.ones((len(cells), bands), dtype=np.float64)
        if measure is None
        else np.asarray(measure, dtype=np.float64)
    )
    grid = QuadraticGrid(q, c, constant, modes, cell_measure)
    grid.validate()
    return grid


def project_simplex(value: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex along the last axis."""

    x = np.asarray(value, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("simplex input must be finite with a candidate axis")
    k = x.shape[-1]
    flat = x.reshape(-1, k)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cssv = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, k + 1, dtype=np.float64)
    positive = ordered - cssv / ranks > 0
    rho = positive.sum(axis=1) - 1
    if np.any(rho < 0):
        raise RoutingMathError("simplex projection failed to find support")
    theta = cssv[np.arange(len(flat)), rho] / (rho + 1)
    projected = np.maximum(flat - theta[:, None], 0.0)
    result = projected.reshape(x.shape)
    if np.max(np.abs(result.sum(axis=-1) - 1.0)) > 1e-10:
        raise RoutingMathError("simplex projection sum invariant failed")
    return result


def unary_costs(grid: QuadraticGrid) -> np.ndarray:
    """Cost of each one-hot candidate in every cell."""

    grid.validate()
    diagonal = np.diagonal(grid.Q, axis1=-2, axis2=-1)
    return diagonal - 2.0 * grid.c + grid.constant[..., None]


def route_objective(
    weights: np.ndarray,
    grid: QuadraticGrid,
    *,
    temporal_smoothness: float = 0.0,
    frequency_smoothness: float = 0.0,
) -> tuple[float, float, float, float]:
    """Evaluate the complete convex objective on a simplex grid."""

    t, b, k = grid.validate()
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (t, b, k):
        raise ValueError("weight grid shape does not match quadratics")
    if np.any(w < -1e-10) or not np.allclose(
        w.sum(axis=-1), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("weights must lie on the per-cell simplex")
    if temporal_smoothness < 0 or frequency_smoothness < 0:
        raise ValueError("smoothness weights must be non-negative")

    data_cells = (
        np.einsum("tbk,tbkl,tbl->tb", w, grid.Q, w)
        - 2.0 * np.einsum("tbk,tbk->tb", grid.c, w)
        + grid.constant
    )
    cell_weights = grid.normalized_measure()
    count = float(t * b)
    data = float(np.sum(cell_weights * data_cells))
    temporal = float(np.square(w[1:] - w[:-1]).sum() / count)
    frequency = float(np.square(w[:, 1:] - w[:, :-1]).sum() / count)
    total = (
        data
        + float(temporal_smoothness) * temporal
        + float(frequency_smoothness) * frequency
    )
    return total, data, temporal, frequency


def best_whole_track(grid: QuadraticGrid) -> tuple[int, np.ndarray]:
    """O1: best whole-work vertex under the complete exact cell objective."""

    costs = np.einsum(
        "tb,tbk->k", grid.normalized_measure(), unary_costs(grid)
    )
    return int(np.argmin(costs)), costs


def one_hot(labels: np.ndarray, candidates: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or np.any(labels < 0) or np.any(labels >= candidates):
        raise ValueError("invalid route labels")
    result = np.zeros(labels.shape + (candidates,), dtype=np.float64)
    np.put_along_axis(result, labels[..., None], 1.0, axis=-1)
    return result


def _edges(t: int, b: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    temporal = [
        (time * b + band, (time + 1) * b + band)
        for time in range(t - 1)
        for band in range(b)
    ]
    frequency = [
        (time * b + band, time * b + band + 1)
        for time in range(t)
        for band in range(b - 1)
    ]
    return temporal, frequency


def _o2_objective(
    labels: np.ndarray,
    grid: QuadraticGrid,
    *,
    temporal_switch_penalty: float,
    frequency_switch_penalty: float,
) -> tuple[float, float, int, int]:
    t, b, k = grid.validate()
    lab = np.asarray(labels, dtype=np.int64)
    if lab.shape != (t, b) or np.any(lab < 0) or np.any(lab >= k):
        raise ValueError("invalid O2 labels")
    unary = unary_costs(grid)
    chosen = np.take_along_axis(unary, lab[..., None], axis=-1)[..., 0]
    count = float(t * b)
    data = float(np.sum(grid.normalized_measure() * chosen))
    temporal = int(np.count_nonzero(lab[1:] != lab[:-1]))
    frequency = int(np.count_nonzero(lab[:, 1:] != lab[:, :-1]))
    total = (
        data
        + float(temporal_switch_penalty) * temporal / count
        + float(frequency_switch_penalty) * frequency / count
    )
    return total, data, temporal, frequency


def solve_o2_global(
    grid: QuadraticGrid,
    *,
    temporal_switch_penalty: float = 0.05,
    frequency_switch_penalty: float = 0.05,
    time_limit_seconds: float = 180.0,
    mip_relative_gap: float = 0.0,
) -> O2Result:
    """O2: globally solve the finite Potts labeling problem by MILP."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    t, b, k = grid.validate()
    for name, value in (
        ("temporal_switch_penalty", temporal_switch_penalty),
        ("frequency_switch_penalty", frequency_switch_penalty),
        ("mip_relative_gap", mip_relative_gap),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(float(time_limit_seconds)) or float(time_limit_seconds) <= 0:
        raise ValueError("time_limit_seconds must be finite and strictly positive")
    if float(mip_relative_gap) >= 1.0:
        raise ValueError("mip_relative_gap must be less than one")

    cells = t * b
    temporal_edges, frequency_edges = _edges(t, b)
    edges = [
        (u, v, float(temporal_switch_penalty)) for u, v in temporal_edges
    ] + [
        (u, v, float(frequency_switch_penalty)) for u, v in frequency_edges
    ]

    x_count = cells * k
    d_count = len(edges) * k
    variables = x_count + d_count
    count = float(cells)

    objective = np.zeros(variables, dtype=np.float64)
    objective[:x_count] = (
        unary_costs(grid) * grid.normalized_measure()[..., None]
    ).reshape(-1)
    for edge_index, (_, _, penalty) in enumerate(edges):
        # For one-hot endpoints, sum_k |x_uk-x_vk| is 2 on a switch.
        objective[
            x_count + edge_index * k : x_count + (edge_index + 1) * k
        ] = penalty / (2.0 * count)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    for cell in range(cells):
        for candidate in range(k):
            rows.append(row)
            columns.append(cell * k + candidate)
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        row += 1

    for edge_index, (left_cell, right_cell, _) in enumerate(edges):
        for candidate in range(k):
            d_index = x_count + edge_index * k + candidate
            for left, right in (
                (left_cell, right_cell),
                (right_cell, left_cell),
            ):
                rows.extend((row, row, row))
                columns.extend(
                    (left * k + candidate, right * k + candidate, d_index)
                )
                values.extend((1.0, -1.0, -1.0))
                lower.append(-np.inf)
                upper.append(0.0)
                row += 1

    constraints = coo_matrix(
        (values, (rows, columns)), shape=(row, variables)
    ).tocsr()
    result = milp(
        objective,
        integrality=np.r_[
            np.ones(x_count, dtype=np.int8),
            np.zeros(d_count, dtype=np.int8),
        ],
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(
            constraints, np.asarray(lower), np.asarray(upper)
        ),
        options={
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": float(mip_relative_gap),
        },
    )
    if result.x is None:
        raise RoutingMathError(
            f"O2 MILP produced no incumbent: status={result.status}, "
            f"message={result.message}"
        )
    gap = getattr(result, "mip_gap", None)
    gap_value = None if gap is None else float(gap)
    if not result.success:
        raise RoutingMathError(
            f"O2 MILP was not certified optimal: status={result.status}, "
            f"message={result.message}, gap={gap_value}"
        )
    if gap_value is None or not math.isfinite(gap_value):
        raise RoutingMathError("O2 MILP did not publish a finite optimality gap")
    if gap_value > float(mip_relative_gap) + 1e-12:
        raise RoutingMathError(
            f"O2 MILP gap {gap_value} exceeds requested {mip_relative_gap}"
        )

    x_values = np.asarray(result.x[:x_count], dtype=np.float64).reshape(t, b, k)
    if np.max(np.abs(x_values - np.rint(x_values)), initial=0.0) > 1e-7:
        raise RoutingMathError("O2 MILP returned a non-integral label assignment")
    if np.max(np.abs(x_values.sum(axis=-1) - 1.0), initial=0.0) > 1e-7:
        raise RoutingMathError("O2 MILP label assignment violates one-hot sums")
    labels = np.argmax(x_values, axis=-1)
    total, data, temporal, frequency = _o2_objective(
        labels,
        grid,
        temporal_switch_penalty=temporal_switch_penalty,
        frequency_switch_penalty=frequency_switch_penalty,
    )
    if result.fun is None or abs(float(result.fun) - total) > max(
        1e-9, 1e-9 * max(1.0, abs(total))
    ):
        raise RoutingMathError(
            f"O2 solver/objective mismatch: {result.fun} != {total}"
        )
    return O2Result(
        labels=labels,
        objective=total,
        data_objective=data,
        temporal_switches=temporal,
        frequency_switches=frequency,
        solver_status=str(result.message),
        mip_gap=gap_value,
    )


def _gradient(
    weights: np.ndarray,
    grid: QuadraticGrid,
    *,
    temporal_smoothness: float,
    frequency_smoothness: float,
) -> np.ndarray:
    """Gradient of the exact weighted convex O3 objective."""

    t, b, _ = grid.validate()
    count = float(t * b)
    measure = grid.normalized_measure()[..., None]
    gradient = measure * (
        2.0 * np.einsum("tbkl,tbl->tbk", grid.Q, weights)
        - 2.0 * grid.c
    )
    if temporal_smoothness:
        difference = weights[1:] - weights[:-1]
        gradient[1:] += (
            2.0 * float(temporal_smoothness) * difference / count
        )
        gradient[:-1] -= (
            2.0 * float(temporal_smoothness) * difference / count
        )
    if frequency_smoothness:
        difference = weights[:, 1:] - weights[:, :-1]
        gradient[:, 1:] += (
            2.0 * float(frequency_smoothness) * difference / count
        )
        gradient[:, :-1] -= (
            2.0 * float(frequency_smoothness) * difference / count
        )
    return gradient


def _lipschitz_bound(
    grid: QuadraticGrid,
    *,
    temporal_smoothness: float,
    frequency_smoothness: float,
) -> float:
    """A certified upper bound for the complete objective Hessian norm."""

    t, b, _ = grid.validate()
    count = float(t * b)
    measure = grid.normalized_measure().reshape(-1)
    matrices = grid.Q.reshape(
        -1, grid.Q.shape[-1], grid.Q.shape[-1]
    )
    data_bound = max(
        2.0 * float(cell_measure)
        * max(float(np.linalg.eigvalsh(matrix).max(initial=0.0)), 0.0)
        for cell_measure, matrix in zip(measure, matrices)
    )
    # A path-graph Laplacian has spectral norm at most four.  Each squared
    # difference term contributes ``2 * lambda * L / (T*B)`` to the Hessian.
    smoothness_bound = (
        8.0 * float(temporal_smoothness) / count
        + 8.0 * float(frequency_smoothness) / count
    )
    return max(data_bound + smoothness_bound, _EPS)


def _initial_weights(
    name: str,
    grid: QuadraticGrid,
    *,
    o1_index: int,
    o2_labels: np.ndarray | None,
) -> np.ndarray:
    t, b, k = grid.validate()
    if name == "uniform":
        return np.full((t, b, k), 1.0 / k, dtype=np.float64)
    if name == "O1":
        labels = np.full((t, b), int(o1_index), dtype=np.int64)
        return one_hot(labels, k)
    if name == "independent":
        return one_hot(np.argmin(unary_costs(grid), axis=-1), k)
    if name == "O2":
        if o2_labels is None:
            raise ValueError("O2 initialization requested without labels")
        return one_hot(o2_labels, k)
    raise ValueError(f"unknown initialization {name!r}")


def solve_o3_convex(
    grid: QuadraticGrid,
    *,
    temporal_smoothness: float = 0.05,
    frequency_smoothness: float = 0.05,
    tolerance: float = 1e-7,
    max_iterations: int = 20_000,
    o1_index: int | None = None,
    o2_labels: np.ndarray | None = None,
    starts: Iterable[str] = ("uniform", "O1", "independent", "O2"),
) -> O3Result:
    """O3: solve the convex local hull with a returned-iterate certificate.

    Every declared deterministic start must converge.  The solver uses monotone
    accelerated projected gradient with restart, then recomputes the infinity
    norm of the projected-gradient mapping *at the returned iterate*.  Convex
    starts must agree on their final objective within a scale-aware tolerance.
    """

    t, b, k = grid.validate()
    for name, value in (
        ("temporal_smoothness", temporal_smoothness),
        ("frequency_smoothness", frequency_smoothness),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError("tolerance must be finite and strictly positive")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    declared_starts = tuple(str(value) for value in starts)
    if not declared_starts or any(not value for value in declared_starts):
        raise ValueError("at least one non-empty O3 start is required")
    if len(set(declared_starts)) != len(declared_starts):
        raise ValueError("O3 starts must be unique")
    allowed = {"uniform", "O1", "independent", "O2"}
    unknown = set(declared_starts) - allowed
    if unknown:
        raise ValueError(f"unknown O3 starts: {sorted(unknown)}")
    if "O2" in declared_starts and o2_labels is None:
        raise ValueError("O2 initialization requested without labels")

    if o1_index is None:
        o1_index, _ = best_whole_track(grid)
    if not isinstance(o1_index, (int, np.integer)) or not 0 <= int(o1_index) < k:
        raise ValueError("o1_index is outside the candidate basis")

    lipschitz = _lipschitz_bound(
        grid,
        temporal_smoothness=temporal_smoothness,
        frequency_smoothness=frequency_smoothness,
    )
    step = 1.0 / lipschitz
    count = float(t * b)
    measure = grid.normalized_measure()
    q = np.asarray(grid.Q, dtype=np.float64)
    c = np.asarray(grid.c, dtype=np.float64)
    constant = np.asarray(grid.constant, dtype=np.float64)
    temporal_lambda = float(temporal_smoothness)
    frequency_lambda = float(frequency_smoothness)

    def objective_parts(value: np.ndarray) -> tuple[float, float, float, float]:
        data_cells = (
            np.einsum("tbk,tbkl,tbl->tb", value, q, value)
            - 2.0 * np.einsum("tbk,tbk->tb", c, value)
            + constant
        )
        data = float(np.sum(measure * data_cells))
        temporal = float(np.square(value[1:] - value[:-1]).sum() / count)
        frequency = float(np.square(value[:, 1:] - value[:, :-1]).sum() / count)
        total = data + temporal_lambda * temporal + frequency_lambda * frequency
        return total, data, temporal, frequency

    def gradient_unchecked(value: np.ndarray) -> np.ndarray:
        gradient = measure[..., None] * (
            2.0 * np.einsum("tbkl,tbl->tbk", q, value) - 2.0 * c
        )
        if temporal_lambda:
            difference = value[1:] - value[:-1]
            gradient[1:] += 2.0 * temporal_lambda * difference / count
            gradient[:-1] -= 2.0 * temporal_lambda * difference / count
        if frequency_lambda:
            difference = value[:, 1:] - value[:, :-1]
            gradient[:, 1:] += 2.0 * frequency_lambda * difference / count
            gradient[:, :-1] -= 2.0 * frequency_lambda * difference / count
        return gradient

    def mapping_norm(value: np.ndarray) -> float:
        projected = project_simplex(value - step * gradient_unchecked(value))
        return float(np.max(np.abs((value - projected) / step), initial=0.0))

    start_objectives: dict[str, float] = {}
    final_objectives: dict[str, float] = {}
    projected_norms: dict[str, float] = {}
    start_iterations: dict[str, int] = {}
    solutions: dict[str, np.ndarray] = {}
    failures: dict[str, dict[str, float | int]] = {}

    for name in declared_starts:
        weights = _initial_weights(
            name, grid, o1_index=int(o1_index), o2_labels=o2_labels
        )
        extrapolated = weights.copy()
        momentum = 1.0
        previous = objective_parts(weights)[0]
        start_objectives[name] = previous
        returned_norm = mapping_norm(weights)
        iterations = 0
        converged = returned_norm <= float(tolerance)

        for iterations in range(1, max_iterations + 1):
            if converged:
                iterations -= 1
                break
            proposal = project_simplex(
                extrapolated - step * gradient_unchecked(extrapolated)
            )
            current = objective_parts(proposal)[0]
            monotone_tolerance = max(
                1e-12,
                1e-11 * max(1.0, abs(previous), abs(current)),
            )
            if current > previous + monotone_tolerance:
                # Restart at the last accepted feasible iterate.  The fallback
                # projected-gradient step must itself satisfy descent.
                extrapolated = weights
                momentum = 1.0
                proposal = project_simplex(
                    weights - step * gradient_unchecked(weights)
                )
                current = objective_parts(proposal)[0]
                monotone_tolerance = max(
                    1e-12,
                    1e-11 * max(1.0, abs(previous), abs(current)),
                )
            if current > previous + monotone_tolerance:
                raise RoutingMathError(
                    f"O3 objective increased for {name}: "
                    f"{previous} -> {current}"
                )

            next_momentum = 0.5 * (
                1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
            )
            next_extrapolated = proposal + (
                (momentum - 1.0) / next_momentum
            ) * (proposal - weights)
            weights = proposal
            extrapolated = next_extrapolated
            momentum = next_momentum
            previous = current
            returned_norm = mapping_norm(weights)
            converged = returned_norm <= float(tolerance)
            if converged:
                break

        final_objectives[name] = objective_parts(weights)[0]
        projected_norms[name] = returned_norm
        start_iterations[name] = iterations
        if not converged:
            failures[name] = {
                "objective": final_objectives[name],
                "iterations": iterations,
                "projected_gradient_norm": returned_norm,
            }
        else:
            solutions[name] = weights.copy()

    if failures:
        raise RoutingMathError(
            f"O3 did not converge from every declared start: {failures}"
        )

    objective_values = tuple(final_objectives[name] for name in declared_starts)
    objective_spread = max(objective_values) - min(objective_values)
    objective_scale = max(1.0, *(abs(value) for value in objective_values))
    agreement_tolerance = max(
        1e-10,
        10.0 * float(tolerance),
        1e-9 * objective_scale,
    )
    if objective_spread > agreement_tolerance:
        raise RoutingMathError(
            "convex O3 starts disagree after convergence: "
            f"objectives={final_objectives}, spread={objective_spread}, "
            f"tolerance={agreement_tolerance}"
        )

    selected_start = min(
        declared_starts,
        key=lambda name: (final_objectives[name], name),
    )
    weights = solutions[selected_start]
    certified_norm = mapping_norm(weights)
    if certified_norm > float(tolerance):
        raise RoutingMathError(
            "returned O3 iterate failed independent mapping recomputation: "
            f"{certified_norm} > {tolerance}"
        )
    total, data, temporal, frequency = objective_parts(weights)
    if abs(total - final_objectives[selected_start]) > max(
        1e-12, 1e-10 * max(1.0, abs(total))
    ):
        raise RoutingMathError("returned O3 objective changed during certification")

    return O3Result(
        weights=weights,
        objective=total,
        data_objective=data,
        temporal_smoothness=temporal,
        frequency_smoothness=frequency,
        iterations=start_iterations[selected_start],
        converged=True,
        projected_gradient_norm=certified_norm,
        selected_start=selected_start,
        start_objectives=start_objectives,
        final_objectives=final_objectives,
        start_projected_gradient_norms=projected_norms,
        start_iterations=start_iterations,
        objective_spread=objective_spread,
    )
