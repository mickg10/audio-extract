"""Certified convex mathematics for exact O1/O2/O3 routing envelopes.

The functions in this module consume clean accompaniment/featured-solo truth and
are therefore research diagnostics, not production selectors.  Every exact cell
contributes: identifiable cells use convex source-coordinate risk, while
A-only, V-only, silent, and ill-conditioned cells use exact direct-target risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

import numpy as np

_TINY = np.finfo(np.float64).tiny


class RoutingMathError(RuntimeError):
    """The routing input, optimizer, or certificate is invalid."""


@dataclass(frozen=True)
class CellQuadratic:
    """One cell objective ``w.T @ Q @ w - 2*c.T @ w + constant``."""

    Q: np.ndarray
    c: np.ndarray
    constant: float
    mode: str
    accompaniment_energy: float
    vocal_energy: float
    condition_number: float | None


@dataclass(frozen=True)
class QuadraticGrid:
    Q: np.ndarray          # (time, band, candidates, candidates)
    c: np.ndarray          # (time, band, candidates)
    constant: np.ndarray   # (time, band)
    modes: np.ndarray      # (time, band), strings

    def validate(self) -> tuple[int, int, int]:
        q = np.asarray(self.Q, dtype=np.float64)
        c = np.asarray(self.c, dtype=np.float64)
        constant = np.asarray(self.constant, dtype=np.float64)
        modes = np.asarray(self.modes)
        if q.ndim != 4 or q.shape[-1] != q.shape[-2]:
            raise ValueError("Q must have shape (time, band, K, K)")
        if c.shape != q.shape[:-1] or constant.shape != q.shape[:2]:
            raise ValueError("c/constant shapes do not match Q")
        if modes.shape != q.shape[:2]:
            raise ValueError("mode grid does not match Q")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(c))
                and np.all(np.isfinite(constant))):
            raise ValueError("quadratic grid contains non-finite values")
        if np.max(np.abs(q - np.swapaxes(q, -1, -2))) > 1e-9:
            raise ValueError("Q is not symmetric")
        for matrix in q.reshape(-1, q.shape[-1], q.shape[-1]):
            if float(np.linalg.eigvalsh(matrix).min()) < -1e-8:
                raise ValueError("Q contains a materially non-PSD cell")
        return q.shape[0], q.shape[1], q.shape[2]


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
    projected_gradient_norm: float
    start_objectives: dict[str, float]


def _complex_rows(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a non-empty (K,N) matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _complex_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128).reshape(-1)
    if result.size < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and non-empty")
    return result


def _psd(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    result = 0.5 * (
        np.asarray(matrix, dtype=np.float64)
        + np.asarray(matrix, dtype=np.float64).T
    )
    minimum = float(np.linalg.eigvalsh(result).min())
    if minimum < -tolerance:
        raise RoutingMathError(
            f"derived quadratic is materially non-PSD: {minimum}"
        )
    if minimum < 0.0:
        result = result + np.eye(result.shape[0]) * (-minimum + tolerance)
    return result


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
    """Build one convex exact-target objective for real simplex weights.

    Identifiable cells use

    ``la*|alpha(w)-1|^2 + lv*|beta(w)|^2*Ev/(Ea+floor)
       + lr*||R(w)||^2/(Ea+floor)``.

    Other cells use ``ld*||sum_i w_i*(Y_i-A)||^2/max(Ea+Ev,floor)``.
    """

    ys = _complex_rows(candidates, "candidates")
    a = _complex_vector(accompaniment, "accompaniment")
    v = _complex_vector(vocal, "vocal")
    if ys.shape[1] != a.size or v.size != a.size:
        raise ValueError("candidate and truth cell lengths differ")
    for name, value in (
        ("alpha_weight", alpha_weight), ("voice_weight", voice_weight),
        ("artifact_weight", artifact_weight), ("direct_weight", direct_weight),
        ("ridge_relative", ridge_relative), ("max_condition", max_condition),
        ("accompaniment_floor", accompaniment_floor),
        ("vocal_floor", vocal_floor), ("reference_floor", reference_floor),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    ea = float(np.real(np.vdot(a, a)))
    ev = float(np.real(np.vdot(v, v)))
    basis = np.column_stack((a, v))
    gram = basis.conj().T @ basis
    try:
        condition = float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        condition = math.inf
    identifiable = (
        ea > accompaniment_floor and ev > vocal_floor
        and math.isfinite(condition) and condition <= max_condition
    )

    if identifiable:
        ridge = ridge_relative * max(float(np.trace(gram).real) / 2.0, _TINY)
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(2, dtype=np.complex128),
            basis.conj().T @ ys.T,
        )
        alpha, beta = coefficients[0], coefficients[1]
        residuals = ys - alpha[:, None] * a - beta[:, None] * v
        scale = max(ea, reference_floor)
        Q = (
            alpha_weight * np.real(np.outer(np.conj(alpha), alpha))
            + voice_weight * (ev / scale)
            * np.real(np.outer(np.conj(beta), beta))
            + artifact_weight
            * np.real(residuals.conj() @ residuals.T) / scale
        )
        c = alpha_weight * np.real(alpha)
        constant = alpha_weight
        mode = "source_coordinates"
    else:
        errors = ys - a[None]
        scale = max(ea + ev, reference_floor)
        Q = direct_weight * np.real(errors.conj() @ errors.T) / scale
        c = np.zeros(ys.shape[0], dtype=np.float64)
        constant = 0.0
        if ea <= accompaniment_floor and ev <= vocal_floor:
            mode = "silent_direct_fallback"
        elif ev <= vocal_floor:
            mode = "no_vocal_direct_fallback"
        elif ea <= accompaniment_floor:
            mode = "vocal_only_direct_fallback"
        else:
            mode = "ill_conditioned_direct_fallback"

    return CellQuadratic(
        Q=_psd(Q), c=np.asarray(c, dtype=np.float64),
        constant=float(constant), mode=mode,
        accompaniment_energy=ea, vocal_energy=ev,
        condition_number=condition if math.isfinite(condition) else None,
    )


def stack_cells(cells: Sequence[Sequence[CellQuadratic]]) -> QuadraticGrid:
    if not cells or not cells[0]:
        raise ValueError("cell grid is empty")
    bands = len(cells[0])
    if any(len(row) != bands for row in cells):
        raise ValueError("cell grid must be rectangular")
    candidates = cells[0][0].Q.shape[0]
    if any(cell.Q.shape != (candidates, candidates)
           for row in cells for cell in row):
        raise ValueError("candidate basis changes across cells")
    result = QuadraticGrid(
        Q=np.stack([[cell.Q for cell in row] for row in cells]),
        c=np.stack([[cell.c for cell in row] for row in cells]),
        constant=np.asarray(
            [[cell.constant for cell in row] for row in cells], dtype=np.float64
        ),
        modes=np.asarray([[cell.mode for cell in row] for row in cells]),
    )
    result.validate()
    return result


def project_simplex(value: np.ndarray) -> np.ndarray:
    """Euclidean projection onto a probability simplex on the last axis."""

    x = np.asarray(value, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("simplex input must be finite")
    k = x.shape[-1]
    flat = x.reshape(-1, k)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cssv = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, k + 1, dtype=np.float64)
    positive = ordered - cssv / ranks > 0
    rho = positive.sum(axis=1) - 1
    if np.any(rho < 0):
        raise RoutingMathError("simplex projection has empty support")
    theta = cssv[np.arange(len(flat)), rho] / (rho + 1)
    result = np.maximum(flat - theta[:, None], 0.0).reshape(x.shape)
    if np.max(np.abs(result.sum(axis=-1) - 1.0)) > 1e-10:
        raise RoutingMathError("simplex sum invariant failed")
    return result


def unary_costs(grid: QuadraticGrid) -> np.ndarray:
    grid.validate()
    return (
        np.diagonal(grid.Q, axis1=-2, axis2=-1)
        - 2.0 * grid.c + grid.constant[..., None]
    )


def best_whole_track(grid: QuadraticGrid) -> tuple[int, np.ndarray]:
    costs = unary_costs(grid).mean(axis=(0, 1))
    return int(np.argmin(costs)), costs


def one_hot(labels: np.ndarray, candidates: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or np.any(labels < 0) or np.any(labels >= candidates):
        raise ValueError("invalid routing labels")
    result = np.zeros(labels.shape + (candidates,), dtype=np.float64)
    np.put_along_axis(result, labels[..., None], 1.0, axis=-1)
    return result


def route_objective(
    weights: np.ndarray,
    grid: QuadraticGrid,
    *,
    temporal_smoothness: float = 0.0,
    frequency_smoothness: float = 0.0,
) -> tuple[float, float, float, float]:
    time, bands, candidates = grid.validate()
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (time, bands, candidates):
        raise ValueError("weight grid does not match quadratic grid")
    if np.any(w < -1e-10) or not np.allclose(
        w.sum(-1), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("weights are outside the simplex")
    if temporal_smoothness < 0 or frequency_smoothness < 0:
        raise ValueError("smoothness must be non-negative")
    count = float(time * bands)
    cells = (
        np.einsum("tbk,tbkl,tbl->tb", w, grid.Q, w)
        - 2.0 * np.einsum("tbk,tbk->tb", grid.c, w)
        + grid.constant
    )
    data = float(cells.sum() / count)
    temporal = float(np.square(w[1:] - w[:-1]).sum() / count)
    frequency = float(np.square(w[:, 1:] - w[:, :-1]).sum() / count)
    total = data + temporal_smoothness * temporal + frequency_smoothness * frequency
    return total, data, temporal, frequency


def _edges(time: int, bands: int):
    temporal = [
        (t * bands + b, (t + 1) * bands + b)
        for t in range(time - 1) for b in range(bands)
    ]
    frequency = [
        (t * bands + b, t * bands + b + 1)
        for t in range(time) for b in range(bands - 1)
    ]
    return temporal, frequency


def solve_o2_global(
    grid: QuadraticGrid,
    *,
    temporal_switch_penalty: float = 0.05,
    frequency_switch_penalty: float = 0.05,
    time_limit_seconds: float = 180.0,
    mip_relative_gap: float = 0.0,
) -> O2Result:
    """Globally solve the finite Potts-labeling problem with SciPy MILP."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    time, bands, candidates = grid.validate()
    cells = time * bands
    temporal_edges, frequency_edges = _edges(time, bands)
    edges = (
        [(u, v, temporal_switch_penalty) for u, v in temporal_edges]
        + [(u, v, frequency_switch_penalty) for u, v in frequency_edges]
    )
    x_count = cells * candidates
    d_count = len(edges) * candidates
    variable_count = x_count + d_count
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:x_count] = unary_costs(grid).reshape(-1) / cells
    for edge_index, (_, _, penalty) in enumerate(edges):
        objective[
            x_count + edge_index * candidates:
            x_count + (edge_index + 1) * candidates
        ] = penalty / (2.0 * cells)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for cell in range(cells):
        for candidate in range(candidates):
            rows.append(row); columns.append(cell * candidates + candidate)
            values.append(1.0)
        lower.append(1.0); upper.append(1.0); row += 1
    for edge_index, (left_cell, right_cell, _) in enumerate(edges):
        for candidate in range(candidates):
            d_index = x_count + edge_index * candidates + candidate
            for left, right in ((left_cell, right_cell), (right_cell, left_cell)):
                rows.extend((row, row, row))
                columns.extend((
                    left * candidates + candidate,
                    right * candidates + candidate,
                    d_index,
                ))
                values.extend((1.0, -1.0, -1.0))
                lower.append(-np.inf); upper.append(0.0); row += 1
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(row, variable_count)
    ).tocsr()
    result = milp(
        objective,
        integrality=np.r_[
            np.ones(x_count, dtype=np.int8),
            np.zeros(d_count, dtype=np.int8),
        ],
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": mip_relative_gap},
    )
    gap = getattr(result, "mip_gap", None)
    gap_value = None if gap is None else float(gap)
    if not result.success or result.x is None:
        raise RoutingMathError(
            f"O2 lacks an optimality certificate: {result.message}; gap={gap_value}"
        )
    if gap_value is not None and gap_value > mip_relative_gap + 1e-12:
        raise RoutingMathError(f"O2 gap {gap_value} exceeds {mip_relative_gap}")
    labels = np.argmax(result.x[:x_count].reshape(time, bands, candidates), -1)
    unary = unary_costs(grid)
    chosen = np.take_along_axis(unary, labels[..., None], -1)[..., 0]
    data = float(chosen.mean())
    temporal = int(np.count_nonzero(labels[1:] != labels[:-1]))
    frequency = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    total = (
        data + temporal_switch_penalty * temporal / cells
        + frequency_switch_penalty * frequency / cells
    )
    return O2Result(
        labels, total, data, temporal, frequency,
        str(result.message), gap_value,
    )


def _gradient(weights, grid, temporal_smoothness, frequency_smoothness):
    time, bands, _ = grid.validate()
    count = float(time * bands)
    gradient = (
        2.0 * np.einsum("tbkl,tbl->tbk", grid.Q, weights) - 2.0 * grid.c
    ) / count
    if temporal_smoothness:
        difference = weights[1:] - weights[:-1]
        gradient[1:] += 2.0 * temporal_smoothness * difference / count
        gradient[:-1] -= 2.0 * temporal_smoothness * difference / count
    if frequency_smoothness:
        difference = weights[:, 1:] - weights[:, :-1]
        gradient[:, 1:] += 2.0 * frequency_smoothness * difference / count
        gradient[:, :-1] -= 2.0 * frequency_smoothness * difference / count
    return gradient


def _lipschitz(grid, temporal_smoothness, frequency_smoothness):
    time, bands, _ = grid.validate()
    largest = max(
        float(np.linalg.eigvalsh(matrix).max())
        for matrix in grid.Q.reshape(-1, grid.Q.shape[-1], grid.Q.shape[-1])
    )
    return max(
        (2.0 * largest + 8.0 * temporal_smoothness
         + 8.0 * frequency_smoothness) / (time * bands),
        _TINY,
    )


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
    """Solve the convex local-hull problem with projected-gradient certificates."""

    time, bands, candidates = grid.validate()
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid O3 convergence settings")
    if o1_index is None:
        o1_index, _ = best_whole_track(grid)
    step = 1.0 / _lipschitz(
        grid, temporal_smoothness, frequency_smoothness
    )
    solutions = []
    start_objectives: dict[str, float] = {}
    for name in starts:
        if name == "uniform":
            weights = np.full((time, bands, candidates), 1.0 / candidates)
        elif name == "O1":
            weights = one_hot(
                np.full((time, bands), o1_index, dtype=np.int64), candidates
            )
        elif name == "independent":
            weights = one_hot(np.argmin(unary_costs(grid), -1), candidates)
        elif name == "O2":
            if o2_labels is None:
                continue
            weights = one_hot(o2_labels, candidates)
        else:
            raise ValueError(f"unknown O3 start {name!r}")
        previous = route_objective(
            weights, grid, temporal_smoothness=temporal_smoothness,
            frequency_smoothness=frequency_smoothness,
        )[0]
        start_objectives[name] = previous
        converged = False
        projected_norm = math.inf
        for iteration in range(1, max_iterations + 1):
            gradient = _gradient(
                weights, grid, temporal_smoothness, frequency_smoothness
            )
            proposal = project_simplex(weights - step * gradient)
            projected_norm = float(np.linalg.norm((weights - proposal) / step))
            current = route_objective(
                proposal, grid, temporal_smoothness=temporal_smoothness,
                frequency_smoothness=frequency_smoothness,
            )[0]
            if current > previous + 1e-9:
                raise RoutingMathError(
                    f"O3 objective increased {previous} -> {current}"
                )
            weights, previous = proposal, current
            if projected_norm <= tolerance:
                converged = True
                break
        if converged:
            solutions.append((previous, name, weights, iteration, projected_norm))
    if not solutions:
        raise RoutingMathError("O3 did not converge from any deterministic start")
    objectives = [item[0] for item in solutions]
    if max(objectives) - min(objectives) > max(1e-8, 10.0 * tolerance):
        raise RoutingMathError(
            f"convex O3 starts disagree after convergence: {objectives}"
        )
    _, _, weights, iterations, projected_norm = min(solutions, key=lambda x: x[0])
    total, data, temporal, frequency = route_objective(
        weights, grid, temporal_smoothness=temporal_smoothness,
        frequency_smoothness=frequency_smoothness,
    )
    return O3Result(
        weights, total, data, temporal, frequency, iterations,
        projected_norm, start_objectives,
    )
