"""Certified exact-reference routing mathematics for opera separation research.

This module is version-separated from :mod:`audio_extract.oracle_routing`.  It
contains no storage or audio-I/O policy and is deliberately not wired into the
active runner.  It provides the mathematical core for a binding O1/O2/O3
experiment:

* every exact time/frequency cell contributes to route selection;
* identifiable cells use a convex source-coordinate quadratic;
* A-only, V-only, silent, and ill-conditioned cells use exact direct-target
  error rather than becoming zero-cost cells;
* O2 is a globally certified Potts-labeling MILP;
* O3 is a genuinely convex simplex problem with squared graph smoothness and a
  projected-gradient convergence certificate.

The functions consume clean A/V references and therefore define an oracle
research diagnostic, not a deployable separator or reference-free judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

import numpy as np

_EPS = np.finfo(np.float64).tiny


class CertifiedRoutingError(RuntimeError):
    """The exact routing input, optimization, or certificate is invalid."""


@dataclass(frozen=True)
class CertifiedRoutingConfig:
    alpha_weight: float = 1.0
    voice_weight: float = 1.0
    residual_weight: float = 1.0
    direct_weight: float = 1.0
    ridge_relative: float = 1e-8
    max_condition: float = 1e6
    accompaniment_floor: float = 1e-12
    vocal_floor: float = 1e-12
    reference_floor: float = 1e-12
    temporal_switch_penalty: float = 0.05
    frequency_switch_penalty: float = 0.05
    temporal_weight_smoothness: float = 0.05
    frequency_weight_smoothness: float = 0.05
    projected_gradient_tolerance: float = 1e-7
    max_projected_gradient_iterations: int = 20_000

    def validate(self) -> None:
        nonnegative = (
            "alpha_weight",
            "voice_weight",
            "residual_weight",
            "direct_weight",
            "ridge_relative",
            "max_condition",
            "accompaniment_floor",
            "vocal_floor",
            "reference_floor",
            "temporal_switch_penalty",
            "frequency_switch_penalty",
            "temporal_weight_smoothness",
            "frequency_weight_smoothness",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_condition <= 1:
            raise ValueError("max_condition must be greater than one")
        if self.projected_gradient_tolerance <= 0:
            raise ValueError("projected_gradient_tolerance must be positive")
        if self.max_projected_gradient_iterations < 1:
            raise ValueError("max_projected_gradient_iterations must be positive")


@dataclass(frozen=True)
class CellQuadratic:
    """One exact cell cost ``w.T Q w - 2 c.T w + constant``."""

    Q: np.ndarray
    c: np.ndarray
    constant: float
    mode: str
    accompaniment_energy: float
    vocal_energy: float
    condition_number: float | None

    def validate(self) -> int:
        q = np.asarray(self.Q, dtype=np.float64)
        c = np.asarray(self.c, dtype=np.float64)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("Q must be square")
        if c.shape != (q.shape[0],):
            raise ValueError("c must match Q")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(c))
                and math.isfinite(float(self.constant))):
            raise ValueError("quadratic coefficients must be finite")
        if not np.allclose(q, q.T, atol=1e-10, rtol=1e-10):
            raise ValueError("Q must be symmetric")
        if np.linalg.eigvalsh(q).min(initial=0.0) < -1e-8:
            raise ValueError("Q must be positive semidefinite")
        return q.shape[0]


@dataclass(frozen=True)
class QuadraticGrid:
    """A complete time/band grid of exact convex costs."""

    Q: np.ndarray          # (T, B, K, K)
    c: np.ndarray          # (T, B, K)
    constant: np.ndarray   # (T, B)
    modes: np.ndarray      # (T, B)

    def validate(self) -> tuple[int, int, int]:
        q = np.asarray(self.Q, dtype=np.float64)
        c = np.asarray(self.c, dtype=np.float64)
        constant = np.asarray(self.constant, dtype=np.float64)
        if q.ndim != 4 or q.shape[-1] != q.shape[-2]:
            raise ValueError("Q must be (time, band, K, K)")
        if c.shape != q.shape[:-1] or constant.shape != q.shape[:2]:
            raise ValueError("c/constant shapes do not match Q")
        if np.asarray(self.modes).shape != q.shape[:2]:
            raise ValueError("mode grid does not match Q")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(c))
                and np.all(np.isfinite(constant))):
            raise ValueError("grid coefficients must be finite")
        if np.max(np.abs(q - np.swapaxes(q, -1, -2))) > 1e-9:
            raise ValueError("Q grid is not symmetric")
        for matrix in q.reshape(-1, q.shape[-1], q.shape[-1]):
            if np.linalg.eigvalsh(matrix).min(initial=0.0) < -1e-8:
                raise ValueError("Q grid contains a non-PSD matrix")
        return q.shape[0], q.shape[1], q.shape[2]


@dataclass(frozen=True)
class DiscreteRoutingResult:
    labels: np.ndarray
    objective: float
    data_objective: float
    temporal_switches: int
    frequency_switches: int
    solver_status: str
    mip_gap: float | None


@dataclass(frozen=True)
class ConvexRoutingResult:
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


def _complex_rows(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim < 2 or array.shape[0] < 1:
        raise ValueError(f"{name} must have a leading candidate axis")
    array = array.reshape(array.shape[0], -1)
    if array.shape[1] < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite non-empty cells")
    return array


def _complex_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and non-empty")
    return array


def _numerically_psd(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    q = np.asarray(matrix, dtype=np.float64)
    q = 0.5 * (q + q.T)
    minimum = float(np.linalg.eigvalsh(q).min(initial=0.0))
    if minimum < -tolerance:
        raise CertifiedRoutingError(
            f"derived quadratic is materially non-PSD: {minimum}"
        )
    if minimum < 0:
        q = q + np.eye(q.shape[0], dtype=np.float64) * (-minimum + tolerance)
    return q


def build_cell_quadratic(
    candidates: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    config: CertifiedRoutingConfig | None = None,
) -> CellQuadratic:
    """Build one exact convex routing cost.

    On identifiable cells, fit ``Y_i = alpha_i A + beta_i V + R_i`` and use

    ``lambda_a |sum(w_i alpha_i)-1|^2``
    ``+ lambda_v |sum(w_i beta_i)|^2 E_V/(E_A+floor)``
    ``+ lambda_r ||sum(w_i R_i)||^2/(E_A+floor)``.

    The denominators are independent of ``w``.  When source coordinates are not
    identifiable, use exact direct-target error
    ``||sum(w_i Y_i)-A||^2/max(E_A+E_V,floor)``.  Thus no exact cell vanishes.
    """

    cfg = config or CertifiedRoutingConfig()
    cfg.validate()
    ys = _complex_rows(candidates, "candidates")
    a = _complex_vector(accompaniment, "accompaniment")
    v = _complex_vector(vocal, "vocal")
    if ys.shape[1] != a.size or v.size != a.size:
        raise ValueError("candidate and truth cell lengths differ")

    ea = float(np.real(np.vdot(a, a)))
    ev = float(np.real(np.vdot(v, v)))
    basis = np.column_stack((a, v))
    gram = basis.conj().T @ basis
    try:
        condition = float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        condition = math.inf
    identifiable = (
        ea > cfg.accompaniment_floor
        and ev > cfg.vocal_floor
        and math.isfinite(condition)
        and condition <= cfg.max_condition
    )

    if identifiable:
        ridge = cfg.ridge_relative * max(
            float(np.trace(gram).real) / 2.0, _EPS
        )
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(2, dtype=np.complex128),
            basis.conj().T @ ys.T,
        )
        alpha, beta = coefficients[0], coefficients[1]
        residuals = ys - alpha[:, None] * a[None] - beta[:, None] * v[None]
        scale = max(ea, cfg.reference_floor)
        q = (
            cfg.alpha_weight * np.real(np.outer(np.conj(alpha), alpha))
            + cfg.voice_weight * (ev / scale)
            * np.real(np.outer(np.conj(beta), beta))
            + cfg.residual_weight
            * np.real(residuals.conj() @ residuals.T) / scale
        )
        c = cfg.alpha_weight * np.real(alpha)
        constant = cfg.alpha_weight
        mode = "source_coordinates"
    else:
        errors = ys - a[None]
        scale = max(ea + ev, cfg.reference_floor)
        q = cfg.direct_weight * np.real(errors.conj() @ errors.T) / scale
        c = np.zeros(ys.shape[0], dtype=np.float64)
        constant = 0.0
        if ea <= cfg.accompaniment_floor and ev <= cfg.vocal_floor:
            mode = "silent_direct_fallback"
        elif ev <= cfg.vocal_floor:
            mode = "no_vocal_direct_fallback"
        elif ea <= cfg.accompaniment_floor:
            mode = "vocal_only_direct_fallback"
        else:
            mode = "ill_conditioned_direct_fallback"

    result = CellQuadratic(
        Q=_numerically_psd(q),
        c=np.asarray(c, dtype=np.float64),
        constant=float(constant),
        mode=mode,
        accompaniment_energy=ea,
        vocal_energy=ev,
        condition_number=condition if math.isfinite(condition) else None,
    )
    result.validate()
    return result


def stack_cell_grid(cells: Sequence[Sequence[CellQuadratic]]) -> QuadraticGrid:
    if not cells or not cells[0]:
        raise ValueError("at least one time/band cell is required")
    bands = len(cells[0])
    if any(len(row) != bands for row in cells):
        raise ValueError("cell grid must be rectangular")
    candidates = cells[0][0].validate()
    if any(cell.validate() != candidates for row in cells for cell in row):
        raise ValueError("all cells must share one candidate basis")
    grid = QuadraticGrid(
        Q=np.stack([[cell.Q for cell in row] for row in cells]),
        c=np.stack([[cell.c for cell in row] for row in cells]),
        constant=np.asarray(
            [[cell.constant for cell in row] for row in cells], dtype=np.float64
        ),
        modes=np.asarray(
            [[cell.mode for cell in row] for row in cells], dtype=object
        ),
    )
    grid.validate()
    return grid


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex along the final axis."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("simplex input must be finite with a candidate axis")
    k = x.shape[-1]
    flat = x.reshape(-1, k)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cssv = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, k + 1, dtype=np.float64)
    support = ordered - cssv / ranks > 0
    rho = support.sum(axis=1) - 1
    if np.any(rho < 0):
        raise CertifiedRoutingError("simplex projection found no support")
    theta = cssv[np.arange(len(flat)), rho] / (rho + 1)
    result = np.maximum(flat - theta[:, None], 0.0).reshape(x.shape)
    if np.max(np.abs(result.sum(axis=-1) - 1.0)) > 1e-10:
        raise CertifiedRoutingError("simplex projection invariant failed")
    return result


def unary_costs(grid: QuadraticGrid) -> np.ndarray:
    grid.validate()
    return (
        np.diagonal(grid.Q, axis1=-2, axis2=-1)
        - 2.0 * grid.c
        + grid.constant[..., None]
    )


def one_hot_weights(labels: np.ndarray, candidates: int) -> np.ndarray:
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
    t, b, k = grid.validate()
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (t, b, k):
        raise ValueError("weight grid does not match quadratics")
    if np.any(w < -1e-10) or not np.allclose(
        w.sum(axis=-1), 1.0, atol=1e-8, rtol=0
    ):
        raise ValueError("weights must be a per-cell simplex")
    if temporal_smoothness < 0 or frequency_smoothness < 0:
        raise ValueError("smoothness weights must be non-negative")
    cell_cost = (
        np.einsum("tbk,tbkl,tbl->tb", w, grid.Q, w)
        - 2.0 * np.einsum("tbk,tbk->tb", grid.c, w)
        + grid.constant
    )
    count = float(t * b)
    data = float(cell_cost.sum() / count)
    temporal = float(np.square(w[1:] - w[:-1]).sum() / count)
    frequency = float(np.square(w[:, 1:] - w[:, :-1]).sum() / count)
    total = (
        data
        + float(temporal_smoothness) * temporal
        + float(frequency_smoothness) * frequency
    )
    return total, data, temporal, frequency


def best_whole_track(grid: QuadraticGrid) -> tuple[int, np.ndarray]:
    costs = unary_costs(grid).mean(axis=(0, 1))
    return int(np.argmin(costs)), costs


def _edges(t: int, b: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    temporal = [
        (time * b + band, (time + 1) * b + band)
        for time in range(t - 1) for band in range(b)
    ]
    frequency = [
        (time * b + band, time * b + band + 1)
        for time in range(t) for band in range(b - 1)
    ]
    return temporal, frequency


def solve_discrete_global(
    grid: QuadraticGrid,
    config: CertifiedRoutingConfig | None = None,
    *,
    time_limit_seconds: float = 180.0,
    mip_relative_gap: float = 0.0,
) -> DiscreteRoutingResult:
    """O2: globally solve the finite Potts labeling problem through SciPy MILP."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    cfg = config or CertifiedRoutingConfig()
    cfg.validate()
    t, b, k = grid.validate()
    cells = t * b
    temporal, frequency = _edges(t, b)
    edges = [
        (u, v, cfg.temporal_switch_penalty) for u, v in temporal
    ] + [
        (u, v, cfg.frequency_switch_penalty) for u, v in frequency
    ]
    x_count = cells * k
    d_count = len(edges) * k
    variables = x_count + d_count
    objective = np.zeros(variables, dtype=np.float64)
    objective[:x_count] = unary_costs(grid).reshape(-1) / float(cells)
    for edge_index, (_, _, penalty) in enumerate(edges):
        objective[
            x_count + edge_index * k:x_count + (edge_index + 1) * k
        ] = penalty / (2.0 * float(cells))

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for cell in range(cells):
        for candidate in range(k):
            rows.append(row); columns.append(cell * k + candidate); values.append(1.0)
        lower.append(1.0); upper.append(1.0); row += 1
    for edge_index, (left_cell, right_cell, _) in enumerate(edges):
        for candidate in range(k):
            d_index = x_count + edge_index * k + candidate
            for left, right in ((left_cell, right_cell), (right_cell, left_cell)):
                rows.extend((row, row, row))
                columns.extend((left * k + candidate, right * k + candidate, d_index))
                values.extend((1.0, -1.0, -1.0))
                lower.append(-np.inf); upper.append(0.0); row += 1
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(row, variables)
    ).tocsr()
    result = milp(
        objective,
        integrality=np.r_[
            np.ones(x_count, dtype=np.int8),
            np.zeros(d_count, dtype=np.int8),
        ],
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": float(time_limit_seconds),
                 "mip_rel_gap": float(mip_relative_gap)},
    )
    gap = getattr(result, "mip_gap", None)
    gap_value = None if gap is None else float(gap)
    if result.x is None or not result.success:
        raise CertifiedRoutingError(
            f"O2 MILP is not certified: status={result.status}, "
            f"message={result.message}, gap={gap_value}"
        )
    if gap_value is not None and gap_value > mip_relative_gap + 1e-12:
        raise CertifiedRoutingError(
            f"O2 gap {gap_value} exceeds requested {mip_relative_gap}"
        )
    labels = np.argmax(result.x[:x_count].reshape(t, b, k), axis=-1)
    unary = unary_costs(grid)
    chosen = np.take_along_axis(unary, labels[..., None], axis=-1)[..., 0]
    data = float(chosen.mean())
    temporal_switches = int(np.count_nonzero(labels[1:] != labels[:-1]))
    frequency_switches = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    total = (
        data
        + cfg.temporal_switch_penalty * temporal_switches / float(cells)
        + cfg.frequency_switch_penalty * frequency_switches / float(cells)
    )
    return DiscreteRoutingResult(
        labels=labels,
        objective=total,
        data_objective=data,
        temporal_switches=temporal_switches,
        frequency_switches=frequency_switches,
        solver_status=str(result.message),
        mip_gap=gap_value,
    )


def _gradient(weights: np.ndarray, grid: QuadraticGrid,
              config: CertifiedRoutingConfig) -> np.ndarray:
    t, b, _ = grid.validate()
    count = float(t * b)
    gradient = (
        2.0 * np.einsum("tbkl,tbl->tbk", grid.Q, weights) - 2.0 * grid.c
    ) / count
    if config.temporal_weight_smoothness:
        delta = weights[1:] - weights[:-1]
        scale = 2.0 * config.temporal_weight_smoothness / count
        gradient[1:] += scale * delta
        gradient[:-1] -= scale * delta
    if config.frequency_weight_smoothness:
        delta = weights[:, 1:] - weights[:, :-1]
        scale = 2.0 * config.frequency_weight_smoothness / count
        gradient[:, 1:] += scale * delta
        gradient[:, :-1] -= scale * delta
    return gradient


def _lipschitz_bound(grid: QuadraticGrid,
                     config: CertifiedRoutingConfig) -> float:
    t, b, _ = grid.validate()
    largest = max(
        float(np.linalg.eigvalsh(matrix).max(initial=0.0))
        for matrix in grid.Q.reshape(-1, grid.Q.shape[-1], grid.Q.shape[-1])
    )
    count = float(t * b)
    return max(
        (2.0 * largest
         + 8.0 * config.temporal_weight_smoothness
         + 8.0 * config.frequency_weight_smoothness) / count,
        _EPS,
    )


def _initial_weights(name: str, grid: QuadraticGrid, o1_index: int,
                     o2_labels: np.ndarray | None) -> np.ndarray:
    t, b, k = grid.validate()
    if name == "uniform":
        return np.full((t, b, k), 1.0 / k, dtype=np.float64)
    if name == "O1":
        return one_hot_weights(np.full((t, b), o1_index, dtype=np.int64), k)
    if name == "independent":
        return one_hot_weights(np.argmin(unary_costs(grid), axis=-1), k)
    if name == "O2" and o2_labels is not None:
        return one_hot_weights(o2_labels, k)
    raise ValueError(f"unavailable initialization {name!r}")


def solve_convex_certified(
    grid: QuadraticGrid,
    config: CertifiedRoutingConfig | None = None,
    *,
    o1_index: int | None = None,
    o2_labels: np.ndarray | None = None,
    starts: Iterable[str] = ("uniform", "O1", "independent", "O2"),
) -> ConvexRoutingResult:
    """O3: solve the convex simplex problem with a convergence certificate."""

    cfg = config or CertifiedRoutingConfig()
    cfg.validate()
    if o1_index is None:
        o1_index, _ = best_whole_track(grid)
    step = 1.0 / _lipschitz_bound(grid, cfg)
    solutions = []
    start_objectives: dict[str, float] = {}
    for name in starts:
        if name == "O2" and o2_labels is None:
            continue
        weights = _initial_weights(name, grid, o1_index, o2_labels)
        current = route_objective(
            weights, grid,
            temporal_smoothness=cfg.temporal_weight_smoothness,
            frequency_smoothness=cfg.frequency_weight_smoothness,
        )[0]
        start_objectives[name] = current
        converged = False
        projected_norm = math.inf
        for iteration in range(1, cfg.max_projected_gradient_iterations + 1):
            gradient = _gradient(weights, grid, cfg)
            proposal = project_simplex(weights - step * gradient)
            projected_norm = float(np.linalg.norm((weights - proposal) / step))
            proposal_value = route_objective(
                proposal, grid,
                temporal_smoothness=cfg.temporal_weight_smoothness,
                frequency_smoothness=cfg.frequency_weight_smoothness,
            )[0]
            if proposal_value > current + 1e-9:
                raise CertifiedRoutingError(
                    f"O3 objective increased {current} -> {proposal_value}"
                )
            weights, current = proposal, proposal_value
            if projected_norm <= cfg.projected_gradient_tolerance:
                converged = True
                break
        solutions.append(
            (current, name, weights.copy(), iteration, projected_norm, converged)
        )

    certified = [solution for solution in solutions if solution[-1]]
    if len(certified) != len(solutions):
        details = {
            name: {"objective": value, "iterations": iterations,
                   "projected_gradient_norm": norm,
                   "converged": converged}
            for value, name, _, iterations, norm, converged in solutions
        }
        raise CertifiedRoutingError(
            f"O3 did not converge from every declared start: {details}"
        )
    values = [solution[0] for solution in certified]
    if max(values) - min(values) > max(
        1e-8, 10.0 * cfg.projected_gradient_tolerance
    ):
        raise CertifiedRoutingError(
            f"convex starts disagree after convergence: {values}"
        )
    value, name, weights, iterations, norm, converged = min(
        certified, key=lambda solution: solution[0]
    )
    total, data, temporal, frequency = route_objective(
        weights, grid,
        temporal_smoothness=cfg.temporal_weight_smoothness,
        frequency_smoothness=cfg.frequency_weight_smoothness,
    )
    return ConvexRoutingResult(
        weights=weights,
        objective=total,
        data_objective=data,
        temporal_smoothness=temporal,
        frequency_smoothness=frequency,
        iterations=iterations,
        converged=converged,
        projected_gradient_norm=norm,
        selected_start=name,
        start_objectives=start_objectives,
    )
