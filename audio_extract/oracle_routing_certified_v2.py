"""Certified weighted exact-reference routing mathematics for opera separation.

This module is a version-separated research core. It is not wired into the
active full-work runner. It provides automatically checkable mathematics for a
binding O1/O2/O3 oracle-envelope experiment:

* every exact time/frequency cell has one finite convex cost and a positive,
  explicit aggregation weight;
* identifiable cells use a convex source-coordinate quadratic;
* A-only, V-only, silent, and ill-conditioned cells use exact direct-target
  error instead of becoming zero-cost omissions;
* O2 is a globally certified Potts-labeling MILP;
* O3 is a convex simplex problem with weighted squared graph smoothness and a
  projected-gradient/KKT convergence certificate;
* scaling all aggregation weights by one positive constant is invariant.

The functions consume clean accompaniment and target-voice references. They are
therefore an oracle research diagnostic, not a deployable separator or
reference-free judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

import numpy as np

_TINY = np.finfo(np.float64).tiny


class CertifiedRoutingError(RuntimeError):
    """An exact routing input, optimization, or certificate is invalid."""


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
    psd_relative_tolerance: float = 1e-10
    temporal_switch_penalty: float = 0.05
    frequency_switch_penalty: float = 0.05
    temporal_weight_smoothness: float = 0.05
    frequency_weight_smoothness: float = 0.05
    projected_gradient_tolerance: float = 1e-7
    max_projected_gradient_iterations: int = 20_000

    def validate(self) -> None:
        nonnegative = (
            "alpha_weight", "voice_weight", "residual_weight", "direct_weight",
            "ridge_relative", "accompaniment_floor", "vocal_floor",
            "temporal_switch_penalty", "frequency_switch_penalty",
            "temporal_weight_smoothness", "frequency_weight_smoothness",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        positive = (
            "reference_floor", "psd_relative_tolerance",
            "projected_gradient_tolerance",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.max_condition)) or self.max_condition <= 1.0:
            raise ValueError("max_condition must be finite and greater than one")
        if int(self.max_projected_gradient_iterations) < 1:
            raise ValueError("max_projected_gradient_iterations must be positive")


def _matrix_scale(matrix: np.ndarray) -> float:
    q = np.asarray(matrix, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("PSD matrix must be square")
    if not np.all(np.isfinite(q)):
        raise ValueError("PSD matrix must be finite")
    return max(
        _TINY,
        float(np.max(np.abs(q), initial=0.0)),
        float(np.linalg.norm(q, ord=2)),
        abs(float(np.trace(q))),
    )


def stabilize_psd(
    matrix: np.ndarray, *, relative_tolerance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Symmetrize and repair only scale-relative numerical negativity."""
    relative_tolerance = float(relative_tolerance)
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    q = np.asarray(matrix, dtype=np.float64)
    q = 0.5 * (q + q.T)
    scale = _matrix_scale(q)
    tolerance = relative_tolerance * scale
    minimum = float(np.linalg.eigvalsh(q).min(initial=0.0))
    if minimum < -tolerance:
        raise CertifiedRoutingError(
            "derived quadratic is materially non-PSD: "
            f"min_eigenvalue={minimum}, tolerance={tolerance}, scale={scale}"
        )
    shift = 0.0
    if minimum < 0.0:
        shift = -minimum + tolerance
        q = q + np.eye(q.shape[0], dtype=np.float64) * shift
    return q, {
        "matrix_scale": scale,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": tolerance,
        "minimum_eigenvalue_before": minimum,
        "diagonal_shift": shift,
        "minimum_eigenvalue_after": float(np.linalg.eigvalsh(q).min(initial=0.0)),
    }


def _validate_psd(matrix: np.ndarray, *, relative_tolerance: float) -> None:
    q = np.asarray(matrix, dtype=np.float64)
    if not np.allclose(q, q.T, atol=0.0, rtol=1e-12):
        raise ValueError("Q must be symmetric")
    scale = _matrix_scale(q)
    tolerance = float(relative_tolerance) * scale
    minimum = float(np.linalg.eigvalsh(q).min(initial=0.0))
    if minimum < -tolerance:
        raise ValueError(
            f"Q is not PSD: min_eigenvalue={minimum}, tolerance={tolerance}"
        )


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
    psd_relative_tolerance: float
    psd_diagonal_shift: float = 0.0

    def validate(self) -> int:
        q = np.asarray(self.Q, dtype=np.float64)
        c = np.asarray(self.c, dtype=np.float64)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("Q must be square")
        if c.shape != (q.shape[0],):
            raise ValueError("c must match Q")
        if not (
            np.all(np.isfinite(q)) and np.all(np.isfinite(c))
            and math.isfinite(float(self.constant))
            and math.isfinite(float(self.accompaniment_energy))
            and math.isfinite(float(self.vocal_energy))
            and math.isfinite(float(self.psd_relative_tolerance))
            and math.isfinite(float(self.psd_diagonal_shift))
        ):
            raise ValueError("cell facts must be finite")
        if self.accompaniment_energy < 0.0 or self.vocal_energy < 0.0:
            raise ValueError("source energies must be non-negative")
        if self.psd_relative_tolerance <= 0.0 or self.psd_diagonal_shift < 0.0:
            raise ValueError("PSD tolerance/shift are invalid")
        if self.condition_number is not None and (
            not math.isfinite(float(self.condition_number))
            or self.condition_number < 1.0
        ):
            raise ValueError("condition_number must be None or finite >= 1")
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("every cell must have a non-empty mode")
        _validate_psd(q, relative_tolerance=float(self.psd_relative_tolerance))
        return q.shape[0]


@dataclass(frozen=True)
class QuadraticGrid:
    """A complete weighted time/band grid of exact convex costs."""
    Q: np.ndarray
    c: np.ndarray
    constant: np.ndarray
    modes: np.ndarray
    cell_weights: np.ndarray
    psd_relative_tolerance: float

    def validate(self) -> tuple[int, int, int]:
        q = np.asarray(self.Q, dtype=np.float64)
        c = np.asarray(self.c, dtype=np.float64)
        constant = np.asarray(self.constant, dtype=np.float64)
        modes = np.asarray(self.modes, dtype=object)
        weights = np.asarray(self.cell_weights, dtype=np.float64)
        if q.ndim != 4 or q.shape[-1] != q.shape[-2]:
            raise ValueError("Q must be (time, band, K, K)")
        if c.shape != q.shape[:-1] or constant.shape != q.shape[:2]:
            raise ValueError("c/constant shapes do not match Q")
        if modes.shape != q.shape[:2] or weights.shape != q.shape[:2]:
            raise ValueError("mode/weight grids do not match Q")
        if not (
            np.all(np.isfinite(q)) and np.all(np.isfinite(c))
            and np.all(np.isfinite(constant)) and np.all(np.isfinite(weights))
        ):
            raise ValueError("grid coefficients/weights must be finite")
        if np.any(weights <= 0.0):
            raise ValueError("every cell weight must be strictly positive")
        if any(not isinstance(value, str) or not value.strip()
               for value in modes.reshape(-1)):
            raise ValueError("every cell must declare exactly one non-empty mode")
        if not math.isfinite(float(self.psd_relative_tolerance)) or (
            self.psd_relative_tolerance <= 0.0
        ):
            raise ValueError("grid PSD tolerance must be finite and positive")
        if not np.allclose(q, np.swapaxes(q, -1, -2), atol=0.0, rtol=1e-12):
            raise ValueError("Q grid is not symmetric")
        for matrix in q.reshape(-1, q.shape[-1], q.shape[-1]):
            _validate_psd(
                matrix, relative_tolerance=float(self.psd_relative_tolerance)
            )
        return q.shape[0], q.shape[1], q.shape[2]

    @property
    def total_cell_weight(self) -> float:
        self.validate()
        return float(np.asarray(self.cell_weights, dtype=np.float64).sum())


@dataclass(frozen=True)
class DiscreteRoutingResult:
    labels: np.ndarray
    objective: float
    data_objective: float
    temporal_switches: int
    frequency_switches: int
    weighted_temporal_switches: float
    weighted_frequency_switches: float
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


def cell_measure_weights(
    time_lengths: Sequence[float], frequency_widths: Sequence[float],
) -> np.ndarray:
    """Return positive coefficient-measure weights for rectangular cells."""
    time = np.asarray(time_lengths, dtype=np.float64)
    frequency = np.asarray(frequency_widths, dtype=np.float64)
    if (
        time.ndim != 1 or frequency.ndim != 1
        or time.size < 1 or frequency.size < 1
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(frequency))
        or np.any(time <= 0.0) or np.any(frequency <= 0.0)
    ):
        raise ValueError(
            "time_lengths and frequency_widths must be finite positive vectors"
        )
    return time[:, None] * frequency[None, :]


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


def build_cell_quadratic(
    candidates: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    config: CertifiedRoutingConfig | None = None,
) -> CellQuadratic:
    """Build one exact convex routing cost, with direct exact fallbacks."""
    cfg = config or CertifiedRoutingConfig()
    cfg.validate()
    ys = _complex_rows(candidates, "candidates")
    a = _complex_vector(accompaniment, "accompaniment")
    v = _complex_vector(vocal, "vocal")
    if ys.shape[1] != a.size or v.size != a.size:
        raise ValueError("candidate and truth cell lengths differ")
    accompaniment_energy = float(np.real(np.vdot(a, a)))
    vocal_energy = float(np.real(np.vdot(v, v)))
    basis = np.column_stack((a, v))
    gram = basis.conj().T @ basis
    try:
        condition = float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        condition = math.inf
    identifiable = (
        accompaniment_energy > cfg.accompaniment_floor
        and vocal_energy > cfg.vocal_floor
        and math.isfinite(condition) and condition <= cfg.max_condition
    )
    if identifiable:
        ridge = cfg.ridge_relative * max(
            float(np.trace(gram).real) / 2.0, _TINY
        )
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(2, dtype=np.complex128),
            basis.conj().T @ ys.T,
        )
        alpha, beta = coefficients[0], coefficients[1]
        residuals = ys - alpha[:, None] * a[None] - beta[:, None] * v[None]
        scale = max(accompaniment_energy, cfg.reference_floor)
        raw_q = (
            cfg.alpha_weight * np.real(np.outer(np.conj(alpha), alpha))
            + cfg.voice_weight * (vocal_energy / scale)
            * np.real(np.outer(np.conj(beta), beta))
            + cfg.residual_weight
            * np.real(residuals.conj() @ residuals.T) / scale
        )
        c = cfg.alpha_weight * np.real(alpha)
        constant = cfg.alpha_weight
        mode = "source_coordinates"
    else:
        errors = ys - a[None]
        scale = max(accompaniment_energy + vocal_energy, cfg.reference_floor)
        raw_q = cfg.direct_weight * np.real(errors.conj() @ errors.T) / scale
        c = np.zeros(ys.shape[0], dtype=np.float64)
        constant = 0.0
        if (
            accompaniment_energy <= cfg.accompaniment_floor
            and vocal_energy <= cfg.vocal_floor
        ):
            mode = "silent_direct_fallback"
        elif vocal_energy <= cfg.vocal_floor:
            mode = "no_vocal_direct_fallback"
        elif accompaniment_energy <= cfg.accompaniment_floor:
            mode = "vocal_only_direct_fallback"
        else:
            mode = "ill_conditioned_direct_fallback"
    q, psd = stabilize_psd(
        raw_q, relative_tolerance=cfg.psd_relative_tolerance
    )
    cell = CellQuadratic(
        Q=q, c=np.asarray(c, dtype=np.float64), constant=float(constant),
        mode=mode, accompaniment_energy=accompaniment_energy,
        vocal_energy=vocal_energy,
        condition_number=condition if math.isfinite(condition) else None,
        psd_relative_tolerance=cfg.psd_relative_tolerance,
        psd_diagonal_shift=psd["diagonal_shift"],
    )
    cell.validate()
    return cell


def stack_cell_grid(
    cells: Sequence[Sequence[CellQuadratic]],
    *, cell_weights: np.ndarray | None = None,
) -> QuadraticGrid:
    if not cells or not cells[0]:
        raise ValueError("at least one time/band cell is required")
    bands = len(cells[0])
    if any(len(row) != bands for row in cells):
        raise ValueError("cell grid must be rectangular")
    candidates = cells[0][0].validate()
    tolerance = float(cells[0][0].psd_relative_tolerance)
    for row in cells:
        for cell in row:
            if cell.validate() != candidates:
                raise ValueError("all cells must share one candidate basis")
            if not math.isclose(
                float(cell.psd_relative_tolerance), tolerance,
                rel_tol=0.0, abs_tol=0.0,
            ):
                raise ValueError("all cells must share one PSD tolerance")
    shape = (len(cells), bands)
    weight_array = (
        np.ones(shape, dtype=np.float64)
        if cell_weights is None
        else np.asarray(cell_weights, dtype=np.float64)
    )
    grid = QuadraticGrid(
        Q=np.stack([[cell.Q for cell in row] for row in cells]),
        c=np.stack([[cell.c for cell in row] for row in cells]),
        constant=np.asarray(
            [[cell.constant for cell in row] for row in cells], dtype=np.float64
        ),
        modes=np.asarray([[cell.mode for cell in row] for row in cells], dtype=object),
        cell_weights=weight_array,
        psd_relative_tolerance=tolerance,
    )
    grid.validate()
    return grid


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex along the final axis."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("simplex input must be finite with a candidate axis")
    candidates = x.shape[-1]
    flat = x.reshape(-1, candidates)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cssv = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, candidates + 1, dtype=np.float64)
    support = ordered - cssv / ranks > 0.0
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
        - 2.0 * grid.c + grid.constant[..., None]
    )


def one_hot_weights(labels: np.ndarray, candidates: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2 or np.any(labels < 0) or np.any(labels >= candidates):
        raise ValueError("invalid routing labels")
    result = np.zeros(labels.shape + (candidates,), dtype=np.float64)
    np.put_along_axis(result, labels[..., None], 1.0, axis=-1)
    return result


def _temporal_edge_weights(grid: QuadraticGrid) -> np.ndarray:
    return 0.5 * (grid.cell_weights[1:] + grid.cell_weights[:-1])


def _frequency_edge_weights(grid: QuadraticGrid) -> np.ndarray:
    return 0.5 * (grid.cell_weights[:, 1:] + grid.cell_weights[:, :-1])


def route_objective(
    weights: np.ndarray, grid: QuadraticGrid,
    *, temporal_smoothness: float = 0.0, frequency_smoothness: float = 0.0,
) -> tuple[float, float, float, float]:
    """Evaluate the complete weighted convex route objective."""
    time, bands, candidates = grid.validate()
    route = np.asarray(weights, dtype=np.float64)
    if route.shape != (time, bands, candidates):
        raise ValueError("weight grid does not match quadratics")
    if np.any(route < -1e-10) or not np.allclose(
        route.sum(axis=-1), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("weights must be a per-cell simplex")
    if temporal_smoothness < 0.0 or frequency_smoothness < 0.0:
        raise ValueError("smoothness weights must be non-negative")
    cell_cost = (
        np.einsum("tbk,tbkl,tbl->tb", route, grid.Q, route)
        - 2.0 * np.einsum("tbk,tbk->tb", grid.c, route)
        + grid.constant
    )
    total_weight = grid.total_cell_weight
    data = float(np.sum(grid.cell_weights * cell_cost) / total_weight)
    temporal = 0.0
    if time > 1:
        temporal = float(
            np.sum(
                _temporal_edge_weights(grid)[..., None]
                * np.square(route[1:] - route[:-1])
            ) / total_weight
        )
    frequency = 0.0
    if bands > 1:
        frequency = float(
            np.sum(
                _frequency_edge_weights(grid)[..., None]
                * np.square(route[:, 1:] - route[:, :-1])
            ) / total_weight
        )
    return (
        data + float(temporal_smoothness) * temporal
        + float(frequency_smoothness) * frequency,
        data, temporal, frequency,
    )


def best_whole_track(grid: QuadraticGrid) -> tuple[int, np.ndarray]:
    costs = np.sum(
        grid.cell_weights[..., None] * unary_costs(grid), axis=(0, 1)
    ) / grid.total_cell_weight
    return int(np.argmin(costs)), costs


def _edges(time: int, bands: int, grid: QuadraticGrid):
    temporal_weights = _temporal_edge_weights(grid)
    frequency_weights = _frequency_edge_weights(grid)
    temporal = [
        (t * bands + b, (t + 1) * bands + b, float(temporal_weights[t, b]))
        for t in range(time - 1) for b in range(bands)
    ]
    frequency = [
        (t * bands + b, t * bands + b + 1, float(frequency_weights[t, b]))
        for t in range(time) for b in range(bands - 1)
    ]
    return temporal, frequency


def _discrete_objective(labels, grid, config):
    time, bands, candidates = grid.validate()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (time, bands) or np.any((labels < 0) | (labels >= candidates)):
        raise ValueError("invalid O2 labels")
    chosen = np.take_along_axis(
        unary_costs(grid), labels[..., None], axis=-1
    )[..., 0]
    total_weight = grid.total_cell_weight
    data = float(np.sum(grid.cell_weights * chosen) / total_weight)
    temporal_count = int(np.count_nonzero(labels[1:] != labels[:-1]))
    frequency_count = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    temporal_weighted = float(np.sum(
        _temporal_edge_weights(grid) * (labels[1:] != labels[:-1])
    )) if time > 1 else 0.0
    frequency_weighted = float(np.sum(
        _frequency_edge_weights(grid) * (labels[:, 1:] != labels[:, :-1])
    )) if bands > 1 else 0.0
    total = (
        data + config.temporal_switch_penalty * temporal_weighted / total_weight
        + config.frequency_switch_penalty * frequency_weighted / total_weight
    )
    return total, data, temporal_count, frequency_count, temporal_weighted, frequency_weighted


def solve_discrete_global(
    grid: QuadraticGrid, config: CertifiedRoutingConfig | None = None,
    *, time_limit_seconds: float = 180.0, mip_relative_gap: float = 0.0,
) -> DiscreteRoutingResult:
    """O2: globally solve the weighted Potts labeling problem by MILP."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    cfg = config or CertifiedRoutingConfig()
    cfg.validate()
    for name, value in (
        ("time_limit_seconds", time_limit_seconds),
        ("mip_relative_gap", mip_relative_gap),
    ):
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    time, bands, candidates = grid.validate()
    cells = time * bands
    temporal, frequency = _edges(time, bands, grid)
    edges = [
        (u, v, weight, cfg.temporal_switch_penalty)
        for u, v, weight in temporal
    ] + [
        (u, v, weight, cfg.frequency_switch_penalty)
        for u, v, weight in frequency
    ]
    x_count = cells * candidates
    d_count = len(edges) * candidates
    variables = x_count + d_count
    total_weight = grid.total_cell_weight
    objective = np.zeros(variables, dtype=np.float64)
    objective[:x_count] = (
        grid.cell_weights[..., None] * unary_costs(grid)
    ).reshape(-1) / total_weight
    for edge_index, (_, _, edge_weight, penalty) in enumerate(edges):
        objective[
            x_count + edge_index * candidates:
            x_count + (edge_index + 1) * candidates
        ] = penalty * edge_weight / (2.0 * total_weight)
    rows, columns, values, lower, upper = [], [], [], [], []
    row = 0
    for cell in range(cells):
        for candidate in range(candidates):
            rows.append(row); columns.append(cell * candidates + candidate); values.append(1.0)
        lower.append(1.0); upper.append(1.0); row += 1
    for edge_index, (left_cell, right_cell, _, _) in enumerate(edges):
        for candidate in range(candidates):
            d_index = x_count + edge_index * candidates + candidate
            for left, right in ((left_cell, right_cell), (right_cell, left_cell)):
                rows.extend((row, row, row))
                columns.extend((left * candidates + candidate,
                                right * candidates + candidate, d_index))
                values.extend((1.0, -1.0, -1.0))
                lower.append(-np.inf); upper.append(0.0); row += 1
    matrix = coo_matrix((values, (rows, columns)), shape=(row, variables)).tocsr()
    result = milp(
        objective,
        integrality=np.r_[np.ones(x_count, dtype=np.int8),
                           np.zeros(d_count, dtype=np.int8)],
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
    if gap_value is not None and gap_value > float(mip_relative_gap) + 1e-12:
        raise CertifiedRoutingError(
            f"O2 gap {gap_value} exceeds requested {mip_relative_gap}"
        )
    labels = np.argmax(result.x[:x_count].reshape(time, bands, candidates), axis=-1)
    total, data, tc, fc, tw, fw = _discrete_objective(labels, grid, cfg)
    return DiscreteRoutingResult(
        labels=labels, objective=total, data_objective=data,
        temporal_switches=tc, frequency_switches=fc,
        weighted_temporal_switches=tw, weighted_frequency_switches=fw,
        solver_status=str(result.message), mip_gap=gap_value,
    )


def _gradient(route, grid, config):
    total_weight = grid.total_cell_weight
    gradient = (
        2.0 * grid.cell_weights[..., None]
        * (np.einsum("tbkl,tbl->tbk", grid.Q, route) - grid.c)
        / total_weight
    )
    if route.shape[0] > 1 and config.temporal_weight_smoothness:
        delta = route[1:] - route[:-1]
        scale = (
            2.0 * config.temporal_weight_smoothness
            * _temporal_edge_weights(grid)[..., None] / total_weight
        )
        gradient[1:] += scale * delta
        gradient[:-1] -= scale * delta
    if route.shape[1] > 1 and config.frequency_weight_smoothness:
        delta = route[:, 1:] - route[:, :-1]
        scale = (
            2.0 * config.frequency_weight_smoothness
            * _frequency_edge_weights(grid)[..., None] / total_weight
        )
        gradient[:, 1:] += scale * delta
        gradient[:, :-1] -= scale * delta
    return gradient


def _lipschitz_bound(grid, config):
    time, bands, _ = grid.validate()
    total_weight = grid.total_cell_weight
    data_bound = max(
        2.0 * float(grid.cell_weights[t, b])
        * float(np.linalg.eigvalsh(grid.Q[t, b]).max(initial=0.0))
        / total_weight
        for t in range(time) for b in range(bands)
    )
    weighted_degree = np.zeros((time, bands), dtype=np.float64)
    if time > 1 and config.temporal_weight_smoothness:
        edge = config.temporal_weight_smoothness * _temporal_edge_weights(grid)
        weighted_degree[1:] += edge; weighted_degree[:-1] += edge
    if bands > 1 and config.frequency_weight_smoothness:
        edge = config.frequency_weight_smoothness * _frequency_edge_weights(grid)
        weighted_degree[:, 1:] += edge; weighted_degree[:, :-1] += edge
    smoothness_bound = 4.0 * float(weighted_degree.max(initial=0.0)) / total_weight
    return max(data_bound + smoothness_bound, _TINY)


def _initial_weights(name, grid, o1_index, o2_labels):
    time, bands, candidates = grid.validate()
    if name == "uniform":
        return np.full((time, bands, candidates), 1.0 / candidates)
    if name == "O1":
        return one_hot_weights(np.full((time, bands), o1_index, dtype=np.int64), candidates)
    if name == "independent":
        return one_hot_weights(np.argmin(unary_costs(grid), axis=-1), candidates)
    if name == "O2" and o2_labels is not None:
        return one_hot_weights(o2_labels, candidates)
    raise ValueError(f"unavailable initialization {name!r}")


def solve_convex_certified(
    grid: QuadraticGrid, config: CertifiedRoutingConfig | None = None,
    *, o1_index: int | None = None, o2_labels: np.ndarray | None = None,
    starts: Iterable[str] = ("uniform", "O1", "independent", "O2"),
) -> ConvexRoutingResult:
    """O3: solve the convex weighted simplex problem with a KKT certificate."""
    cfg = config or CertifiedRoutingConfig()
    cfg.validate()
    if o1_index is None:
        o1_index, _ = best_whole_track(grid)
    step = 1.0 / _lipschitz_bound(grid, cfg)
    solutions = []
    start_objectives = {}
    for name in starts:
        if name == "O2" and o2_labels is None:
            continue
        route = _initial_weights(name, grid, o1_index, o2_labels)
        current = route_objective(
            route, grid,
            temporal_smoothness=cfg.temporal_weight_smoothness,
            frequency_smoothness=cfg.frequency_weight_smoothness,
        )[0]
        start_objectives[name] = current
        converged = False
        projected_norm = math.inf
        iteration = 0
        for iteration in range(1, cfg.max_projected_gradient_iterations + 1):
            gradient = _gradient(route, grid, cfg)
            proposal = project_simplex(route - step * gradient)
            projected_norm = float(np.linalg.norm((route - proposal) / step))
            proposal_value = route_objective(
                proposal, grid,
                temporal_smoothness=cfg.temporal_weight_smoothness,
                frequency_smoothness=cfg.frequency_weight_smoothness,
            )[0]
            tolerance = 1e-12 * max(1.0, abs(current), abs(proposal_value))
            if proposal_value > current + tolerance:
                raise CertifiedRoutingError(
                    f"O3 objective increased {current} -> {proposal_value}"
                )
            route, current = proposal, proposal_value
            if projected_norm <= cfg.projected_gradient_tolerance:
                converged = True
                break
        solutions.append((current, name, route.copy(), iteration,
                          projected_norm, converged))
    certified = [solution for solution in solutions if solution[-1]]
    if not certified:
        details = {
            name: {"objective": value, "iterations": iterations,
                   "projected_gradient_norm": norm}
            for value, name, _, iterations, norm, _ in solutions
        }
        raise CertifiedRoutingError(f"O3 did not converge: {details}")
    values = [solution[0] for solution in certified]
    agreement = max(1e-8, 10.0 * cfg.projected_gradient_tolerance)
    if max(values) - min(values) > agreement:
        raise CertifiedRoutingError(
            f"convex starts disagree after convergence: {values}"
        )
    _, selected_start, route, iterations, norm, converged = min(
        certified, key=lambda solution: solution[0]
    )
    total, data, temporal, frequency = route_objective(
        route, grid,
        temporal_smoothness=cfg.temporal_weight_smoothness,
        frequency_smoothness=cfg.frequency_weight_smoothness,
    )
    return ConvexRoutingResult(
        weights=route, objective=total, data_objective=data,
        temporal_smoothness=temporal, frequency_smoothness=frequency,
        iterations=iterations, converged=converged,
        projected_gradient_norm=norm, selected_start=selected_start,
        start_objectives=start_objectives,
    )
