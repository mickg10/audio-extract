"""Unsmoothed exact routing envelopes and shared discrete costs.

A smooth mean-optimal route can conceal the product's rare worst-event failures.
This module supplies two diagnostic upper bounds on a fixed candidate basis:

``O0D``
    Independent best real candidate in each identifiable time/frequency cell.

``O0C``
    Independent exact convex-hull optimum in each identifiable cell, solved by
    exhaustive active-set KKT enumeration. With the preregistered five-member
    basis this is only 31 supports per cell and avoids iterative/convergence
    ambiguity.

Unavailable cells are filled with the exact O1 label for deterministic audio
reconstruction but are excluded from the data objective. O0 outputs are
``diagnostic_only`` and are never production candidates.

The same PSD quadratic defines the corrected whole-candidate unary used by O1
and the exact O2 Potts MILP:

    vertex_unary = diag(G) - 2*c + constant.

The cell solver is invariant to a positive rescaling of ``G`` and ``c``. Every
cell is equilibrated before KKT certification, and the equality constraint is
handled through a sum-zero nullspace rather than an ill-conditioned augmented
KKT matrix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from .oracle_convex import ConvexOracleConfig, ConvexQuadratic, build_quadratic
from .oracle_routing import CellStatistics


class TailOracleError(RuntimeError):
    """The independent-cell oracle problem or certificate is invalid."""


@dataclass(frozen=True)
class ActiveSetConfig:
    feasibility_tolerance: float = 1e-10
    kkt_tolerance: float = 1e-9
    objective_tolerance: float = 1e-12
    strict_support_tolerance: float = 1e-10

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_dict(self) -> dict[str, Any]:
        return {name: str(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class IndependentDiscreteResult:
    labels: np.ndarray
    weights: np.ndarray
    data_objective: float
    available_cells: int
    unavailable_cells: int
    occupancy: tuple[int, ...]


@dataclass(frozen=True)
class IndependentConvexResult:
    weights: np.ndarray
    data_objective: float
    available_cells: int
    unavailable_cells: int
    active_set_size_histogram: dict[int, int]
    interpolated_cells: int
    interpolated_fraction: float
    max_simplex_error: float
    max_normalized_stationarity_residual: float
    min_normalized_inactive_reduced_gradient: float | None
    mean_weights: tuple[float, ...]
    support_masks: np.ndarray
    cell_normalization_scale_min: float
    cell_normalization_scale_max: float


def corrected_vertex_unary(quadratic: ConvexQuadratic) -> np.ndarray:
    """Return each candidate vertex cost in the shared quadratic objective."""
    diagonal = np.diagonal(quadratic.gram, axis1=-2, axis2=-1)
    unary = diagonal - 2.0 * quadratic.linear + quadratic.constant[..., None]
    if unary.shape != quadratic.linear.shape or not np.all(np.isfinite(unary)):
        raise TailOracleError("corrected vertex unary is invalid")
    return unary


def corrected_quadratic(stats: CellStatistics,
                        config: ConvexOracleConfig | None = None
                        ) -> tuple[ConvexQuadratic, np.ndarray]:
    cfg = config or ConvexOracleConfig()
    quadratic = build_quadratic(stats, cfg)
    return quadratic, corrected_vertex_unary(quadratic)


def _one_hot(labels: np.ndarray, candidates: int) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64)
    if lab.ndim != 2 or np.any(lab < 0) or np.any(lab >= candidates):
        raise ValueError("labels must be a valid (time,band) candidate map")
    weights = np.zeros(lab.shape + (candidates,), dtype=np.float64)
    np.put_along_axis(weights, lab[..., None], 1.0, axis=-1)
    return weights


def _validate_fallback(fallback_index: int, candidates: int) -> int:
    index = int(fallback_index)
    if not 0 <= index < candidates:
        raise ValueError("fallback/O1 index is outside candidate basis")
    return index


def solve_independent_discrete(quadratic: ConvexQuadratic,
                               fallback_index: int) -> IndependentDiscreteResult:
    """Solve O0D exactly by independent vertex minimization."""
    unary = corrected_vertex_unary(quadratic)
    q, b, k = unary.shape
    fallback = _validate_fallback(fallback_index, k)
    labels = np.full((q, b), fallback, dtype=np.int32)
    labels[quadratic.available] = np.argmin(
        unary[quadratic.available], axis=-1
    ).astype(np.int32)
    values = unary[
        np.arange(q)[:, None], np.arange(b)[None, :], labels
    ]
    data = float(values[quadratic.available].sum() / quadratic.denominator)
    occupancy = tuple(int(np.count_nonzero(labels == index)) for index in range(k))
    return IndependentDiscreteResult(
        labels=labels,
        weights=_one_hot(labels, k),
        data_objective=data,
        available_cells=quadratic.denominator,
        unavailable_cells=int(quadratic.available.size - quadratic.denominator),
        occupancy=occupancy,
    )


def _supports(candidates: int) -> Iterable[tuple[int, ...]]:
    for size in range(1, candidates + 1):
        yield from combinations(range(candidates), size)


def _cell_value(weight: np.ndarray, gram: np.ndarray,
                linear: np.ndarray, constant: float) -> float:
    return float(weight @ gram @ weight - 2.0 * linear @ weight + constant)


def _cell_variable_value(weight: np.ndarray, gram: np.ndarray,
                         linear: np.ndarray) -> float:
    return float(weight @ gram @ weight - 2.0 * linear @ weight)


def _nullspace_basis(size: int) -> np.ndarray:
    """Orthonormal basis for vectors whose coordinates sum to zero."""
    if size <= 1:
        return np.zeros((size, 0), dtype=np.float64)
    raw = np.zeros((size, size - 1), dtype=np.float64)
    raw[np.arange(size - 1), np.arange(size - 1)] = 1.0
    raw[-1, :] = -1.0
    basis, _ = np.linalg.qr(raw, mode="reduced")
    if np.max(np.abs(np.ones(size) @ basis)) > 1e-12:
        raise TailOracleError("failed to construct sum-zero nullspace")
    return basis


def _cell_scale(gram: np.ndarray, linear: np.ndarray) -> float:
    raw = max(
        float(np.max(np.abs(gram), initial=0.0)),
        float(np.max(np.abs(linear), initial=0.0)),
    )
    return raw if raw > 0.0 else 1.0


def _solve_support(
    normalized_gram: np.ndarray,
    normalized_linear: np.ndarray,
    support: tuple[int, ...],
    config: ActiveSetConfig,
) -> tuple[np.ndarray, float, float, float] | None:
    """Return a scale-normalized KKT-certified support solution, or ``None``.

    The equality constraint is built into ``w = 1/m + Zz``, avoiding the severe
    conditioning imbalance of an augmented ``[G,1;1',0]`` system.
    """
    k = len(normalized_linear)
    indices = np.asarray(support, dtype=np.int64)
    gs = normalized_gram[np.ix_(indices, indices)]
    cs = normalized_linear[indices]
    m = len(indices)
    base = np.full(m, 1.0 / m, dtype=np.float64)
    basis = _nullspace_basis(m)
    if basis.shape[1]:
        reduced_gram = basis.T @ gs @ basis
        reduced_rhs = basis.T @ (cs - gs @ base)
        rcond = np.finfo(np.float64).eps * max(reduced_gram.shape) * 16.0
        coordinate, _, _, _ = np.linalg.lstsq(
            reduced_gram, reduced_rhs, rcond=rcond
        )
        reduced_residual = reduced_gram @ coordinate - reduced_rhs
        residual_scale = max(1.0, float(np.max(np.abs(reduced_rhs), initial=0.0)))
        if float(np.max(np.abs(reduced_residual), initial=0.0)) > (
            config.kkt_tolerance * residual_scale
        ):
            return None
        active = base + basis @ coordinate
    else:
        active = base

    simplex_error = abs(float(active.sum()) - 1.0)
    if simplex_error > config.feasibility_tolerance:
        return None
    if m > 1 and float(active.min()) <= config.strict_support_tolerance:
        # Boundary solutions are represented and certified on a smaller support.
        return None
    if float(active.min()) < -config.feasibility_tolerance:
        return None

    weight = np.zeros(k, dtype=np.float64)
    weight[indices] = active
    multiplier = float(np.mean(cs - gs @ active))
    reduced = normalized_gram @ weight - normalized_linear + multiplier
    stationarity = float(np.max(np.abs(reduced[indices]), initial=0.0))
    inactive = np.ones(k, dtype=bool)
    inactive[indices] = False
    minimum_inactive = (
        float(reduced[inactive].min()) if np.any(inactive) else float("inf")
    )
    if stationarity > config.kkt_tolerance:
        return None
    if minimum_inactive < -config.kkt_tolerance:
        return None
    return weight, stationarity, minimum_inactive, simplex_error


def solve_cell_active_set(
    gram: np.ndarray,
    linear: np.ndarray,
    constant: float,
    config: ActiveSetConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Globally solve one PSD simplex QP by scale-invariant support enumeration."""
    cfg = config or ActiveSetConfig()
    cfg.validate()
    g = np.asarray(gram, dtype=np.float64)
    c = np.asarray(linear, dtype=np.float64)
    if g.shape != (len(c), len(c)) or len(c) < 1:
        raise ValueError("cell Gram/linear shapes are invalid")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(c)) or not np.isfinite(constant):
        raise ValueError("cell quadratic must be finite")
    g = 0.5 * (g + g.T)
    scale = _cell_scale(g, c)
    gn = g / scale
    cn = c / scale
    eigenvalues = np.linalg.eigvalsh(gn)
    if eigenvalues[0] < -cfg.kkt_tolerance:
        raise TailOracleError(f"cell Gram is not PSD after equilibration: {eigenvalues[0]}")
    if eigenvalues[0] < 0:
        gn += np.eye(len(c)) * (-float(eigenvalues[0]) + 1e-15)

    best: tuple[float, tuple[int, ...], np.ndarray, dict[str, float]] | None = None
    for support in _supports(len(c)):
        solved = _solve_support(gn, cn, support, cfg)
        if solved is None:
            continue
        weight, stationarity, minimum_inactive, simplex_error = solved
        normalized_value = _cell_variable_value(weight, gn, cn)
        original_value = _cell_value(weight, g, c, float(constant))
        facts = {
            "objective": original_value,
            "normalized_variable_objective": normalized_value,
            "normalization_scale": scale,
            "normalized_stationarity_residual": stationarity,
            "normalized_minimum_inactive_reduced_gradient": minimum_inactive,
            "simplex_error": simplex_error,
        }
        candidate = (normalized_value, support, weight, facts)
        if best is None:
            best = candidate
        elif normalized_value < best[0] - cfg.objective_tolerance:
            best = candidate
        elif abs(normalized_value - best[0]) <= cfg.objective_tolerance:
            left = (len(support), support, tuple(np.round(weight, 15)))
            right = (len(best[1]), best[1], tuple(np.round(best[2], 15)))
            if left < right:
                best = candidate
    if best is None:
        raise TailOracleError("no KKT-feasible simplex active set found")

    normalized_value, support, weight, facts = best
    vertices = np.eye(len(c), dtype=np.float64)
    vertex_values = np.asarray([
        _cell_variable_value(vertex, gn, cn) for vertex in vertices
    ])
    if normalized_value > float(vertex_values.min()) + cfg.objective_tolerance:
        raise TailOracleError("active-set optimum is worse than a simplex vertex")
    return weight, {
        "support": list(support),
        "support_size": len(support),
        "interpolated": int(np.count_nonzero(weight > cfg.feasibility_tolerance)) > 1,
        **facts,
    }


def solve_independent_convex(
    quadratic: ConvexQuadratic,
    fallback_index: int,
    config: ActiveSetConfig | None = None,
) -> IndependentConvexResult:
    """Solve O0C exactly per identifiable cell and fill masked cells with O1."""
    cfg = config or ActiveSetConfig()
    cfg.validate()
    q, b, k = quadratic.linear.shape
    fallback = _validate_fallback(fallback_index, k)
    weights = np.zeros((q, b, k), dtype=np.float64)
    weights[..., fallback] = 1.0
    support_masks = np.zeros((q, b), dtype=np.uint32)
    histogram: dict[int, int] = {}
    interpolated = 0
    value_sum = 0.0
    max_simplex = 0.0
    max_stationarity = 0.0
    min_inactive = float("inf")
    scales: list[float] = []
    for qi in range(q):
        for bi in range(b):
            if not quadratic.available[qi, bi]:
                continue
            weight, facts = solve_cell_active_set(
                quadratic.gram[qi, bi], quadratic.linear[qi, bi],
                float(quadratic.constant[qi, bi]), cfg,
            )
            weights[qi, bi] = weight
            support = tuple(int(index) for index in facts["support"])
            mask = 0
            for index in support:
                mask |= 1 << index
            support_masks[qi, bi] = mask
            histogram[len(support)] = histogram.get(len(support), 0) + 1
            interpolated += int(bool(facts["interpolated"]))
            value_sum += float(facts["objective"])
            scales.append(float(facts["normalization_scale"]))
            max_simplex = max(max_simplex, float(facts["simplex_error"]))
            max_stationarity = max(
                max_stationarity,
                float(facts["normalized_stationarity_residual"]),
            )
            reduced = float(facts["normalized_minimum_inactive_reduced_gradient"])
            if np.isfinite(reduced):
                min_inactive = min(min_inactive, reduced)
    data = value_sum / quadratic.denominator
    return IndependentConvexResult(
        weights=weights,
        data_objective=float(data),
        available_cells=quadratic.denominator,
        unavailable_cells=int(quadratic.available.size - quadratic.denominator),
        active_set_size_histogram=histogram,
        interpolated_cells=interpolated,
        interpolated_fraction=interpolated / quadratic.denominator,
        max_simplex_error=max_simplex,
        max_normalized_stationarity_residual=max_stationarity,
        min_normalized_inactive_reduced_gradient=(
            min_inactive if np.isfinite(min_inactive) else None
        ),
        mean_weights=tuple(float(value) for value in weights.mean(axis=(0, 1))),
        support_masks=support_masks,
        cell_normalization_scale_min=float(min(scales)),
        cell_normalization_scale_max=float(max(scales)),
    )
