"""Unsmoothed exact routing envelopes and shared discrete costs.

A smooth mean-optimal route can conceal the product's rare worst-event failures.
This module supplies two diagnostic upper bounds on the fixed candidate basis:

``O0D``
    Independent best real candidate in each identifiable time/frequency cell.

``O0C``
    Independent exact convex-hull optimum in each identifiable cell, solved by
    exhaustive active-set KKT enumeration.  With the preregistered five-member
    basis this is only 31 supports per cell and avoids iterative/convergence
    ambiguity.

Unavailable cells are filled with the exact O1 label for deterministic audio
reconstruction but are excluded from the data objective.  O0 outputs are
``diagnostic_only`` and are never production candidates.

The same PSD quadratic also defines the corrected whole-candidate unary used by
O1 and the exact O2 Potts MILP:

    vertex_unary = diag(G) - 2*c + constant.
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
    feasibility_tolerance: float = 1e-9
    kkt_tolerance: float = 1e-8
    objective_tolerance: float = 1e-12

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")


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
    max_stationarity_residual: float
    min_inactive_reduced_gradient: float
    mean_weights: tuple[float, ...]
    support_masks: np.ndarray


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


def _solve_support(
    gram: np.ndarray,
    linear: np.ndarray,
    support: tuple[int, ...],
    config: ActiveSetConfig,
) -> tuple[np.ndarray, float, float, float] | None:
    """Return a KKT-certified simplex solution on one support, or ``None``."""
    k = len(linear)
    indices = np.asarray(support, dtype=np.int64)
    gs = gram[np.ix_(indices, indices)]
    cs = linear[indices]
    ones = np.ones(len(indices), dtype=np.float64)
    system = np.block([
        [gs, ones[:, None]],
        [ones[None, :], np.zeros((1, 1), dtype=np.float64)],
    ])
    rhs = np.concatenate((cs, np.array([1.0], dtype=np.float64)))
    solution, _, _, _ = np.linalg.lstsq(system, rhs, rcond=None)
    residual = system @ solution - rhs
    if float(np.max(np.abs(residual))) > config.kkt_tolerance:
        return None
    active = solution[:-1]
    multiplier = float(solution[-1])
    if float(active.min(initial=0.0)) < -config.feasibility_tolerance:
        return None
    active = np.maximum(active, 0.0)
    total = float(active.sum())
    if total <= 0:
        return None
    active /= total
    weight = np.zeros(k, dtype=np.float64)
    weight[indices] = active

    reduced = gram @ weight - linear + multiplier
    stationarity = float(np.max(np.abs(reduced[indices])))
    inactive = np.ones(k, dtype=bool)
    inactive[indices] = False
    minimum_inactive = (
        float(reduced[inactive].min()) if np.any(inactive) else float("inf")
    )
    simplex_error = abs(float(weight.sum()) - 1.0)
    if stationarity > config.kkt_tolerance:
        return None
    if minimum_inactive < -config.kkt_tolerance:
        return None
    if simplex_error > config.feasibility_tolerance:
        return None
    return weight, stationarity, minimum_inactive, simplex_error


def solve_cell_active_set(
    gram: np.ndarray,
    linear: np.ndarray,
    constant: float,
    config: ActiveSetConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Globally solve one PSD simplex QP by active-set enumeration."""
    cfg = config or ActiveSetConfig()
    cfg.validate()
    g = np.asarray(gram, dtype=np.float64)
    c = np.asarray(linear, dtype=np.float64)
    if g.shape != (len(c), len(c)) or len(c) < 1:
        raise ValueError("cell Gram/linear shapes are invalid")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(c)) or not np.isfinite(constant):
        raise ValueError("cell quadratic must be finite")
    g = 0.5 * (g + g.T)
    eigenvalues = np.linalg.eigvalsh(g)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues[0] < -cfg.kkt_tolerance * scale:
        raise TailOracleError(f"cell Gram is not PSD: {eigenvalues[0]}")

    best: tuple[float, tuple[int, ...], np.ndarray, dict[str, float]] | None = None
    for support in _supports(len(c)):
        solved = _solve_support(g, c, support, cfg)
        if solved is None:
            continue
        weight, stationarity, minimum_inactive, simplex_error = solved
        value = _cell_value(weight, g, c, float(constant))
        facts = {
            "stationarity_residual": stationarity,
            "minimum_inactive_reduced_gradient": minimum_inactive,
            "simplex_error": simplex_error,
        }
        candidate = (value, support, weight, facts)
        if best is None:
            best = candidate
        elif value < best[0] - cfg.objective_tolerance:
            best = candidate
        elif abs(value - best[0]) <= cfg.objective_tolerance:
            # Deterministic tie-break: smaller support, then lexicographic support,
            # then lexicographic rounded weight vector.
            left = (len(support), support, tuple(np.round(weight, 15)))
            right = (len(best[1]), best[1], tuple(np.round(best[2], 15)))
            if left < right:
                best = candidate
    if best is None:
        raise TailOracleError("no KKT-feasible simplex active set found")

    value, support, weight, facts = best
    vertices = np.eye(len(c), dtype=np.float64)
    vertex_values = np.asarray([
        _cell_value(vertex, g, c, float(constant)) for vertex in vertices
    ])
    if value > float(vertex_values.min()) + cfg.objective_tolerance:
        raise TailOracleError("active-set optimum is worse than a simplex vertex")
    return weight, {
        "objective": value,
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
            max_simplex = max(max_simplex, float(facts["simplex_error"]))
            max_stationarity = max(
                max_stationarity, float(facts["stationarity_residual"])
            )
            reduced = float(facts["minimum_inactive_reduced_gradient"])
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
        max_stationarity_residual=max_stationarity,
        min_inactive_reduced_gradient=min_inactive,
        mean_weights=tuple(float(value) for value in weights.mean(axis=(0, 1))),
        support_masks=support_masks,
    )
