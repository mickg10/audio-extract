"""Globally converged convex-hull oracle for exact source-routing diagnostics.

The canonical discrete O2 route is solved globally as a Potts MILP. This module
provides the complementary O3 question:

    What is the best smoothly varying *convex combination* of the available
    separator estimates under a source-assignment-aware quadratic objective?

Unlike a softmax/Adam fit to the nonlinear evaluation metric, this program is
convex. It uses local complex source coordinates already held in
``CellStatistics``:

    Y_i = alpha_i A + beta_i V + R_i.

For real simplex weights ``w``:

    alpha(w) = sum_i w_i alpha_i,
    beta(w)  = sum_i w_i beta_i,
    R(w)     = sum_i w_i R_i.

The objective penalizes ``|alpha-1|^2``, retained-solo energy with a denominator
fixed by the true accompaniment energy, orthogonal residual energy, and
quadratic time/frequency weight variation. It is a positive-semidefinite
quadratic over a product of simplices.

The solver is deterministic monotone FISTA with exact simplex projections. A
result is returned only when the projected-gradient mapping satisfies the
configured KKT tolerance and independent starts agree. Iteration-limit results
are rejected rather than presented as an oracle envelope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .oracle_routing import CellStatistics


class ConvexOracleError(RuntimeError):
    """The convex oracle input or convergence certificate is invalid."""


@dataclass(frozen=True)
class ConvexOracleConfig:
    transfer_weight: float = 2.0
    retained_voice_weight: float = 1.0
    orthogonal_artifact_weight: float = 0.5
    temporal_l2_weight: float = 0.05
    frequency_l2_weight: float = 0.025
    reference_floor_relative: float = 1e-8
    max_iterations: int = 10_000
    gradient_mapping_tolerance: float = 1e-6
    objective_tolerance: float = 1e-10
    solution_agreement_tolerance: float = 1e-5
    psd_tolerance: float = 1e-10

    def validate(self) -> None:
        for name in (
            "transfer_weight", "retained_voice_weight",
            "orthogonal_artifact_weight", "temporal_l2_weight",
            "frequency_l2_weight", "reference_floor_relative",
            "gradient_mapping_tolerance", "objective_tolerance",
            "solution_agreement_tolerance", "psd_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.reference_floor_relative <= 0:
            raise ValueError("reference_floor_relative must be positive")
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if self.gradient_mapping_tolerance <= 0:
            raise ValueError("gradient_mapping_tolerance must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_dict(self) -> dict[str, Any]:
        return {
            key: (str(value) if isinstance(value, float) else value)
            for key, value in self.to_dict().items()
        }


@dataclass(frozen=True)
class ConvexQuadratic:
    gram: np.ndarray          # (Q,B,K,K), real symmetric PSD; zero if unavailable
    linear: np.ndarray        # (Q,B,K), objective has -2 linear·w
    constant: np.ndarray      # (Q,B), transfer constant; zero if unavailable
    available: np.ndarray     # (Q,B)
    denominator: int
    max_local_eigenvalue: float


@dataclass(frozen=True)
class ConvexOracleResult:
    weights: np.ndarray
    objective: float
    data_objective: float
    temporal_smoothness: float
    frequency_smoothness: float
    iterations: int
    converged: bool
    projected_gradient_mapping_inf: float
    simplex_error: float
    min_weight: float
    initialization: str
    solution_objective_spread: float
    best_vertex_index: int
    best_vertex_objective: float
    start_objectives: dict[str, float]
    restart_count: int
    accelerated_steps: int


@dataclass(frozen=True)
class IndependentDiscreteResult:
    labels: np.ndarray
    weights: np.ndarray
    objective: float
    local_objective_p90: float
    local_objective_max: float
    selection_counts: tuple[int, ...]


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection on the simplex, independently on the final axis."""
    x = np.asarray(values, dtype=np.float64)
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
        raise ConvexOracleError("simplex projection active set is empty")
    theta = cssv[np.arange(len(flat)), rho] / (rho + 1)
    result = np.maximum(flat - theta[:, None], 0.0).reshape(x.shape)
    if np.max(np.abs(result.sum(axis=-1) - 1.0)) > 1e-12:
        raise ConvexOracleError("simplex projection failed")
    return result


def build_quadratic(stats: CellStatistics,
                    config: ConvexOracleConfig) -> ConvexQuadratic:
    """Build the PSD source-coordinate quadratic from exact sufficient statistics."""
    config.validate()
    alpha = np.asarray(stats.alpha, dtype=np.complex128)
    beta = np.asarray(stats.beta, dtype=np.complex128)
    residual = np.asarray(stats.residual_gram, dtype=np.complex128)
    ea = np.asarray(stats.accompaniment_energy, dtype=np.float64)
    ev = np.asarray(stats.vocal_energy, dtype=np.float64)
    available = np.asarray(stats.available, dtype=bool)
    if alpha.shape != beta.shape or alpha.ndim != 3:
        raise ValueError("alpha and beta must be (time,band,candidate)")
    if residual.shape != alpha.shape + (alpha.shape[-1],):
        raise ValueError("residual Gram shape does not match source coordinates")
    if ea.shape != alpha.shape[:2] or ev.shape != ea.shape or available.shape != ea.shape:
        raise ValueError("energy/availability grids do not match source coordinates")
    if not (np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta))
            and np.all(np.isfinite(residual)) and np.all(np.isfinite(ea))
            and np.all(np.isfinite(ev))):
        raise ValueError("source-coordinate statistics must be finite")
    residual_scale = max(1.0, float(np.max(np.abs(residual))))
    hermitian_error = float(
        np.max(np.abs(residual - np.conj(np.swapaxes(residual, -1, -2))))
    )
    if hermitian_error > config.psd_tolerance * residual_scale:
        raise ConvexOracleError(
            "source-coordinate residual Gram is not Hermitian: "
            f"relative_error={hermitian_error / residual_scale}"
        )
    if not np.any(available):
        raise ConvexOracleError("no identifiable cells for convex oracle")

    q, b, k = alpha.shape
    gram = np.zeros((q, b, k, k), dtype=np.float64)
    linear = np.zeros((q, b, k), dtype=np.float64)
    constant = np.zeros((q, b), dtype=np.float64)
    maximum = 0.0
    for qi in range(q):
        for bi in range(b):
            if not available[qi, bi]:
                continue
            epsilon = config.reference_floor_relative * max(
                float(ea[qi, bi] + ev[qi, bi]), np.finfo(float).tiny
            )
            accompaniment_scale = max(float(ea[qi, bi]), 0.0) + epsilon
            local = (
                config.transfer_weight
                * np.real(np.outer(np.conj(alpha[qi, bi]), alpha[qi, bi]))
                + config.retained_voice_weight
                * (max(float(ev[qi, bi]), 0.0) / accompaniment_scale)
                * np.real(np.outer(np.conj(beta[qi, bi]), beta[qi, bi]))
                + config.orthogonal_artifact_weight
                * np.real(residual[qi, bi]) / accompaniment_scale
            )
            local = 0.5 * (local + local.T)
            eigenvalues = np.linalg.eigvalsh(local)
            eigen_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
            if eigenvalues[0] < -config.psd_tolerance * eigen_scale:
                raise ConvexOracleError(
                    f"source-coordinate quadratic is not PSD: min={eigenvalues[0]}"
                )
            if eigenvalues[0] < 0:
                local += np.eye(k) * (-float(eigenvalues[0]) + 1e-12)
                eigenvalues = np.linalg.eigvalsh(local)
            gram[qi, bi] = local
            linear[qi, bi] = config.transfer_weight * np.real(alpha[qi, bi])
            constant[qi, bi] = config.transfer_weight
            maximum = max(maximum, float(eigenvalues[-1]))
    return ConvexQuadratic(
        gram=gram, linear=linear, constant=constant,
        available=available, denominator=int(available.sum()),
        max_local_eigenvalue=maximum,
    )


def local_data_objective(
    weights: np.ndarray, quadratic: ConvexQuadratic,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    expected = quadratic.linear.shape
    if w.shape != expected or not np.all(np.isfinite(w)):
        raise ValueError(f"weights must be finite with shape {expected}")
    simplex_error = float(np.max(np.abs(w.sum(axis=-1) - 1.0)))
    if float(w.min()) < -1e-10 or simplex_error > 1e-8:
        raise ValueError("weights are outside the product simplex")
    return (
        np.einsum("qbk,qbkl,qbl->qb", w, quadratic.gram, w)
        - 2.0 * np.einsum("qbk,qbk->qb", quadratic.linear, w)
        + quadratic.constant
    )


def objective(weights: np.ndarray, quadratic: ConvexQuadratic,
              config: ConvexOracleConfig) -> tuple[float, float, float, float]:
    w = np.asarray(weights, dtype=np.float64)
    local = local_data_objective(w, quadratic)
    data = float(local[quadratic.available].sum() / quadratic.denominator)
    temporal_raw = float(np.square(w[1:] - w[:-1]).sum())
    frequency_raw = float(np.square(w[:, 1:] - w[:, :-1]).sum())
    temporal = config.temporal_l2_weight * temporal_raw / quadratic.denominator
    frequency = config.frequency_l2_weight * frequency_raw / quadratic.denominator
    return data + temporal + frequency, data, temporal, frequency


def solve_independent_discrete_oracle(
    stats: CellStatistics, *, fallback_index: int,
    config: ConvexOracleConfig | None = None,
) -> IndependentDiscreteResult:
    """O0D: choose the best corrected-quadratic vertex in each available cell."""
    cfg = config or ConvexOracleConfig()
    quadratic = build_quadratic(stats, cfg)
    _, _, candidates = quadratic.linear.shape
    if not 0 <= int(fallback_index) < candidates:
        raise ValueError("fallback_index outside candidate basis")
    vertex_cost = (
        np.diagonal(quadratic.gram, axis1=-2, axis2=-1)
        - 2.0 * quadratic.linear + quadratic.constant[..., None]
    )
    labels = np.argmin(vertex_cost, axis=-1).astype(np.int64)
    labels[~quadratic.available] = int(fallback_index)
    weights = _one_hot(labels, candidates)
    local = local_data_objective(weights, quadratic)[quadratic.available]
    return IndependentDiscreteResult(
        labels=labels,
        weights=weights,
        objective=float(local.mean()),
        local_objective_p90=float(np.percentile(local, 90)),
        local_objective_max=float(local.max()),
        selection_counts=tuple(int(np.count_nonzero(labels == i)) for i in range(candidates)),
    )


def solve_independent_convex_oracle(
    stats: CellStatistics, *, o1_index: int, o2_labels: np.ndarray,
    config: ConvexOracleConfig | None = None,
) -> tuple[ConvexOracleResult, ConvexOracleConfig]:
    """O0C: globally certify the separable no-smoothness convex-cell envelope."""
    base = config or ConvexOracleConfig()
    independent = replace(base, temporal_l2_weight=0.0, frequency_l2_weight=0.0)
    result = solve_true_convex_oracle(
        stats, o1_index=o1_index, o2_labels=o2_labels, config=independent
    )
    weights = result.weights.copy()
    weights[~stats.available] = 0.0
    weights[~stats.available, int(o1_index)] = 1.0
    return replace(result, weights=weights), independent


def gradient(weights: np.ndarray, quadratic: ConvexQuadratic,
             config: ConvexOracleConfig) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    grad = (
        2.0 * np.einsum("qbkl,qbl->qbk", quadratic.gram, w)
        - 2.0 * quadratic.linear
    ) / quadratic.denominator
    if config.temporal_l2_weight:
        scale = 2.0 * config.temporal_l2_weight / quadratic.denominator
        grad[1:] += scale * (w[1:] - w[:-1])
        grad[:-1] += scale * (w[:-1] - w[1:])
    if config.frequency_l2_weight:
        scale = 2.0 * config.frequency_l2_weight / quadratic.denominator
        grad[:, 1:] += scale * (w[:, 1:] - w[:, :-1])
        grad[:, :-1] += scale * (w[:, :-1] - w[:, 1:])
    return grad


def lipschitz_constant(quadratic: ConvexQuadratic,
                       config: ConvexOracleConfig) -> float:
    # A path-graph Laplacian has eigenvalue <=4. The Hessian of
    # lambda*sum||w_i-w_j||^2 is 2*lambda*L, hence <=8*lambda.
    value = (
        2.0 * quadratic.max_local_eigenvalue
        + 8.0 * config.temporal_l2_weight
        + 8.0 * config.frequency_l2_weight
    ) / quadratic.denominator
    if not np.isfinite(value) or value <= 0:
        raise ConvexOracleError(f"invalid gradient Lipschitz constant: {value}")
    return float(value)


def projected_gradient_mapping_inf(weights: np.ndarray, quadratic: ConvexQuadratic,
                                   config: ConvexOracleConfig,
                                   lipschitz: float) -> float:
    projected = project_simplex(
        weights - gradient(weights, quadratic, config) / lipschitz
    )
    mapping = lipschitz * (weights - projected)
    return float(np.max(np.abs(mapping)))


def _one_hot(labels: np.ndarray, candidates: int) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64)
    if np.any(lab < 0) or np.any(lab >= candidates):
        raise ValueError("label outside candidate basis")
    result = np.zeros(lab.shape + (candidates,), dtype=np.float64)
    np.put_along_axis(result, lab[..., None], 1.0, axis=-1)
    return result


def _monotone_fista(start: np.ndarray, quadratic: ConvexQuadratic,
                    config: ConvexOracleConfig) -> tuple[np.ndarray, dict[str, Any]]:
    L = lipschitz_constant(quadratic, config)
    x = project_simplex(start)
    y = x.copy()
    momentum = 1.0
    current = objective(x, quadratic, config)[0]
    best = x.copy()
    best_value = current
    previous_value = current
    converged = False
    mapping = projected_gradient_mapping_inf(x, quadratic, config, L)
    iterations = 0
    restarts = 0
    accelerated_steps = 0
    for iterations in range(1, config.max_iterations + 1):
        extrapolation_origin = y
        proposal = project_simplex(y - gradient(y, quadratic, config) / L)
        proposal_value = objective(proposal, quadratic, config)[0]
        # Monotone restart: acceleration is optional, objective increase is not.
        if proposal_value > current + config.objective_tolerance:
            y = x
            extrapolation_origin = y
            momentum = 1.0
            restarts += 1
            proposal = project_simplex(y - gradient(y, quadratic, config) / L)
            proposal_value = objective(proposal, quadratic, config)[0]
        if proposal_value > current + config.objective_tolerance:
            raise ConvexOracleError(
                f"projected step increased convex objective {current} -> {proposal_value}"
            )
        previous = x
        x = proposal
        current = proposal_value
        if current < best_value:
            best_value = current
            best = x.copy()
        mapping = projected_gradient_mapping_inf(x, quadratic, config, L)
        objective_change = abs(previous_value - current)
        if (mapping <= config.gradient_mapping_tolerance
                and objective_change <= config.objective_tolerance * max(1.0, abs(current))):
            converged = True
            break
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = x + ((momentum - 1.0) / next_momentum) * (x - previous)
        # O'Donoghue-Candes gradient restart criterion:
        # <y_k - x_{k+1}, x_{k+1} - x_k> > 0.
        if float(np.sum((extrapolation_origin - x) * (x - previous))) > 0:
            y = x
            next_momentum = 1.0
            restarts += 1
        else:
            y = accelerated
            if next_momentum > 1.0:
                accelerated_steps += 1
        momentum = next_momentum
        previous_value = current
    if not converged:
        raise ConvexOracleError(
            "convex O3 did not converge: "
            f"iterations={iterations}, projected_gradient_mapping_inf={mapping}"
        )
    # ``x`` is the iterate for which the stopping certificate was evaluated.
    # Returning an earlier best-objective iterate can violate KKT by a few ULPs
    # when monotone tolerance admits a numerically insignificant increase.
    certified = x
    total, data, temporal, frequency = objective(certified, quadratic, config)
    certified_mapping = projected_gradient_mapping_inf(
        certified, quadratic, config, L
    )
    if certified_mapping > config.gradient_mapping_tolerance:
        raise ConvexOracleError(
            f"returned convex iterate lacks KKT certificate: {certified_mapping}"
        )
    if total > best_value + config.objective_tolerance * max(1.0, abs(best_value)):
        raise ConvexOracleError(
            f"certified iterate lost objective monotonicity: {total} > {best_value}"
        )
    return certified, {
        "objective": total,
        "data_objective": data,
        "temporal_smoothness": temporal,
        "frequency_smoothness": frequency,
        "iterations": iterations,
        "converged": True,
        "projected_gradient_mapping_inf": certified_mapping,
        "simplex_error": float(np.max(np.abs(certified.sum(axis=-1) - 1.0))),
        "min_weight": float(certified.min()),
        "lipschitz_constant": L,
        "restart_count": restarts,
        "accelerated_steps": accelerated_steps,
    }


def solve_true_convex_oracle(
    stats: CellStatistics, *, o1_index: int, o2_labels: np.ndarray,
    config: ConvexOracleConfig | None = None,
) -> ConvexOracleResult:
    cfg = config or ConvexOracleConfig()
    quadratic = build_quadratic(stats, cfg)
    q, b, k = stats.alpha.shape
    starts = {
        "uniform": np.full((q, b, k), 1.0 / k, dtype=np.float64),
        "O1": _one_hot(np.full((q, b), int(o1_index)), k),
        "O2": _one_hot(np.asarray(o2_labels, dtype=np.int64), k),
    }
    solutions: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for name, start in starts.items():
        weights, report = _monotone_fista(start, quadratic, cfg)
        solutions.append((name, weights, report))
    objectives = np.asarray([item[2]["objective"] for item in solutions])
    spread = float(objectives.max() - objectives.min())
    if spread > cfg.solution_agreement_tolerance:
        raise ConvexOracleError(
            f"independent convex starts disagree: objective spread={spread}"
        )
    name, weights, report = min(solutions, key=lambda item: item[2]["objective"])

    vertex_objectives = []
    for candidate in range(k):
        vertex = _one_hot(np.full((q, b), candidate, dtype=np.int64), k)
        vertex_objectives.append(objective(vertex, quadratic, cfg)[0])
    best_vertex = int(np.argmin(vertex_objectives))
    best_vertex_objective = float(vertex_objectives[best_vertex])
    if report["objective"] > best_vertex_objective + cfg.objective_tolerance:
        raise ConvexOracleError(
            "converged convex solution is worse than the best feasible vertex"
        )
    return ConvexOracleResult(
        weights=weights,
        objective=float(report["objective"]),
        data_objective=float(report["data_objective"]),
        temporal_smoothness=float(report["temporal_smoothness"]),
        frequency_smoothness=float(report["frequency_smoothness"]),
        iterations=int(report["iterations"]),
        converged=True,
        projected_gradient_mapping_inf=float(report["projected_gradient_mapping_inf"]),
        simplex_error=float(report["simplex_error"]),
        min_weight=float(report["min_weight"]),
        initialization=name,
        solution_objective_spread=spread,
        best_vertex_index=best_vertex,
        best_vertex_objective=best_vertex_objective,
        start_objectives={item[0]: float(item[2]["objective"]) for item in solutions},
        restart_count=int(report["restart_count"]),
        accelerated_steps=int(report["accelerated_steps"]),
    )
