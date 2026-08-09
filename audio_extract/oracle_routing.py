"""Exact-reference routing envelope over immutable accompaniment candidates.

This module is deliberately a diagnostic, not a deployable judge.  It may use
known accompaniment/featured-solo truth to answer one bounded question: does a
smooth route through the existing estimators contain a materially better answer
than every whole-track estimator?  The rendered output is still a normal FLOAT
recipe node with explicit parents; truth never enters the production separator.

The local loss follows the preregistered source-coordinate decomposition

``Y = alpha A + beta V + R``

on complex STFT time/band cells.  Cells with too little source energy or an
ill-conditioned ``[A V]`` basis are unavailable, never clean-looking zeros.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RoutingConfig:
    """Frozen identity-bearing oracle-envelope settings."""

    sample_rate_hz: int = 44_100
    n_fft: int = 2048
    hop_length: int = 1024
    tile_seconds: float = 2.0
    band_edges_hz: tuple[int, ...] = (
        0, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 22_050
    )
    ridge_relative: float = 1e-8
    max_condition: float = 1e6
    min_source_power_relative: float = 1e-6
    reference_floor_relative: float = 1e-8
    lambda_hole: float = 1.0
    lambda_voice: float = 1.0
    lambda_residual: float = 1.0
    temporal_switch_penalty: float = 0.05
    frequency_switch_penalty: float = 0.05
    o3_iterations: int = 800
    o3_learning_rate: float = 0.08
    o3_softmax_extent: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # JSON round-trips must compare equal on immutable cache re-entry.
        payload["band_edges_hz"] = list(payload["band_edges_hz"])
        return payload

    def identity_dict(self) -> dict[str, Any]:
        """Canonical-recipe form: every non-integer quantity is exact text."""

        payload = self.to_dict()
        return {
            key: (str(value) if isinstance(value, float) else value)
            for key, value in payload.items()
        }


@dataclass
class CellStatistics:
    """Sufficient statistics for every time/band cell and candidate."""

    alpha: np.ndarray  # (Q, B, K), complex128
    beta: np.ndarray  # (Q, B, K), complex128
    residual_gram: np.ndarray  # (Q, B, K, K), complex128
    accompaniment_energy: np.ndarray  # (Q, B)
    vocal_energy: np.ndarray  # (Q, B)
    condition_number: np.ndarray  # (Q, B)
    available: np.ndarray  # (Q, B), bool
    unary_risk: np.ndarray  # (Q, B, K), finite; zero where unavailable
    time_frame_ranges: tuple[tuple[int, int], ...]
    frequency_bin_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DiscreteRoutingResult:
    labels: np.ndarray
    objective: float
    data_objective: float
    temporal_switches: int
    frequency_switches: int
    solver_status: str
    mip_gap: float | None
    mip_node_count: int | None


@dataclass(frozen=True)
class ConvexRoutingResult:
    weights: np.ndarray
    objective: float
    data_objective: float
    temporal_tv: float
    frequency_tv: float
    initialization: str
    iterations: int


def _as_exact_stereo(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must have shape (frames, 2), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite floating samples")
    return np.asarray(arr, dtype=np.float32)


def validate_basis(
    candidates: Sequence[np.ndarray], accompaniment: np.ndarray, vocal: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Reject any implicit truncation, resampling, or channel coercion."""

    a = _as_exact_stereo(accompaniment, "accompaniment")
    v = _as_exact_stereo(vocal, "vocal")
    if a.shape != v.shape:
        raise ValueError(f"truth grids differ: {a.shape} != {v.shape}")
    ys = [_as_exact_stereo(value, f"candidate[{i}]") for i, value in enumerate(candidates)]
    if len(ys) < 2:
        raise ValueError("routing envelope needs at least two candidates")
    for i, y in enumerate(ys):
        if y.shape != a.shape:
            raise ValueError(f"candidate[{i}] grid {y.shape} != truth grid {a.shape}")
    return ys, a, v


def stft_stack(signals: Sequence[np.ndarray], config: RoutingConfig) -> np.ndarray:
    """Return complex64 ``(members, channels, frequency, time)`` spectra."""

    import librosa

    result = []
    for member, signal in enumerate(signals):
        x = _as_exact_stereo(signal, f"signal[{member}]")
        channels = [
            librosa.stft(
                x[:, channel], n_fft=config.n_fft, hop_length=config.hop_length,
                win_length=config.n_fft, window="hann", center=True,
                pad_mode="constant",
            )
            for channel in range(2)
        ]
        result.append(np.stack(channels, axis=0))
    return np.asarray(result, dtype=np.complex64)


def _time_ranges(time_frames: int, config: RoutingConfig) -> tuple[tuple[int, int], ...]:
    per_tile = max(1, round(config.tile_seconds * config.sample_rate_hz / config.hop_length))
    return tuple((start, min(start + per_tile, time_frames))
                 for start in range(0, time_frames, per_tile))


def _frequency_ranges(frequency_bins: int, config: RoutingConfig) -> tuple[tuple[int, int], ...]:
    frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate_hz)
    if frequency_bins != len(frequencies):
        raise ValueError(f"frequency grid {frequency_bins} != rFFT grid {len(frequencies)}")
    edges = tuple(int(v) for v in config.band_edges_hz)
    if len(edges) < 2 or edges[0] != 0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("band_edges_hz must be strictly increasing from zero")
    if edges[-1] != config.sample_rate_hz // 2:
        raise ValueError("last band edge must equal Nyquist")
    ranges = []
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        start = int(np.searchsorted(frequencies, low, side="left"))
        side = "right" if index == len(edges) - 2 else "left"
        end = int(np.searchsorted(frequencies, high, side=side))
        if end <= start:
            raise ValueError(f"empty frequency band {low}..{high} Hz")
        ranges.append((start, end))
    return tuple(ranges)


def _cell_fit(
    candidates: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    config: RoutingConfig,
    accompaniment_floor: float,
    vocal_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, bool, np.ndarray]:
    """Fit all candidates in one complex cell using a shared exact basis."""

    a = np.asarray(accompaniment, dtype=np.complex128).reshape(-1)
    v = np.asarray(vocal, dtype=np.complex128).reshape(-1)
    ys = np.asarray(candidates, dtype=np.complex128).reshape(len(candidates), -1)
    x = np.column_stack((a, v))
    gram = x.conj().T @ x
    ea, ev = (float(max(value.real, 0.0)) for value in np.diag(gram))
    try:
        condition = float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        condition = np.inf
    available = (
        ea > accompaniment_floor and ev > vocal_floor
        and np.isfinite(condition) and condition <= config.max_condition
    )
    k = len(ys)
    if not available:
        return (
            np.zeros(k, dtype=np.complex128), np.zeros(k, dtype=np.complex128),
            np.zeros((k, k), dtype=np.complex128), ea, ev, condition, False,
            np.zeros(k, dtype=np.float64),
        )
    ridge = config.ridge_relative * max(float(np.trace(gram).real) / 2.0,
                                         np.finfo(np.float64).tiny)
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(2, dtype=np.complex128), x.conj().T @ ys.T
    )
    alpha, beta = coefficients[0], coefficients[1]
    residual = ys - alpha[:, None] * a[None] - beta[:, None] * v[None]
    residual_gram = residual.conj() @ residual.T
    epsilon = config.reference_floor_relative * max(ea + ev, np.finfo(float).tiny)
    hole = np.maximum(1.0 - np.abs(alpha), 0.0) ** 2
    voice = np.abs(beta) ** 2 * ev / (np.abs(alpha) ** 2 * ea + epsilon)
    artifact = np.maximum(np.real(np.diag(residual_gram)), 0.0) / (ea + epsilon)
    risk = (config.lambda_hole * hole + config.lambda_voice * voice
            + config.lambda_residual * artifact)
    if not np.all(np.isfinite(risk)):
        raise RuntimeError("non-finite local oracle risk")
    return alpha, beta, residual_gram, ea, ev, condition, True, risk


def source_coordinate_statistics(
    candidate_spectra: np.ndarray,
    accompaniment_spectrum: np.ndarray,
    vocal_spectrum: np.ndarray,
    config: RoutingConfig,
) -> CellStatistics:
    """Compute masked complex source-coordinate sufficient statistics."""

    ys = np.asarray(candidate_spectra)
    a = np.asarray(accompaniment_spectrum)
    v = np.asarray(vocal_spectrum)
    if ys.ndim != 4 or ys.shape[1] != 2:
        raise ValueError(f"candidate_spectra must be (K,2,F,T), got {ys.shape}")
    if a.shape != ys.shape[1:] or v.shape != a.shape:
        raise ValueError(f"spectral grids differ: {ys.shape}, {a.shape}, {v.shape}")
    if len(ys) < 2 or not (np.all(np.isfinite(ys)) and np.all(np.isfinite(a))
                           and np.all(np.isfinite(v))):
        raise ValueError("spectra must contain at least two finite candidates")
    tranges = _time_ranges(ys.shape[-1], config)
    franges = _frequency_ranges(ys.shape[-2], config)
    shape = (len(tranges), len(franges))
    k = len(ys)
    alpha = np.zeros(shape + (k,), dtype=np.complex128)
    beta = np.zeros_like(alpha)
    residual_gram = np.zeros(shape + (k, k), dtype=np.complex128)
    ea = np.zeros(shape, dtype=np.float64)
    ev = np.zeros(shape, dtype=np.float64)
    condition = np.full(shape, np.inf, dtype=np.float64)
    available = np.zeros(shape, dtype=bool)
    risk = np.zeros(shape + (k,), dtype=np.float64)
    global_a_power = float(np.mean(np.abs(a) ** 2))
    global_v_power = float(np.mean(np.abs(v) ** 2))
    for qi, (t0, t1) in enumerate(tranges):
        for bi, (f0, f1) in enumerate(franges):
            count = 2 * (t1 - t0) * (f1 - f0)
            result = _cell_fit(
                ys[:, :, f0:f1, t0:t1], a[:, f0:f1, t0:t1],
                v[:, f0:f1, t0:t1], config,
                config.min_source_power_relative * global_a_power * count,
                config.min_source_power_relative * global_v_power * count,
            )
            (alpha[qi, bi], beta[qi, bi], residual_gram[qi, bi], ea[qi, bi],
             ev[qi, bi], condition[qi, bi], available[qi, bi], risk[qi, bi]) = result
    if not np.any(available):
        raise RuntimeError("no identifiable oracle-routing cells")
    return CellStatistics(alpha, beta, residual_gram, ea, ev, condition,
                          available, risk, tranges, franges)


def _edge_list(q: int, b: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    temporal = [((qi * b) + bi, ((qi + 1) * b) + bi)
                for qi in range(q - 1) for bi in range(b)]
    frequency = [((qi * b) + bi, (qi * b) + bi + 1)
                 for qi in range(q) for bi in range(b - 1)]
    return temporal, frequency


def _discrete_objective(labels: np.ndarray, stats: CellStatistics,
                        config: RoutingConfig) -> tuple[float, float, int, int]:
    q, b, k = stats.unary_risk.shape
    lab = np.asarray(labels, dtype=np.int64)
    if lab.shape != (q, b) or np.any(lab < 0) or np.any(lab >= k):
        raise ValueError("invalid routing labels")
    chosen = np.take_along_axis(stats.unary_risk, lab[..., None], axis=-1)[..., 0]
    denom = max(1, int(stats.available.sum()))
    data = float(chosen[stats.available].sum() / denom)
    ts = int(np.count_nonzero(lab[1:] != lab[:-1]))
    fs = int(np.count_nonzero(lab[:, 1:] != lab[:, :-1]))
    objective = data + (
        config.temporal_switch_penalty * ts
        + config.frequency_switch_penalty * fs
    ) / denom
    return objective, data, ts, fs


def best_whole_track(stats: CellStatistics) -> tuple[int, np.ndarray]:
    """O1: best feasible one-candidate route on the masked local risk."""

    means = stats.unary_risk[stats.available].mean(axis=0)
    return int(np.argmin(means)), means


def solve_discrete_routing(stats: CellStatistics, config: RoutingConfig) -> DiscreteRoutingResult:
    """O2: globally solve the Potts grid as a mixed-integer linear program."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    q, b, k = stats.unary_risk.shape
    cells = q * b
    temporal, frequency = _edge_list(q, b)
    edges = [(u, v, config.temporal_switch_penalty) for u, v in temporal]
    edges += [(u, v, config.frequency_switch_penalty) for u, v in frequency]
    x_vars = cells * k
    d_vars = len(edges) * k
    nvars = x_vars + d_vars
    denom = max(1, int(stats.available.sum()))
    c = np.zeros(nvars, dtype=np.float64)
    c[:x_vars] = stats.unary_risk.reshape(-1) / denom
    for ei, (_, _, penalty) in enumerate(edges):
        c[x_vars + ei * k:x_vars + (ei + 1) * k] = penalty / (2.0 * denom)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for cell in range(cells):
        for candidate in range(k):
            rows.append(row); cols.append(cell * k + candidate); data.append(1.0)
        lower.append(1.0); upper.append(1.0); row += 1
    for ei, (u, v, _) in enumerate(edges):
        for candidate in range(k):
            d = x_vars + ei * k + candidate
            # x_u - x_v - d <= 0; x_v - x_u - d <= 0.
            for left, right in ((u, v), (v, u)):
                rows.extend((row, row, row))
                cols.extend((left * k + candidate, right * k + candidate, d))
                data.extend((1.0, -1.0, -1.0))
                lower.append(-np.inf); upper.append(0.0); row += 1
    matrix = coo_matrix((data, (rows, cols)), shape=(row, nvars)).tocsr()
    result = milp(
        c, integrality=np.r_[np.ones(x_vars, dtype=np.int8),
                             np.zeros(d_vars, dtype=np.int8)],
        bounds=Bounds(np.zeros(nvars), np.ones(nvars)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 180.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"O2 MILP failed: status={result.status}, message={result.message}")
    labels = np.argmax(result.x[:x_vars].reshape(q, b, k), axis=-1)
    objective, data_obj, ts, fs = _discrete_objective(labels, stats, config)
    return DiscreteRoutingResult(
        labels=labels, objective=objective, data_objective=data_obj,
        temporal_switches=ts, frequency_switches=fs,
        solver_status=str(result.message),
        mip_gap=(None if getattr(result, "mip_gap", None) is None
                 else float(result.mip_gap)),
        mip_node_count=(None if getattr(result, "mip_node_count", None) is None
                        else int(result.mip_node_count)),
    )


def _convex_objective_numpy(weights: np.ndarray, stats: CellStatistics,
                            config: RoutingConfig) -> tuple[float, float, float, float]:
    w = np.asarray(weights, dtype=np.float64)
    q, b, k = stats.unary_risk.shape
    if w.shape != (q, b, k) or np.any(w < -1e-8) or not np.allclose(w.sum(-1), 1.0, atol=1e-6):
        raise ValueError("convex weights must be a per-cell simplex")
    alpha = np.sum(w * stats.alpha, axis=-1)
    beta = np.sum(w * stats.beta, axis=-1)
    residual = np.real(np.einsum("qbi,qbij,qbj->qb", w, stats.residual_gram, w))
    eps = config.reference_floor_relative * np.maximum(
        stats.accompaniment_energy + stats.vocal_energy, np.finfo(float).tiny
    )
    risk = (
        config.lambda_hole * np.maximum(1.0 - np.abs(alpha), 0.0) ** 2
        + config.lambda_voice * np.abs(beta) ** 2 * stats.vocal_energy
        / (np.abs(alpha) ** 2 * stats.accompaniment_energy + eps)
        + config.lambda_residual * np.maximum(residual, 0.0)
        / (stats.accompaniment_energy + eps)
    )
    denom = max(1, int(stats.available.sum()))
    data = float(risk[stats.available].sum() / denom)
    temporal_tv = float(np.abs(w[1:] - w[:-1]).sum())
    frequency_tv = float(np.abs(w[:, 1:] - w[:, :-1]).sum())
    objective = data + 0.5 * (
        config.temporal_switch_penalty * temporal_tv
        + config.frequency_switch_penalty * frequency_tv
    ) / denom
    return objective, data, temporal_tv, frequency_tv


def solve_convex_routing(
    stats: CellStatistics, config: RoutingConfig, *, o1_index: int,
    o2_labels: np.ndarray,
) -> ConvexRoutingResult:
    """O3: optimize nonnegative local weights with time/frequency TV.

    Four deterministic starts are used.  Exact one-hot O1 and O2 routes are also
    feasible candidates, so the reported convex envelope can never be worse than
    either merely because the softmax optimizer missed an endpoint.
    """

    import torch

    torch.set_num_threads(1)
    q, b, k = stats.unary_risk.shape
    available = torch.as_tensor(stats.available, dtype=torch.float64)
    alpha = torch.as_tensor(stats.alpha, dtype=torch.complex128)
    beta = torch.as_tensor(stats.beta, dtype=torch.complex128)
    gram = torch.as_tensor(stats.residual_gram, dtype=torch.complex128)
    ea = torch.as_tensor(stats.accompaniment_energy, dtype=torch.float64)
    ev = torch.as_tensor(stats.vocal_energy, dtype=torch.float64)
    eps = config.reference_floor_relative * torch.clamp(ea + ev, min=torch.finfo(torch.float64).tiny)
    denom = max(1, int(stats.available.sum()))

    def objective(logits):
        w = torch.softmax(logits, dim=-1)
        av = torch.sum(w * alpha, dim=-1)
        bv = torch.sum(w * beta, dim=-1)
        residual = torch.einsum("qbi,qbij,qbj->qb", w.to(torch.complex128), gram,
                                w.to(torch.complex128)).real.clamp_min(0.0)
        risk = (
            config.lambda_hole * torch.relu(1.0 - torch.abs(av)).square()
            + config.lambda_voice * torch.abs(bv).square() * ev
            / (torch.abs(av).square() * ea + eps)
            + config.lambda_residual * residual / (ea + eps)
        )
        data = torch.sum(risk * available) / denom
        tvt = torch.sum(torch.abs(w[1:] - w[:-1]))
        tvf = torch.sum(torch.abs(w[:, 1:] - w[:, :-1]))
        total = data + 0.5 * (
            config.temporal_switch_penalty * tvt
            + config.frequency_switch_penalty * tvf
        ) / denom
        return total, w

    independent = np.argmin(stats.unary_risk, axis=-1)
    starts = {
        "uniform": None,
        "O1": np.full((q, b), int(o1_index), dtype=np.int64),
        "O2": np.asarray(o2_labels, dtype=np.int64),
        "independent": independent,
    }
    feasible: list[tuple[float, str, np.ndarray]] = []
    for name, labels in starts.items():
        initial = np.zeros((q, b, k), dtype=np.float64)
        if labels is not None:
            initial.fill(-config.o3_softmax_extent)
            np.put_along_axis(initial, labels[..., None], config.o3_softmax_extent, axis=-1)
        logits = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=config.o3_learning_rate)
        best_value = np.inf
        best_weights = None
        for _ in range(config.o3_iterations):
            optimizer.zero_grad(set_to_none=True)
            total, weights = objective(logits)
            if not torch.isfinite(total):
                raise RuntimeError("non-finite O3 objective")
            total.backward()
            optimizer.step()
            value = float(total.detach())
            if value < best_value:
                best_value = value
                best_weights = weights.detach().cpu().numpy()
        assert best_weights is not None
        feasible.append((_convex_objective_numpy(best_weights, stats, config)[0], name,
                         best_weights))

    o1 = np.zeros((q, b, k), dtype=np.float64); o1[..., o1_index] = 1.0
    o2 = np.zeros_like(o1); np.put_along_axis(o2, o2_labels[..., None], 1.0, axis=-1)
    feasible.extend(((_convex_objective_numpy(w, stats, config)[0], name, w)
                     for name, w in (("O1_exact", o1), ("O2_exact", o2))))
    _, name, weights = min(feasible, key=lambda item: item[0])
    total, data, tvt, tvf = _convex_objective_numpy(weights, stats, config)
    return ConvexRoutingResult(weights, total, data, tvt, tvf, name,
                               config.o3_iterations)


def one_hot_weights(labels: np.ndarray, candidates: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= candidates):
        raise ValueError("routing label outside candidate basis")
    result = np.zeros(labels.shape + (candidates,), dtype=np.float64)
    np.put_along_axis(result, labels[..., None], 1.0, axis=-1)
    return result


def render_spectral_route(
    candidate_spectra: np.ndarray, weights: np.ndarray, stats: CellStatistics,
    *, frames: int, config: RoutingConfig,
) -> np.ndarray:
    """Render exact-length stereo audio from cell weights via common STFT/ISTFT."""

    import librosa

    spectra = np.asarray(candidate_spectra)
    q, b, k = stats.unary_risk.shape
    w = np.asarray(weights, dtype=np.float64)
    if spectra.shape[0] != k or w.shape != (q, b, k):
        raise ValueError("spectra/weight grid does not match fitted statistics")
    if np.any(w < -1e-8) or not np.allclose(w.sum(-1), 1.0, atol=1e-6):
        raise ValueError("route weights are outside the simplex")
    weight_map = np.empty((k, spectra.shape[2], spectra.shape[3]), dtype=np.float64)
    for qi, (t0, t1) in enumerate(stats.time_frame_ranges):
        for bi, (f0, f1) in enumerate(stats.frequency_bin_ranges):
            weight_map[:, f0:f1, t0:t1] = w[qi, bi, :, None, None]
    combined = np.sum(spectra * weight_map[:, None], axis=0)
    channels = [
        librosa.istft(
            combined[channel], hop_length=config.hop_length, win_length=config.n_fft,
            window="hann", center=True, length=int(frames),
        )
        for channel in range(2)
    ]
    output = np.stack(channels, axis=1).astype(np.float32)
    if output.shape != (frames, 2) or not np.all(np.isfinite(output)):
        raise RuntimeError(f"invalid routed output {output.shape}")
    return output


def seam_check(output: np.ndarray, stats: CellStatistics, config: RoutingConfig) -> dict[str, Any]:
    """Measure sample discontinuities at route-tile boundaries."""

    y = _as_exact_stereo(output, "output")
    delta = np.max(np.abs(np.diff(y.astype(np.float64), axis=0)), axis=1)
    reference = max(float(np.percentile(delta, 99.0)), np.finfo(float).tiny)
    boundaries = [min(len(y) - 1, t1 * config.hop_length)
                  for _, t1 in stats.time_frame_ranges[:-1]]
    jumps = np.asarray([np.max(np.abs(y[index] - y[index - 1]))
                        for index in boundaries], dtype=np.float64)
    return {
        "boundary_count": len(boundaries),
        "p99_derivative": reference,
        "max_boundary_jump": float(jumps.max(initial=0.0)),
        "p99_boundary_jump": float(np.percentile(jumps, 99.0)) if len(jumps) else 0.0,
        "max_boundary_jump_over_p99_derivative": float(jumps.max(initial=0.0) / reference),
    }
