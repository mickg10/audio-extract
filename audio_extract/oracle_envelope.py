"""Exact-reference oracle envelopes over a fixed separator candidate basis.

This diagnostic answers whether existing separator outputs contain locally
complementary correct answers. It is not a deployment selector: it consumes the
clean accompaniment and voice references and therefore defines an oracle upper
bound for low-capacity routing/fusion.

O1 chooses one whole-work candidate.
O2 chooses one real candidate per time/frequency cell with Potts smoothness.
O3 chooses convex weights per cell with quadratic time/frequency smoothness.

All methods use the same local source-coordinate objective. O2/O3 reconstruct
real stereo FLOAT audio, which is then evaluated through the normal exact-label
and stereo metric paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import identity
from .judge_labels import fit_source_coordinates, local_source_coordinate_labels
from .judge_train import label_targets
from .metrics_v2 import brightness_v2, fullness_v2, stereo_v2, transient_v2

_EPS = 1e-12


class OracleEnvelopeError(RuntimeError):
    """The exact-reference envelope input or optimization is invalid."""


@dataclass(frozen=True)
class OracleEnvelopeConfig:
    n_fft: int = 2048
    hop_length: int = 512
    time_cell_seconds: float = 0.5
    band_edges_hz: tuple[float, ...] = (
        0.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0,
        8000.0, 12000.0, 16000.0, 22050.0,
    )
    alpha_weight: float = 2.0
    voice_weight: float = 1.0
    artifact_weight: float = 0.5
    fallback_direct_weight: float = 1.0
    ridge_relative: float = 1e-8
    max_condition: float = 1e6
    min_source_energy: float = 1e-10
    energy_floor_db_below_peak: float = 60.0
    max_stft_roundtrip_abs: float = 2e-5
    temporal_switch_penalty: float = 0.08
    frequency_switch_penalty: float = 0.04
    temporal_weight_smoothness: float = 0.08
    frequency_weight_smoothness: float = 0.04
    medoid_max_sweeps: int = 20
    convex_max_iterations: int = 1500
    convex_tolerance: float = 1e-7

    def validate(self, sr: int) -> None:
        if self.n_fft < 16 or self.hop_length <= 0 or self.hop_length > self.n_fft:
            raise ValueError("invalid STFT geometry")
        if self.time_cell_seconds <= 0:
            raise ValueError("time_cell_seconds must be positive")
        edges = np.asarray(self.band_edges_hz, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("band_edges_hz must be strictly increasing")
        if edges[0] != 0.0 or edges[-1] < sr / 2:
            raise ValueError("band_edges_hz must cover [0, Nyquist]")
        for name in (
            "alpha_weight", "voice_weight", "artifact_weight",
            "fallback_direct_weight", "temporal_switch_penalty",
            "frequency_switch_penalty", "temporal_weight_smoothness",
            "frequency_weight_smoothness",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.ridge_relative < 0 or self.max_condition <= 1 or self.min_source_energy < 0:
            raise ValueError("invalid source-coordinate thresholds")
        if self.energy_floor_db_below_peak <= 0 or self.max_stft_roundtrip_abs <= 0:
            raise ValueError("energy floor and roundtrip threshold must be positive")


@dataclass(frozen=True)
class BasisRow:
    work_id: str
    candidate_id: str
    path: str
    artifact_pcm_sha256: str | None = None
    container_sha256: str | None = None


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _resolve_host_path(value: str) -> Path:
    direct = Path(value)
    if direct.exists():
        return direct
    if ":" in value:
        _, suffix = value.split(":", 1)
        candidate = Path(suffix)
        if candidate.is_absolute() and candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def load_basis_manifest(path: Path, work_id: str | None = None) -> list[BasisRow]:
    rows: list[BasisRow] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        row_work = str(raw["work_id"])
        if work_id is not None and row_work != work_id:
            continue
        candidate_id = str(
            raw.get("candidate_id") or raw.get("candidate") or raw.get("recipe_id")
        )
        rows.append(BasisRow(
            work_id=row_work,
            candidate_id=candidate_id,
            path=str(raw["path"]),
            artifact_pcm_sha256=raw.get("artifact_pcm_sha256"),
            container_sha256=raw.get("container_sha256"),
        ))
    if not rows:
        raise OracleEnvelopeError(f"no basis rows selected from {path}")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.work_id, row.candidate_id)
        if key in seen:
            raise OracleEnvelopeError(f"duplicate basis row {key}")
        seen.add(key)
    return rows


def _read_float_stereo(path: Path, *, expected_sr: int | None = None,
                       expected_frames: int | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    if info.channels != 2 or info.subtype != "FLOAT":
        raise OracleEnvelopeError(f"expected stereo FLOAT WAV: {path} ({info})")
    if expected_sr is not None and info.samplerate != expected_sr:
        raise OracleEnvelopeError(f"sample-rate mismatch for {path}")
    if expected_frames is not None and info.frames != expected_frames:
        raise OracleEnvelopeError(f"frame mismatch for {path}")
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.all(np.isfinite(audio)):
        raise OracleEnvelopeError(f"non-finite audio: {path}")
    return audio, int(sr)


def load_work_arrays(rows: list[BasisRow], truth_root: Path, work_id: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[BasisRow]]:
    selected = [row for row in rows if row.work_id == work_id]
    if len(selected) < 2:
        raise OracleEnvelopeError(f"{work_id}: need at least two basis candidates")
    truth = truth_root / work_id
    mixture, sr = _read_float_stereo(truth / "mix_with_voice.wav")
    accompaniment, _ = _read_float_stereo(
        truth / "orchestra_only.wav", expected_sr=sr, expected_frames=len(mixture)
    )
    vocal, _ = _read_float_stereo(
        truth / "voice_ref.wav", expected_sr=sr, expected_frames=len(mixture)
    )
    residual = mixture.astype(np.float64) - accompaniment.astype(np.float64) - vocal.astype(np.float64)
    if np.max(np.abs(residual)) > 2e-5:
        raise OracleEnvelopeError(
            f"{work_id}: truth identity M=A+V failed max_abs={np.max(np.abs(residual))}"
        )
    candidates = []
    for row in selected:
        candidate_path = _resolve_host_path(row.path)
        candidate, _ = _read_float_stereo(
            candidate_path, expected_sr=sr, expected_frames=len(mixture)
        )
        if row.container_sha256 and _sha_file(candidate_path) != row.container_sha256:
            raise OracleEnvelopeError(f"{work_id}/{row.candidate_id}: container hash mismatch")
        if row.artifact_pcm_sha256:
            pcm = identity.artifact_pcm_sha256(candidate, sr, ["FL", "FR"], len(candidate))
            if pcm != row.artifact_pcm_sha256:
                raise OracleEnvelopeError(f"{work_id}/{row.candidate_id}: PCM hash mismatch")
        candidates.append(candidate)
    return mixture, accompaniment, vocal, np.stack(candidates).astype(np.float32), sr, selected


def _stft_stereo(audio: np.ndarray, cfg: OracleEnvelopeConfig) -> np.ndarray:
    import librosa
    return np.stack([
        librosa.stft(
            audio[:, channel], n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            win_length=cfg.n_fft, window="hann", center=True,
        )
        for channel in range(audio.shape[1])
    ], axis=0)


def _istft_stereo(spec: np.ndarray, frames: int, cfg: OracleEnvelopeConfig) -> np.ndarray:
    import librosa
    audio = np.column_stack([
        librosa.istft(
            spec[channel], hop_length=cfg.hop_length, win_length=cfg.n_fft,
            window="hann", center=True, length=frames,
        )
        for channel in range(spec.shape[0])
    ])
    if audio.shape != (frames, 2) or not np.all(np.isfinite(audio)):
        raise OracleEnvelopeError(f"invalid reconstructed oracle output: {audio.shape}")
    return audio.astype(np.float32)


def _band_bin_slices(sr: int, cfg: OracleEnvelopeConfig) -> list[slice]:
    freqs = np.fft.rfftfreq(cfg.n_fft, d=1.0 / sr)
    edges = np.asarray(cfg.band_edges_hz, dtype=np.float64)
    result = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        start = int(np.searchsorted(freqs, lo, side="left"))
        side = "right" if index == len(edges) - 2 else "left"
        stop = int(np.searchsorted(freqs, hi, side=side))
        stop = max(stop, start + 1)
        result.append(slice(start, min(stop, len(freqs))))
    return result


def _time_slices(n_stft_frames: int, sr: int, cfg: OracleEnvelopeConfig) -> list[slice]:
    frames_per_cell = max(1, int(round(cfg.time_cell_seconds * sr / cfg.hop_length)))
    return [
        slice(start, min(start + frames_per_cell, n_stft_frames))
        for start in range(0, n_stft_frames, frames_per_cell)
    ]


def source_coordinate_quadratics(
    candidate_specs: np.ndarray,
    accompaniment_spec: np.ndarray,
    vocal_spec: np.ndarray,
    sr: int,
    cfg: OracleEnvelopeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return PSD local costs ``w'Gw - 2c'w + constant``."""
    cfg.validate(sr)
    if candidate_specs.ndim != 4 or candidate_specs.shape[1:] != accompaniment_spec.shape:
        raise OracleEnvelopeError("candidate/target STFT shape mismatch")
    if vocal_spec.shape != accompaniment_spec.shape:
        raise OracleEnvelopeError("vocal/accompaniment STFT shape mismatch")
    K = candidate_specs.shape[0]
    time_slices = _time_slices(accompaniment_spec.shape[-1], sr, cfg)
    band_slices = _band_bin_slices(sr, cfg)
    G = np.zeros((len(time_slices), len(band_slices), K, K), dtype=np.float64)
    c = np.zeros((len(time_slices), len(band_slices), K), dtype=np.float64)
    unary = np.zeros((len(time_slices), len(band_slices), K), dtype=np.float64)
    available = np.zeros((len(time_slices), len(band_slices)), dtype=bool)
    condition = np.full((len(time_slices), len(band_slices)), np.nan)
    modes: list[list[str]] = [["" for _ in band_slices] for _ in time_slices]

    accompaniment_energies = np.asarray([
        float(np.real(np.vdot(
            accompaniment_spec[:, bs, ts].reshape(-1),
            accompaniment_spec[:, bs, ts].reshape(-1),
        )))
        for ts in time_slices for bs in band_slices
    ], dtype=np.float64)
    peak_accompaniment_energy = float(accompaniment_energies.max(initial=0.0))
    relative_floor = (
        peak_accompaniment_energy
        * 10.0 ** (-cfg.energy_floor_db_below_peak / 10.0)
    )
    absolute_energy_floor = max(float(cfg.min_source_energy), relative_floor)

    for ti, ts in enumerate(time_slices):
        for bi, bs in enumerate(band_slices):
            a = np.moveaxis(accompaniment_spec[:, bs, ts], 0, -1)
            v = np.moveaxis(vocal_spec[:, bs, ts], 0, -1)
            ys = [np.moveaxis(candidate_specs[k, :, bs, ts], 0, -1) for k in range(K)]
            labels = [
                fit_source_coordinates(
                    y, a, v, ridge_relative=cfg.ridge_relative,
                    max_condition=cfg.max_condition,
                    min_source_energy=absolute_energy_floor,
                )
                for y in ys
            ]
            if all(label.available for label in labels):
                alpha = np.asarray([
                    complex(label.alpha_real, label.alpha_imag) for label in labels
                ])
                beta = np.asarray([
                    complex(label.beta_real, label.beta_imag) for label in labels
                ])
                af = a.reshape(-1).astype(np.complex128)
                vf = v.reshape(-1).astype(np.complex128)
                ea = max(float(np.real(np.vdot(af, af))), _EPS)
                ev = max(float(np.real(np.vdot(vf, vf))), 0.0)
                residuals = []
                for k, y in enumerate(ys):
                    yf = y.reshape(-1).astype(np.complex128)
                    residuals.append(yf - alpha[k] * af - beta[k] * vf)
                R = np.stack(residuals)
                gram_alpha = np.real(np.outer(np.conj(alpha), alpha))
                gram_beta = np.real(np.outer(np.conj(beta), beta))
                gram_residual = np.real(R.conj() @ R.T) / ea
                local_G = (
                    cfg.alpha_weight * gram_alpha
                    + cfg.voice_weight * (ev / ea) * gram_beta
                    + cfg.artifact_weight * gram_residual
                )
                local_c = cfg.alpha_weight * np.real(alpha)
                local_unary = np.diag(local_G) - 2.0 * local_c + cfg.alpha_weight
                available[ti, bi] = True
                condition[ti, bi] = max(
                    float(label.condition_number or np.nan) for label in labels
                )
                modes[ti][bi] = "source_coordinates"
            else:
                errors = np.stack([(y - a).reshape(-1).astype(np.complex128) for y in ys])
                af = a.reshape(-1).astype(np.complex128)
                scale = max(float(np.real(np.vdot(af, af))), absolute_energy_floor)
                local_G = (
                    cfg.fallback_direct_weight
                    * np.real(errors.conj() @ errors.T) / scale
                )
                local_c = np.zeros(K, dtype=np.float64)
                local_unary = np.diag(local_G)
                modes[ti][bi] = "direct_fallback"
            local_G = 0.5 * (local_G + local_G.T)
            eig_min = float(np.linalg.eigvalsh(local_G).min())
            if eig_min < 0:
                local_G += np.eye(K) * (-eig_min + 1e-12)
            G[ti, bi] = local_G
            c[ti, bi] = local_c
            unary[ti, bi] = local_unary
    info = {
        "time_slices": [(s.start, s.stop) for s in time_slices],
        "band_bin_slices": [(s.start, s.stop) for s in band_slices],
        "source_coordinate_cells": int(available.sum()),
        "direct_fallback_cells": int(available.size - available.sum()),
        "condition_numbers": condition.tolist(),
        "cell_modes": modes,
        "peak_accompaniment_cell_energy": peak_accompaniment_energy,
        "absolute_energy_floor": absolute_energy_floor,
        "energy_floor_db_below_peak": cfg.energy_floor_db_below_peak,
    }
    return G, c, unary, info


def project_simplex(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 1:
        raise ValueError("simplex input must have a final candidate axis")
    K = x.shape[-1]
    flat = x.reshape(-1, K)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cssv = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, K + 1, dtype=np.float64)
    positive = ordered - cssv / ranks > 0
    rho = positive.sum(axis=1) - 1
    theta = cssv[np.arange(len(flat)), rho] / (rho + 1)
    projected = np.maximum(flat - theta[:, None], 0.0)
    return projected.reshape(x.shape)


def _viterbi(unary: np.ndarray, switch_penalty: float) -> np.ndarray:
    T, K = unary.shape
    dp = np.empty((T, K), dtype=np.float64)
    back = np.empty((T, K), dtype=np.int32)
    dp[0] = unary[0]
    back[0] = -1
    transition = switch_penalty * (
        np.arange(K)[:, None] != np.arange(K)[None, :]
    )
    for t in range(1, T):
        costs = dp[t - 1, :, None] + transition
        back[t] = np.argmin(costs, axis=0)
        dp[t] = unary[t] + costs[back[t], np.arange(K)]
    labels = np.empty(T, dtype=np.int32)
    labels[-1] = int(np.argmin(dp[-1]))
    for t in range(T - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels


def _medoid_energy(labels: np.ndarray, unary: np.ndarray,
                   temporal: float, frequency: float) -> float:
    T, B = labels.shape
    energy = float(
        unary[np.arange(T)[:, None], np.arange(B)[None, :], labels].sum()
    )
    energy += temporal * float(np.count_nonzero(labels[1:] != labels[:-1]))
    energy += frequency * float(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    return energy


def smooth_medoid_labels(unary: np.ndarray, cfg: OracleEnvelopeConfig) -> tuple[np.ndarray, dict]:
    if unary.ndim != 3:
        raise ValueError("unary must be (time, band, candidate)")
    T, B, K = unary.shape
    starts = [np.argmin(unary, axis=-1)]
    starts.extend(np.full((T, B), k, dtype=np.int32) for k in range(K))
    best_labels = None
    best_energy = math.inf
    best_sweeps = 0
    for initial in starts:
        labels = initial.copy()
        for sweep in range(cfg.medoid_max_sweeps):
            before = labels.copy()
            for bands in (range(B), range(B - 1, -1, -1)):
                for b in bands:
                    local = unary[:, b].copy()
                    if b > 0:
                        local += cfg.frequency_switch_penalty * (
                            np.arange(K)[None, :] != labels[:, b - 1, None]
                        )
                    if b + 1 < B:
                        local += cfg.frequency_switch_penalty * (
                            np.arange(K)[None, :] != labels[:, b + 1, None]
                        )
                    labels[:, b] = _viterbi(local, cfg.temporal_switch_penalty)
            if np.array_equal(labels, before):
                break
        energy = _medoid_energy(
            labels, unary, cfg.temporal_switch_penalty,
            cfg.frequency_switch_penalty,
        )
        if energy < best_energy:
            best_energy = energy
            best_labels = labels.copy()
            best_sweeps = sweep + 1
    assert best_labels is not None
    return best_labels, {
        "objective": best_energy,
        "sweeps": best_sweeps,
        "temporal_switches": int(np.count_nonzero(best_labels[1:] != best_labels[:-1])),
        "frequency_switches": int(np.count_nonzero(best_labels[:, 1:] != best_labels[:, :-1])),
        "occupancy": [int(np.count_nonzero(best_labels == k)) for k in range(K)],
    }


def _convex_objective(weights: np.ndarray, G: np.ndarray, c: np.ndarray,
                      temporal: float, frequency: float) -> float:
    value = float(
        np.einsum("tbk,tbkl,tbl->", weights, G, weights)
        - 2.0 * np.einsum("tbk,tbk->", c, weights)
    )
    if temporal:
        value += temporal * float(np.square(weights[1:] - weights[:-1]).sum())
    if frequency:
        value += frequency * float(np.square(weights[:, 1:] - weights[:, :-1]).sum())
    return value


def smooth_convex_weights(G: np.ndarray, c: np.ndarray,
                          cfg: OracleEnvelopeConfig) -> tuple[np.ndarray, dict]:
    if G.ndim != 4 or c.shape != G.shape[:3]:
        raise ValueError("G/c shapes must be (time,band,K,K)/(time,band,K)")
    T, B, K = c.shape
    unary = np.diagonal(G, axis1=-2, axis2=-1) - 2.0 * c
    labels = np.argmin(unary, axis=-1)
    weights = np.eye(K, dtype=np.float64)[labels]

    max_eigenvalue = max(
        float(np.linalg.eigvalsh(G[t, b]).max())
        for t in range(T) for b in range(B)
    )
    lipschitz = (
        2.0 * max_eigenvalue
        + 8.0 * cfg.temporal_weight_smoothness
        + 8.0 * cfg.frequency_weight_smoothness
        + 1e-12
    )
    step = 1.0 / lipschitz
    objective = _convex_objective(
        weights, G, c, cfg.temporal_weight_smoothness,
        cfg.frequency_weight_smoothness,
    )
    iterations = 0
    for iterations in range(1, cfg.convex_max_iterations + 1):
        gradient = 2.0 * np.einsum("tbkl,tbl->tbk", G, weights) - 2.0 * c
        if cfg.temporal_weight_smoothness:
            gradient[1:] += (
                2.0 * cfg.temporal_weight_smoothness
                * (weights[1:] - weights[:-1])
            )
            gradient[:-1] += (
                2.0 * cfg.temporal_weight_smoothness
                * (weights[:-1] - weights[1:])
            )
        if cfg.frequency_weight_smoothness:
            gradient[:, 1:] += (
                2.0 * cfg.frequency_weight_smoothness
                * (weights[:, 1:] - weights[:, :-1])
            )
            gradient[:, :-1] += (
                2.0 * cfg.frequency_weight_smoothness
                * (weights[:, :-1] - weights[:, 1:])
            )
        proposal = project_simplex(weights - step * gradient)
        maximum_change = float(np.max(np.abs(proposal - weights)))
        weights = proposal
        if maximum_change <= cfg.convex_tolerance:
            break
    final_objective = _convex_objective(
        weights, G, c, cfg.temporal_weight_smoothness,
        cfg.frequency_weight_smoothness,
    )
    if final_objective > objective + 1e-7:
        raise OracleEnvelopeError(
            f"convex optimization increased objective {objective} -> {final_objective}"
        )
    return weights, {
        "objective": final_objective,
        "initial_objective": objective,
        "iterations": iterations,
        "max_simplex_error": float(np.max(np.abs(weights.sum(axis=-1) - 1.0))),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "temporal_total_variation_l2": float(np.sqrt(np.square(weights[1:] - weights[:-1]).sum())),
        "frequency_total_variation_l2": float(np.sqrt(np.square(weights[:, 1:] - weights[:, :-1]).sum())),
        "mean_weights": weights.mean(axis=(0, 1)).tolist(),
    }


def reconstruct_medoid(candidate_specs: np.ndarray, labels: np.ndarray,
                       info: dict) -> np.ndarray:
    K, C, F, N = candidate_specs.shape
    output = np.empty((C, F, N), dtype=candidate_specs.dtype)
    for ti, (t0, t1) in enumerate(info["time_slices"]):
        for bi, (f0, f1) in enumerate(info["band_bin_slices"]):
            label = int(labels[ti, bi])
            if not 0 <= label < K:
                raise OracleEnvelopeError("medoid label outside candidate set")
            output[:, f0:f1, t0:t1] = candidate_specs[label, :, f0:f1, t0:t1]
    return output


def reconstruct_convex(candidate_specs: np.ndarray, weights: np.ndarray,
                       info: dict) -> np.ndarray:
    K, C, F, N = candidate_specs.shape
    output = np.empty((C, F, N), dtype=candidate_specs.dtype)
    for ti, (t0, t1) in enumerate(info["time_slices"]):
        for bi, (f0, f1) in enumerate(info["band_bin_slices"]):
            local = weights[ti, bi]
            output[:, f0:f1, t0:t1] = np.tensordot(
                local, candidate_specs[:, :, f0:f1, t0:t1], axes=(0, 0)
            )
    return output


def exact_metrics(candidate: np.ndarray, accompaniment: np.ndarray,
                  vocal: np.ndarray, sr: int) -> dict:
    if candidate.shape != accompaniment.shape or candidate.shape != vocal.shape:
        raise OracleEnvelopeError("metric arrays must share exact sample grid")
    labels = local_source_coordinate_labels(
        candidate, accompaniment, vocal,
        tile_frames=max(256, round(0.5 * sr)),
        hop_frames=max(128, round(0.25 * sr)),
    )
    result = label_targets(labels)
    error = candidate.astype(np.float64) - accompaniment.astype(np.float64)
    signal_energy = float(np.square(accompaniment.astype(np.float64)).sum())
    error_energy = float(np.square(error).sum())
    result["scale_dependent_sdr_db"] = float(
        10.0 * np.log10((signal_energy + _EPS) / (error_energy + _EPS))
    )
    for observation in (
        fullness_v2(candidate, accompaniment, sr)
        + brightness_v2(candidate, accompaniment, sr)
        + transient_v2(candidate, accompaniment, sr)
        + stereo_v2(candidate, accompaniment)
    ):
        result[observation["metric"]] = observation["value"] if observation["available"] else None
    return result


def _write_float_verified(path: Path, audio: np.ndarray, sr: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        reopened, reopened_sr = _read_float_stereo(
            path, expected_sr=sr, expected_frames=len(audio)
        )
        if not np.array_equal(reopened, audio.astype(np.float32)):
            raise OracleEnvelopeError(f"refusing to replace different output: {path}")
    else:
        sf.write(path, audio.astype(np.float32), sr, subtype="FLOAT")
        reopened, reopened_sr = _read_float_stereo(
            path, expected_sr=sr, expected_frames=len(audio)
        )
        if not np.array_equal(reopened, audio.astype(np.float32)):
            raise OracleEnvelopeError(f"FLOAT reopen mismatch: {path}")
    return {
        "path": str(path),
        "container_sha256": _sha_file(path),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            reopened, reopened_sr, ["FL", "FR"], len(reopened)
        ),
        "frames": len(reopened),
        "sample_rate_hz": reopened_sr,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
    }


def run_work(rows: list[BasisRow], truth_root: Path, work_id: str,
             output_root: Path, cfg: OracleEnvelopeConfig) -> dict:
    mixture, accompaniment, vocal, candidates, sr, selected = load_work_arrays(
        rows, truth_root, work_id
    )
    cfg.validate(sr)
    candidate_specs = np.stack([_stft_stereo(y, cfg) for y in candidates])
    roundtrip = []
    for index, spec in enumerate(candidate_specs):
        reconstructed = _istft_stereo(spec, len(mixture), cfg)
        difference = reconstructed.astype(np.float64) - candidates[index].astype(np.float64)
        facts = {
            "candidate_id": selected[index].candidate_id,
            "max_abs": float(np.max(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(np.square(difference)))),
        }
        if facts["max_abs"] > cfg.max_stft_roundtrip_abs:
            raise OracleEnvelopeError(
                f"STFT roundtrip exceeds threshold for {selected[index].candidate_id}: {facts}"
            )
        roundtrip.append(facts)
    accompaniment_spec = _stft_stereo(accompaniment, cfg)
    vocal_spec = _stft_stereo(vocal, cfg)
    G, c, unary, info = source_coordinate_quadratics(
        candidate_specs, accompaniment_spec, vocal_spec, sr, cfg
    )

    basis_metrics = {
        selected[index].candidate_id: exact_metrics(
            candidates[index].astype(np.float32), accompaniment, vocal, sr
        )
        for index in range(len(selected))
    }
    global_costs = unary.sum(axis=(0, 1))
    o1_index = int(np.argmin(global_costs))
    o1_audio = candidates[o1_index].astype(np.float32)

    labels, o2_optimization = smooth_medoid_labels(unary, cfg)
    o2_audio = _istft_stereo(
        reconstruct_medoid(candidate_specs, labels, info), len(mixture), cfg
    )
    weights, o3_optimization = smooth_convex_weights(G, c, cfg)
    o3_audio = _istft_stereo(
        reconstruct_convex(candidate_specs, weights, info), len(mixture), cfg
    )

    work_dir = output_root / work_id
    methods = {
        "O1_whole_track_single": (
            o1_audio,
            {
                "selected_candidate_index": o1_index,
                "selected_candidate_id": selected[o1_index].candidate_id,
                "candidate_objectives": global_costs.tolist(),
            },
        ),
        "O2_smooth_tilewise_medoid": (
            o2_audio,
            {
                **o2_optimization,
                "candidate_ids": [row.candidate_id for row in selected],
                "labels": labels.tolist(),
            },
        ),
        "O3_smooth_convex_hull": (
            o3_audio,
            {
                **o3_optimization,
                "candidate_ids": [row.candidate_id for row in selected],
                "weights": weights.tolist(),
            },
        ),
    }
    outputs = {}
    for method, (audio, optimization) in methods.items():
        method_dir = work_dir / method
        artifact = _write_float_verified(
            method_dir / "accompaniment.f32.wav", audio, sr
        )
        removed_artifact = _write_float_verified(
            method_dir / "removed-vocal.f32.wav",
            mixture.astype(np.float32) - audio.astype(np.float32), sr,
        )
        outputs[method] = {
            "artifact": artifact,
            "removed_vocal_artifact": removed_artifact,
            "optimization": optimization,
            "metrics": exact_metrics(audio, accompaniment, vocal, sr),
        }
    report = {
        "schema": "audio-extract/oracle-envelope/v1",
        "diagnostic_only": True,
        "work_id": work_id,
        "config": asdict(cfg),
        "basis": [
            {
                **asdict(row),
                "resolved_path": str(_resolve_host_path(row.path)),
                "metrics": basis_metrics[row.candidate_id],
            }
            for row in selected
        ],
        "cell_info": info,
        "stft_roundtrip": roundtrip,
        "methods": outputs,
    }
    report_path = work_dir / "oracle-envelope-report.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report_path.exists() and report_path.read_text() != payload:
        raise OracleEnvelopeError(f"refusing to rewrite differing oracle report: {report_path}")
    if not report_path.exists():
        report_path.write_text(payload)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("audio-extract oracle-envelope")
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-id", action="append", dest="work_ids")
    parser.add_argument("--config-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = OracleEnvelopeConfig()
    if args.config_json:
        config = OracleEnvelopeConfig(**json.loads(args.config_json.read_text()))
    rows = load_basis_manifest(args.basis_manifest)
    works = args.work_ids or sorted({row.work_id for row in rows})
    for work in works:
        run_work(rows, args.truth_root, work, args.output_root, config)
    summary = {
        "schema": "audio-extract/oracle-envelope-run/v1",
        "diagnostic_only": True,
        "works": works,
        "reports": [
            str(args.output_root / work / "oracle-envelope-report.json")
            for work in works
        ],
        "methods": [
            "O1_whole_track_single",
            "O2_smooth_tilewise_medoid",
            "O3_smooth_convex_hull",
        ],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
