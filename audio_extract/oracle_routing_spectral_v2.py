"""Exact spectral adapter for the certified oracle-routing v2 mathematics.

This module owns the common stereo STFT grid, complete time/band coverage,
source-coordinate/direct-fallback cell construction, and shared-stereo route
reconstruction. Storage and experiment orchestration remain in the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .oracle_routing_math_v2 import (
    QuadraticGrid,
    build_exact_cell_quadratic,
    stack_cells,
)


@dataclass(frozen=True)
class RoutingSpectralConfig:
    sample_rate_hz: int = 44_100
    n_fft: int = 2048
    hop_length: int = 1024
    time_cell_seconds: float = 2.0
    band_edges_hz: tuple[int, ...] = (
        0, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 22_050
    )
    alpha_weight: float = 1.0
    voice_weight: float = 1.0
    artifact_weight: float = 1.0
    direct_weight: float = 1.0
    ridge_relative: float = 1e-8
    max_condition: float = 1e6
    min_source_power_relative: float = 1e-6
    reference_floor_relative: float = 1e-8

    def validate(self) -> None:
        if self.sample_rate_hz < 2 or self.n_fft < 16:
            raise ValueError("invalid sample-rate/STFT size")
        if self.hop_length < 1 or self.hop_length > self.n_fft:
            raise ValueError("invalid STFT hop")
        if not math.isfinite(self.time_cell_seconds) or self.time_cell_seconds <= 0:
            raise ValueError("time_cell_seconds must be positive")
        edges = tuple(int(value) for value in self.band_edges_hz)
        if len(edges) < 2 or edges[0] != 0:
            raise ValueError("band edges must start at zero")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("band edges must be strictly increasing")
        if edges[-1] != self.sample_rate_hz // 2:
            raise ValueError("last band edge must equal Nyquist")
        for name in (
            "alpha_weight", "voice_weight", "artifact_weight", "direct_weight",
            "ridge_relative", "min_source_power_relative",
            "reference_floor_relative",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_condition <= 1 or not math.isfinite(self.max_condition):
            raise ValueError("max_condition must be finite and greater than one")
        if self.reference_floor_relative <= 0:
            raise ValueError("reference_floor_relative must be positive")


def _audio(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] < 1:
        raise ValueError(f"{name} must have shape (frames,2)")
    if not np.issubdtype(result.dtype, np.floating):
        raise ValueError(f"{name} must contain floating samples")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite samples")
    return np.asarray(result, dtype=np.float32)


def validate_exact_audio_basis(
    candidates: list[np.ndarray] | tuple[np.ndarray, ...],
    accompaniment: np.ndarray,
    vocal: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    accompaniment = _audio(accompaniment, "accompaniment")
    vocal = _audio(vocal, "vocal")
    if accompaniment.shape != vocal.shape:
        raise ValueError("accompaniment/vocal grids differ")
    values = [_audio(value, f"candidate[{index}]")
              for index, value in enumerate(candidates)]
    if len(values) < 2:
        raise ValueError("at least two candidates are required")
    if any(value.shape != accompaniment.shape for value in values):
        raise ValueError("candidate/truth grids differ")
    return values, accompaniment, vocal


def stft_stack(
    signals: list[np.ndarray] | tuple[np.ndarray, ...],
    config: RoutingSpectralConfig,
) -> np.ndarray:
    """Return complex64 spectra `(members,channels,frequency,time)`."""

    import librosa

    config.validate()
    values = [_audio(value, f"signal[{index}]")
              for index, value in enumerate(signals)]
    frames = len(values[0])
    if any(len(value) != frames for value in values):
        raise ValueError("signal frame counts differ")
    result = np.empty(
        (len(values), 2, config.n_fft // 2 + 1,
         1 + frames // config.hop_length),
        dtype=np.complex64,
    )
    # Librosa's centered frame count can differ at exact boundaries; build one
    # member first and allocate from the observed shape instead.
    first = np.stack([
        librosa.stft(
            values[0][:, channel], n_fft=config.n_fft,
            hop_length=config.hop_length, win_length=config.n_fft,
            window="hann", center=True, pad_mode="constant",
        )
        for channel in range(2)
    ], axis=0).astype(np.complex64)
    result = np.empty((len(values),) + first.shape, dtype=np.complex64)
    result[0] = first
    for member, value in enumerate(values[1:], start=1):
        spectrum = np.stack([
            librosa.stft(
                value[:, channel], n_fft=config.n_fft,
                hop_length=config.hop_length, win_length=config.n_fft,
                window="hann", center=True, pad_mode="constant",
            )
            for channel in range(2)
        ], axis=0).astype(np.complex64)
        if spectrum.shape != first.shape:
            raise RuntimeError("STFT grids differ for exact-length signals")
        result[member] = spectrum
    return result


def time_frame_ranges(
    stft_frames: int, config: RoutingSpectralConfig
) -> tuple[tuple[int, int], ...]:
    config.validate()
    if stft_frames < 1:
        raise ValueError("STFT time grid is empty")
    per_cell = max(
        1,
        round(
            config.time_cell_seconds
            * config.sample_rate_hz / config.hop_length
        ),
    )
    return tuple(
        (start, min(start + per_cell, stft_frames))
        for start in range(0, stft_frames, per_cell)
    )


def frequency_bin_ranges(
    frequency_bins: int, config: RoutingSpectralConfig
) -> tuple[tuple[int, int], ...]:
    """Return contiguous non-empty bands covering every rFFT bin once."""

    config.validate()
    frequencies = np.fft.rfftfreq(
        config.n_fft, d=1.0 / config.sample_rate_hz
    )
    if frequency_bins != len(frequencies):
        raise ValueError("frequency grid does not match configured rFFT")
    boundaries = [0]
    for edge in config.band_edges_hz[1:-1]:
        boundaries.append(int(np.searchsorted(frequencies, edge, side="left")))
    boundaries.append(frequency_bins)
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("configured frequency edges create an empty band")
    ranges = tuple(zip(boundaries, boundaries[1:]))
    covered = np.concatenate([
        np.arange(start, end, dtype=np.int64) for start, end in ranges
    ])
    if not np.array_equal(covered, np.arange(frequency_bins)):
        raise RuntimeError("frequency bands do not cover each bin exactly once")
    return ranges


def build_quadratic_grid(
    candidate_spectra: np.ndarray,
    accompaniment_spectrum: np.ndarray,
    vocal_spectrum: np.ndarray,
    config: RoutingSpectralConfig,
) -> tuple[QuadraticGrid, dict]:
    """Build complete certified cell costs on one exact spectral grid."""

    config.validate()
    candidates = np.asarray(candidate_spectra)
    accompaniment = np.asarray(accompaniment_spectrum)
    vocal = np.asarray(vocal_spectrum)
    if candidates.ndim != 4 or candidates.shape[1] != 2:
        raise ValueError("candidate_spectra must be (K,2,F,T)")
    if accompaniment.shape != candidates.shape[1:] or vocal.shape != accompaniment.shape:
        raise ValueError("candidate/truth spectral grids differ")
    if len(candidates) < 2:
        raise ValueError("at least two candidate spectra are required")
    if not (
        np.all(np.isfinite(candidates))
        and np.all(np.isfinite(accompaniment))
        and np.all(np.isfinite(vocal))
    ):
        raise ValueError("spectra contain non-finite values")

    times = time_frame_ranges(candidates.shape[-1], config)
    bands = frequency_bin_ranges(candidates.shape[-2], config)
    global_a_power = float(np.mean(np.abs(accompaniment) ** 2))
    global_v_power = float(np.mean(np.abs(vocal) ** 2))
    global_power = max(global_a_power + global_v_power, np.finfo(float).tiny)
    cells = []
    mode_counts: dict[str, int] = {}
    mode_energy: dict[str, float] = {}
    for time_start, time_end in times:
        row = []
        for frequency_start, frequency_end in bands:
            a = accompaniment[
                :, frequency_start:frequency_end, time_start:time_end
            ].reshape(-1)
            v = vocal[
                :, frequency_start:frequency_end, time_start:time_end
            ].reshape(-1)
            ys = candidates[
                :, :, frequency_start:frequency_end, time_start:time_end
            ].reshape(len(candidates), -1)
            count = len(a)
            cell = build_exact_cell_quadratic(
                ys, a, v,
                alpha_weight=config.alpha_weight,
                voice_weight=config.voice_weight,
                artifact_weight=config.artifact_weight,
                direct_weight=config.direct_weight,
                ridge_relative=config.ridge_relative,
                max_condition=config.max_condition,
                accompaniment_floor=(
                    config.min_source_power_relative * global_a_power * count
                ),
                vocal_floor=(
                    config.min_source_power_relative * global_v_power * count
                ),
                reference_floor=(
                    config.reference_floor_relative * global_power * count
                ),
            )
            row.append(cell)
            mode_counts[cell.mode] = mode_counts.get(cell.mode, 0) + 1
            energy = cell.accompaniment_energy + cell.vocal_energy
            mode_energy[cell.mode] = mode_energy.get(cell.mode, 0.0) + energy
        cells.append(row)
    grid = stack_cells(cells)
    report = {
        "time_frame_ranges": [list(value) for value in times],
        "frequency_bin_ranges": [list(value) for value in bands],
        "mode_counts": mode_counts,
        "mode_energy": mode_energy,
        "total_cells": len(times) * len(bands),
        "global_accompaniment_power": global_a_power,
        "global_vocal_power": global_v_power,
    }
    return grid, report


def render_spectral_route(
    candidate_spectra: np.ndarray,
    weights: np.ndarray,
    cell_report: dict,
    *,
    frames: int,
    config: RoutingSpectralConfig,
) -> np.ndarray:
    """Render exact-length stereo audio using one weight vector for both channels."""

    import librosa

    spectra = np.asarray(candidate_spectra)
    weights = np.asarray(weights, dtype=np.float64)
    times = tuple(tuple(value) for value in cell_report["time_frame_ranges"])
    bands = tuple(tuple(value) for value in cell_report["frequency_bin_ranges"])
    if spectra.ndim != 4 or spectra.shape[1] != 2:
        raise ValueError("candidate spectra must be (K,2,F,T)")
    if weights.shape != (len(times), len(bands), len(spectra)):
        raise ValueError("weight grid does not match cell/candidate basis")
    if np.any(weights < -1e-10) or not np.allclose(
        weights.sum(-1), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("route weights are outside the simplex")
    weight_map = np.empty(
        (len(spectra), spectra.shape[-2], spectra.shape[-1]),
        dtype=np.float64,
    )
    assigned = np.zeros(spectra.shape[-2:], dtype=np.int8)
    for time_index, (time_start, time_end) in enumerate(times):
        for band_index, (frequency_start, frequency_end) in enumerate(bands):
            weight_map[
                :, frequency_start:frequency_end, time_start:time_end
            ] = weights[time_index, band_index, :, None, None]
            assigned[
                frequency_start:frequency_end, time_start:time_end
            ] += 1
    if not np.all(assigned == 1):
        raise RuntimeError("route cells do not cover every spectral bin exactly once")
    combined = np.sum(spectra * weight_map[:, None], axis=0)
    output = np.column_stack([
        librosa.istft(
            combined[channel], hop_length=config.hop_length,
            win_length=config.n_fft, window="hann", center=True,
            length=int(frames),
        )
        for channel in range(2)
    ]).astype(np.float32)
    if output.shape != (frames, 2) or not np.all(np.isfinite(output)):
        raise RuntimeError(f"invalid routed output {output.shape}")
    return output


def roundtrip_control(audio: np.ndarray, config: RoutingSpectralConfig) -> dict:
    """Measure the common STFT/ISTFT identity route for one stereo signal."""

    value = _audio(audio, "audio")
    spectrum = stft_stack([value], config)[0]
    report = {
        "time_frame_ranges": [
            list(item) for item in time_frame_ranges(spectrum.shape[-1], config)
        ],
        "frequency_bin_ranges": [
            list(item) for item in frequency_bin_ranges(spectrum.shape[-2], config)
        ],
    }
    weights = np.ones((
        len(report["time_frame_ranges"]),
        len(report["frequency_bin_ranges"]), 1,
    ))
    rendered = render_spectral_route(
        spectrum[None], weights, report, frames=len(value), config=config
    )
    difference = rendered.astype(np.float64) - value.astype(np.float64)
    return {
        "audio": rendered,
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(np.square(difference)))),
    }
