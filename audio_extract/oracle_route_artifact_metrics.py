"""Exact route-artifact diagnostics for the binding routing experiment."""

from __future__ import annotations

from typing import Sequence
import math

import librosa
import numpy as np


def _stereo(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.ndim != 2
        or result.shape[0] < 2
        or result.shape[1] != 2
        or not np.issubdtype(result.dtype, np.floating)
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be finite floating stereo audio")
    return np.asarray(result, dtype=np.float64)


def _matching_stereo(
    candidate: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    candidate_array = _stereo(candidate, "candidate")
    reference_array = _stereo(reference, "reference")
    if candidate_array.shape != reference_array.shape:
        raise ValueError("candidate and reference grids differ")
    return candidate_array, reference_array


def _complex_stft(value: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    return np.stack(
        [
            librosa.stft(
                value[:, channel],
                n_fft=n_fft,
                hop_length=hop_length,
                window="hann",
                center=True,
                pad_mode="constant",
            )
            for channel in range(2)
        ]
    )


def transform_identity_metrics(
    raw: np.ndarray,
    transformed: np.ndarray,
    *,
    sample_rate_hz: int,
) -> dict[str, float]:
    """Measure raw versus analysis/synthesis round-trip identity."""

    raw_array, transformed_array = _matching_stereo(raw, transformed)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    error = transformed_array - raw_array
    spectral = []
    for n_fft in (512, 2048, 8192):
        hop = n_fft // 4
        raw_spectrum = _complex_stft(raw_array, n_fft, hop)
        error_spectrum = _complex_stft(error, n_fft, hop)
        spectral.append(
            math.sqrt(
                float(np.sum(np.abs(error_spectrum) ** 2))
                / max(float(np.sum(np.abs(raw_spectrum) ** 2)), 1e-30)
            )
        )
    raw_mid_side = np.column_stack(
        (
            0.5 * (raw_array[:, 0] + raw_array[:, 1]),
            0.5 * (raw_array[:, 0] - raw_array[:, 1]),
        )
    )
    error_mid_side = np.column_stack(
        (
            0.5 * (error[:, 0] + error[:, 1]),
            0.5 * (error[:, 0] - error[:, 1]),
        )
    )
    stereo_error = math.sqrt(
        float(np.sum(np.square(error_mid_side)))
        / max(float(np.sum(np.square(raw_mid_side))), 1e-30)
    )
    return {
        "max_abs": float(np.max(np.abs(error))),
        "rms": float(np.sqrt(np.mean(np.square(error)))),
        "spectral_error": float(max(spectral)),
        "stereo_error": float(stereo_error),
    }


def time_boundary_seam_metrics(
    output: np.ndarray,
    *,
    boundary_samples: Sequence[int],
) -> dict[str, float | int]:
    """Measure sample-derivative concentration at declared time boundaries."""

    value = _stereo(output, "output")
    boundaries = tuple(int(index) for index in boundary_samples)
    if any(index <= 0 or index >= len(value) for index in boundaries):
        raise ValueError("time boundary lies outside the output grid")
    derivative = np.max(np.abs(np.diff(value, axis=0)), axis=1)
    reference = max(float(np.percentile(derivative, 99.0)), 1e-30)
    jumps = np.asarray(
        [
            np.max(np.abs(value[index] - value[index - 1]))
            for index in boundaries
        ],
        dtype=np.float64,
    )
    return {
        "boundary_count": len(boundaries),
        "p99_derivative": reference,
        "max_boundary_jump": float(jumps.max(initial=0.0)),
        "seam_max_boundary_jump_over_p99_derivative": float(
            jumps.max(initial=0.0) / reference
        ),
    }


def frequency_boundary_error_metrics(
    output: np.ndarray,
    reference: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    boundary_bins: Sequence[int],
) -> dict[str, float | int]:
    """Measure whether exact error energy concentrates at routed band edges."""

    candidate, target = _matching_stereo(output, reference)
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("STFT dimensions must be positive")
    error_spectrum = _complex_stft(candidate - target, n_fft, hop_length)
    power = np.mean(np.abs(error_spectrum) ** 2, axis=(0, 2))
    boundaries = tuple(int(index) for index in boundary_bins)
    if any(index <= 0 or index >= len(power) for index in boundaries):
        raise ValueError("frequency boundary lies outside the STFT grid")
    boundary_mask = np.zeros(len(power), dtype=bool)
    neighborhood_mask = np.zeros(len(power), dtype=bool)
    for index in boundaries:
        boundary_mask[max(0, index - 1):min(len(power), index + 2)] = True
        neighborhood_mask[max(0, index - 5):min(len(power), index + 6)] = True
    neighborhood_mask &= ~boundary_mask
    boundary_energy = float(np.mean(power[boundary_mask])) if np.any(
        boundary_mask
    ) else 0.0
    neighborhood_energy = float(
        np.mean(power[neighborhood_mask])
    ) if np.any(neighborhood_mask) else 0.0
    if boundary_energy <= 1e-30 and neighborhood_energy <= 1e-30:
        ratio = 0.0
    else:
        ratio = boundary_energy / max(neighborhood_energy, 1e-30)
    return {
        "boundary_count": len(boundaries),
        "boundary_error_energy": boundary_energy,
        "neighbor_error_energy": neighborhood_energy,
        "frequency_boundary_ringing_ratio": float(ratio),
    }


def hall_tail_preservation_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    sample_rate_hz: int,
) -> dict[str, float | int]:
    """Measure envelope error on low-energy frames that are actively decaying."""

    output, target = _matching_stereo(candidate, reference)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    frame_length = max(32, round(0.050 * sample_rate_hz))
    hop_length = max(16, round(0.025 * sample_rate_hz))

    def envelope(value):
        mono = np.sqrt(np.mean(np.square(value), axis=1))
        framed = librosa.feature.rms(
            y=mono,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
            pad_mode="constant",
        )[0]
        return 20.0 * np.log10(np.maximum(framed, 1e-12))

    reference_db = envelope(target)
    candidate_db = envelope(output)
    frames = min(len(reference_db), len(candidate_db))
    reference_db = reference_db[:frames]
    candidate_db = candidate_db[:frames]
    kernel = np.full(5, 0.2)
    smooth = np.convolve(reference_db, kernel, mode="same")
    slope = np.zeros(frames, dtype=np.float64)
    if frames > 4:
        slope[:-4] = smooth[4:] - smooth[:-4]
    peak = float(np.max(smooth))
    selected = (
        (smooth <= peak - 6.0)
        & (smooth >= peak - 60.0)
        & (slope <= -0.25)
    )
    if np.count_nonzero(selected) < 8:
        finite = smooth > -200.0
        if np.any(finite):
            lower, upper = np.percentile(smooth[finite], [10.0, 50.0])
            selected = finite & (smooth >= lower) & (smooth <= upper)
    if not np.any(selected):
        raise ValueError("reference has no measurable hall-tail frames")
    error = np.abs(candidate_db[selected] - reference_db[selected])
    return {
        "hall_tail_frames": int(np.count_nonzero(selected)),
        "hall_tail_error_db/v1": float(np.percentile(error, 90.0)),
    }
