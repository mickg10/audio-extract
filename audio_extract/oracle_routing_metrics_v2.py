"""Exact complete-work metrics for certified opera routing.

Every metric consumes exact-grid stereo FLOAT arrays. No alignment, truncation,
resampling, gain matching, or hidden normalization is performed. The functions
cover the binding decision axes that were previously written incompletely:
source assignment, stereo, hall tails, transients, no-vocal behavior, worst
identifiable events, and time/frequency route-boundary diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .judge_labels import local_source_coordinate_labels
from .judge_train import label_targets

_EPS = np.finfo(np.float64).tiny


def _audio(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] < 2:
        raise ValueError(f"{name} must have shape (frames,2) with at least 2 frames")
    if not np.issubdtype(result.dtype, np.floating):
        raise ValueError(f"{name} must contain floating samples")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite samples")
    return np.asarray(result, dtype=np.float64)


def _exact(candidate: np.ndarray, accompaniment: np.ndarray, vocal: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = _audio(candidate, "candidate")
    accompaniment = _audio(accompaniment, "accompaniment")
    vocal = _audio(vocal, "vocal")
    if candidate.shape != accompaniment.shape or vocal.shape != accompaniment.shape:
        raise ValueError(
            f"metric grids differ: {candidate.shape}, {accompaniment.shape}, {vocal.shape}"
        )
    return candidate, accompaniment, vocal


def _frames(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if frame < 1 or hop < 1:
        raise ValueError("frame and hop must be positive")
    if len(signal) < frame:
        pad = np.zeros((frame - len(signal), signal.shape[1]), dtype=signal.dtype)
        signal = np.concatenate((signal, pad), axis=0)
    starts = range(0, max(1, len(signal) - frame + 1), hop)
    result = [signal[start:start + frame] for start in starts]
    final_start = len(signal) - frame
    if not result or final_start > (len(result) - 1) * hop:
        result.append(signal[final_start:final_start + frame])
    return np.stack(result)


def _frame_energy(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    framed = _frames(signal, frame, hop)
    return np.mean(np.square(framed), axis=(1, 2))


def _percentile(values: np.ndarray, value: float, default: float = 0.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, value)) if finite.size else float(default)


def source_assignment_metrics(
    candidate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    sample_rate_hz: int,
    *,
    tile_seconds: float = 0.5,
    hop_seconds: float = 0.25,
) -> tuple[dict[str, float | int], tuple]:
    """Return exact alpha/beta/R aggregates and the underlying local labels."""

    candidate, accompaniment, vocal = _exact(candidate, accompaniment, vocal)
    if sample_rate_hz <= 0 or tile_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("invalid source-assignment geometry")
    tile = max(256, round(tile_seconds * sample_rate_hz))
    hop = max(128, round(hop_seconds * sample_rate_hz))
    labels = local_source_coordinate_labels(
        candidate, accompaniment, vocal,
        tile_frames=tile, hop_frames=hop,
    )
    targets = label_targets(labels)
    required = (
        "retained_voice_db_p90",
        "retained_voice_coef_p90",
        "event_hole_db_p90",
        "event_hole_db_max",
        "artifact_ratio_p90",
        "alpha_error_p90",
        "_available_tiles",
        "_total_tiles",
    )
    result: dict[str, float | int] = {}
    for name in required:
        value = targets.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"source metric {name} is missing/non-finite: {value!r}")
        result[name] = int(value) if name.startswith("_") else float(value)
    if result["_available_tiles"] <= 0:
        raise ValueError("no identifiable source-coordinate tiles")
    return result, labels


def stereo_metrics(
    candidate: np.ndarray,
    accompaniment: np.ndarray,
    sample_rate_hz: int,
    *,
    frame_seconds: float = 0.5,
    hop_seconds: float = 0.25,
) -> dict[str, float]:
    """Measure p90 mid/side width and zero-lag coherence deviation."""

    candidate = _audio(candidate, "candidate")
    accompaniment = _audio(accompaniment, "accompaniment")
    if candidate.shape != accompaniment.shape:
        raise ValueError("stereo metric grids differ")
    frame = max(256, round(frame_seconds * sample_rate_hz))
    hop = max(128, round(hop_seconds * sample_rate_hz))
    candidate_frames = _frames(candidate, frame, hop)
    reference_frames = _frames(accompaniment, frame, hop)
    if candidate_frames.shape != reference_frames.shape:
        raise RuntimeError("stereo frame grids differ")

    reference_energy = np.mean(np.square(reference_frames), axis=(1, 2))
    active_floor = max(_percentile(reference_energy, 15), _EPS)
    active = reference_energy > active_floor
    if not np.any(active):
        raise ValueError("no active stereo frames")

    def facts(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left, right = frames[:, :, 0], frames[:, :, 1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        width = 10.0 * np.log10(
            (np.mean(np.square(side), axis=1) + _EPS)
            / (np.mean(np.square(mid), axis=1) + _EPS)
        )
        coherence = np.sum(left * right, axis=1) / np.sqrt(
            (np.sum(np.square(left), axis=1) + _EPS)
            * (np.sum(np.square(right), axis=1) + _EPS)
        )
        return width, coherence

    candidate_width, candidate_coherence = facts(candidate_frames)
    reference_width, reference_coherence = facts(reference_frames)
    return {
        "stereo_width_dev_db/v2": _percentile(
            np.abs(candidate_width[active] - reference_width[active]), 90
        ),
        "interchannel_coherence_dev/v2": _percentile(
            np.abs(candidate_coherence[active] - reference_coherence[active]), 90
        ),
    }


def hall_tail_metric(
    candidate: np.ndarray,
    accompaniment: np.ndarray,
    sample_rate_hz: int,
    *,
    frame_seconds: float = 0.10,
    hop_seconds: float = 0.05,
    lookback_seconds: float = 1.0,
) -> dict[str, float | int]:
    """Measure level deviation on low-energy frames following strong phrases.

    A tail frame is active accompaniment below its work-level p40 whose previous
    one-second window contains energy at or above p75. The metric is the p90
    absolute candidate/reference energy-ratio error in dB.
    """

    candidate = _audio(candidate, "candidate")
    accompaniment = _audio(accompaniment, "accompaniment")
    if candidate.shape != accompaniment.shape:
        raise ValueError("hall-tail grids differ")
    frame = max(128, round(frame_seconds * sample_rate_hz))
    hop = max(64, round(hop_seconds * sample_rate_hz))
    lookback = max(1, round(lookback_seconds * sample_rate_hz / hop))
    reference = _frame_energy(accompaniment, frame, hop)
    estimate = _frame_energy(candidate, frame, hop)
    if reference.shape != estimate.shape:
        raise RuntimeError("hall-tail frame grids differ")
    nonzero = reference[reference > max(reference.max(initial=0.0) * 1e-10, _EPS)]
    if nonzero.size < 4:
        raise ValueError("insufficient active accompaniment for hall-tail metric")
    low = float(np.percentile(nonzero, 40))
    high = float(np.percentile(nonzero, 75))
    active_floor = max(float(np.percentile(nonzero, 5)), _EPS)
    mask = np.zeros_like(reference, dtype=bool)
    for index in range(len(reference)):
        prior = reference[max(0, index - lookback):index]
        mask[index] = (
            active_floor <= reference[index] <= low
            and prior.size > 0
            and float(prior.max()) >= high
        )
    if np.count_nonzero(mask) < 2:
        # Fail closed to a reproducible low-active fallback rather than silently
        # reporting a clean zero.
        mask = (reference >= active_floor) & (reference <= low)
    if not np.any(mask):
        raise ValueError("no hall-tail/low-active frames")
    deviation = np.abs(10.0 * np.log10(
        (estimate[mask] + _EPS) / (reference[mask] + _EPS)
    ))
    return {
        "hall_tail_dev_db/v2": _percentile(deviation, 90),
        "hall_tail_frames/v2": int(np.count_nonzero(mask)),
    }


def transient_metrics(
    candidate: np.ndarray,
    accompaniment: np.ndarray,
    sample_rate_hz: int,
    *,
    frame_seconds: float = 0.020,
    hop_seconds: float = 0.010,
) -> dict[str, float | int]:
    """Measure p90 transient loss/excess on high derivative-energy events."""

    candidate = _audio(candidate, "candidate")
    accompaniment = _audio(accompaniment, "accompaniment")
    if candidate.shape != accompaniment.shape:
        raise ValueError("transient grids differ")
    frame = max(64, round(frame_seconds * sample_rate_hz))
    hop = max(32, round(hop_seconds * sample_rate_hz))
    candidate_flux = _frame_energy(np.diff(candidate, axis=0), frame, hop)
    reference_flux = _frame_energy(np.diff(accompaniment, axis=0), frame, hop)
    if candidate_flux.shape != reference_flux.shape:
        raise RuntimeError("transient frame grids differ")
    active = reference_flux >= _percentile(reference_flux, 80)
    active &= reference_flux > max(reference_flux.max(initial=0.0) * 1e-10, _EPS)
    if not np.any(active):
        raise ValueError("no identifiable transient frames")
    ratio_db = 10.0 * np.log10(
        (candidate_flux[active] + _EPS) / (reference_flux[active] + _EPS)
    )
    return {
        "transient_loss_db/v2": _percentile(np.maximum(-ratio_db, 0.0), 90),
        "transient_excess_db/v2": _percentile(np.maximum(ratio_db, 0.0), 90),
        "transient_event_frames/v2": int(np.count_nonzero(active)),
    }


def worst_identifiable_event(labels: Sequence[Any]) -> dict[str, Any]:
    available = [label for label in labels if getattr(label, "available", False)]
    if not available:
        return {"available": False}

    def risk(label: Any) -> float:
        voice = max(0.0, float(label.retained_voice_energy_ratio or 0.0))
        return (
            float(label.accompaniment_hole_db or 0.0) / 20.0
            + math.sqrt(voice)
            + float(label.orthogonal_artifact_ratio or 0.0)
        )

    label = max(available, key=risk)
    return {
        "available": True,
        "tile_index": int(label.tile_index),
        "start_frame": int(label.start_frame),
        "end_frame": int(label.end_frame),
        "accompaniment_hole_db": float(label.accompaniment_hole_db or 0.0),
        "retained_voice_energy_db": float(label.retained_voice_energy_db or 0.0),
        "orthogonal_artifact_ratio": float(label.orthogonal_artifact_ratio or 0.0),
        "condition_number": (
            None if label.condition_number is None
            else float(label.condition_number)
        ),
        "composite_risk": float(risk(label)),
    }


def no_vocal_false_positive_energy_ratio(
    no_vocal_mixture: np.ndarray,
    accompaniment_candidate: np.ndarray,
) -> float:
    mixture = _audio(no_vocal_mixture, "no_vocal_mixture")
    candidate = _audio(accompaniment_candidate, "accompaniment_candidate")
    if mixture.shape != candidate.shape:
        raise ValueError("no-vocal grids differ")
    removed = mixture - candidate
    return float(
        np.sum(np.square(removed)) / max(np.sum(np.square(mixture)), _EPS)
    )


def route_boundary_metrics(
    candidate: np.ndarray,
    accompaniment: np.ndarray,
    sample_rate_hz: int,
    *,
    time_frame_ranges: Sequence[Sequence[int]],
    frequency_bin_ranges: Sequence[Sequence[int]],
    n_fft: int,
    hop_length: int,
) -> dict[str, float | int]:
    """Measure time jumps and residual-spectrum discontinuities at route borders."""

    import librosa

    candidate = _audio(candidate, "candidate")
    accompaniment = _audio(accompaniment, "accompaniment")
    if candidate.shape != accompaniment.shape:
        raise ValueError("route-boundary grids differ")
    derivative = np.max(np.abs(np.diff(candidate, axis=0)), axis=1)
    derivative_reference = max(_percentile(derivative, 99), _EPS)
    boundaries = sorted({
        min(len(candidate) - 1, int(stop) * int(hop_length))
        for _, stop in time_frame_ranges[:-1]
        if 0 < int(stop) * int(hop_length) < len(candidate)
    })
    # STFT cell boundaries identify a hop-sized sample neighborhood rather than
    # one exact waveform sample (centred windows overlap both sides). Measure the
    # worst derivative in that neighborhood so a discontinuity between adjacent
    # frame centres cannot hide in the integer frame-to-sample quantization.
    jumps = np.asarray([
        derivative[
            max(0, index - int(hop_length)):
            min(len(derivative), index + int(hop_length))
        ].max(initial=0.0)
        for index in boundaries
    ], dtype=np.float64)
    time_ratio = float(jumps.max(initial=0.0) / derivative_reference)

    def spectrum(value: np.ndarray) -> np.ndarray:
        return np.stack([
            librosa.stft(
                value[:, channel], n_fft=n_fft, hop_length=hop_length,
                win_length=n_fft, window="hann", center=True,
                pad_mode="constant",
            )
            for channel in range(2)
        ])

    residual = spectrum(candidate) - spectrum(accompaniment)
    frequency_boundaries = sorted({
        int(stop) for _, stop in frequency_bin_ranges[:-1]
        if 0 < int(stop) < residual.shape[1]
    })
    ratios = []
    for boundary in frequency_boundaries:
        jump = np.abs(residual[:, boundary] - residual[:, boundary - 1])
        lo = max(1, boundary - 4)
        hi = min(residual.shape[1], boundary + 4)
        local_differences = np.abs(np.diff(residual[:, lo:hi], axis=1))
        reference = max(_percentile(local_differences, 95), _EPS)
        ratios.append(float(_percentile(jump, 99) / reference))
    frequency_ratio = max(ratios, default=0.0)
    return {
        "time_boundary_count": len(boundaries),
        "max_time_boundary_jump_over_p99_derivative": time_ratio,
        "frequency_boundary_count": len(frequency_boundaries),
        "max_frequency_ringing_ratio": float(frequency_ratio),
    }


def complete_work_metrics(
    candidate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    sample_rate_hz: int,
) -> tuple[dict[str, float | int], tuple]:
    source, labels = source_assignment_metrics(
        candidate, accompaniment, vocal, sample_rate_hz
    )
    result: dict[str, float | int] = dict(source)
    result.update(stereo_metrics(candidate, accompaniment, sample_rate_hz))
    result.update(hall_tail_metric(candidate, accompaniment, sample_rate_hz))
    result.update(transient_metrics(candidate, accompaniment, sample_rate_hz))
    required = (
        "retained_voice_db_p90", "event_hole_db_p90", "artifact_ratio_p90",
        "stereo_width_dev_db/v2", "interchannel_coherence_dev/v2",
        "hall_tail_dev_db/v2", "transient_loss_db/v2",
        "transient_excess_db/v2",
    )
    for name in required:
        value = result.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"complete-work metric {name} is invalid: {value!r}")
    return result, labels
