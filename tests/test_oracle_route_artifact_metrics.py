import numpy as np
import pytest

from audio_extract.oracle_route_artifact_metrics import (
    frequency_boundary_error_metrics,
    hall_tail_preservation_metrics,
    time_boundary_seam_metrics,
    transform_identity_metrics,
)


def _stereo_tone(frames=48_000, sample_rate=48_000):
    time = np.arange(frames) / sample_rate
    left = 0.2 * np.sin(2 * np.pi * 440 * time)
    right = 0.15 * np.sin(2 * np.pi * 660 * time)
    return np.column_stack((left, right)).astype(np.float32)


def test_transform_identity_is_zero_for_exact_roundtrip():
    value = _stereo_tone(12_000)
    result = transform_identity_metrics(
        value, value.copy(), sample_rate_hz=48_000
    )
    assert result == {
        "max_abs": 0.0,
        "rms": 0.0,
        "spectral_error": 0.0,
        "stereo_error": 0.0,
    }


def test_time_boundary_metric_detects_a_declared_discontinuity():
    smooth = _stereo_tone(12_000)
    broken = smooth.copy()
    broken[6_000:] += 0.5
    clean = time_boundary_seam_metrics(smooth, boundary_samples=(6_000,))
    damaged = time_boundary_seam_metrics(broken, boundary_samples=(6_000,))
    assert damaged["seam_max_boundary_jump_over_p99_derivative"] > clean[
        "seam_max_boundary_jump_over_p99_derivative"
    ]
    assert damaged["seam_max_boundary_jump_over_p99_derivative"] > 2.0


def test_frequency_boundary_metric_detects_boundary_localized_error():
    sample_rate = 48_000
    n_fft = 2_048
    hop = 512
    frames = 48_000
    reference = np.zeros((frames, 2), dtype=np.float32)
    boundary_bin = 64
    frequency = boundary_bin * sample_rate / n_fft
    time = np.arange(frames) / sample_rate
    error = 0.1 * np.sin(2 * np.pi * frequency * time)
    damaged = np.column_stack((error, error)).astype(np.float32)
    exact = frequency_boundary_error_metrics(
        reference,
        reference,
        n_fft=n_fft,
        hop_length=hop,
        boundary_bins=(boundary_bin,),
    )
    result = frequency_boundary_error_metrics(
        damaged,
        reference,
        n_fft=n_fft,
        hop_length=hop,
        boundary_bins=(boundary_bin,),
    )
    assert exact["frequency_boundary_ringing_ratio"] == 0.0
    assert result["frequency_boundary_ringing_ratio"] > 10.0


def test_hall_tail_metric_penalizes_truncated_decay():
    sample_rate = 8_000
    frames = 4 * sample_rate
    time = np.arange(frames) / sample_rate
    decay = np.exp(-time / 1.0)
    reference = np.column_stack(
        (
            0.2 * decay * np.sin(2 * np.pi * 330 * time),
            0.15 * decay * np.sin(2 * np.pi * 440 * time),
        )
    ).astype(np.float32)
    exact = hall_tail_preservation_metrics(
        reference, reference, sample_rate_hz=sample_rate
    )
    truncated = reference.copy()
    truncated[2 * sample_rate:] = 0.0
    damaged = hall_tail_preservation_metrics(
        truncated, reference, sample_rate_hz=sample_rate
    )
    assert exact["hall_tail_error_db/v1"] == pytest.approx(0.0)
    assert damaged["hall_tail_error_db/v1"] > 10.0


def test_metric_primitives_refuse_grid_mismatch():
    value = _stereo_tone(1_000)
    with pytest.raises(ValueError, match="grids differ"):
        transform_identity_metrics(
            value[:-1], value, sample_rate_hz=48_000
        )
