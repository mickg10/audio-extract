import numpy as np
import pytest

pytest.importorskip("librosa")

from audio_extract.oracle_routing_spectral_v2 import (
    RoutingSpectralConfig,
    build_quadratic_grid,
    frequency_bin_ranges,
    render_spectral_route,
    roundtrip_control,
    stft_stack,
    validate_exact_audio_basis,
)


def config(**kwargs):
    values = dict(
        sample_rate_hz=8_000,
        n_fft=64,
        hop_length=16,
        time_cell_seconds=0.064,
        band_edges_hz=(0, 1_000, 2_000, 4_000),
        min_source_power_relative=1e-8,
        reference_floor_relative=1e-10,
    )
    values.update(kwargs)
    return RoutingSpectralConfig(**values)


def stereo_signal(frames=512):
    time = np.arange(frames) / 8_000
    left = (
        0.5 * np.sin(2 * np.pi * 310 * time)
        + 0.2 * np.sin(2 * np.pi * 970 * time)
    )
    return np.column_stack((left, 0.7 * left)).astype(np.float32)


def test_frequency_ranges_cover_every_bin_exactly_once():
    cfg = config()
    ranges = frequency_bin_ranges(cfg.n_fft // 2 + 1, cfg)
    covered = np.concatenate([
        np.arange(start, end) for start, end in ranges
    ])
    assert np.array_equal(covered, np.arange(cfg.n_fft // 2 + 1))
    assert all(end > start for start, end in ranges)


def test_exact_audio_basis_refuses_implicit_truncation():
    accompaniment = stereo_signal(512)
    vocal = np.zeros_like(accompaniment)
    with pytest.raises(ValueError, match="grids differ"):
        validate_exact_audio_basis(
            [accompaniment, accompaniment[:-1]], accompaniment, vocal
        )


def test_stft_identity_roundtrip_is_small_and_exact_length():
    signal = stereo_signal()
    result = roundtrip_control(signal, config())
    assert result["audio"].shape == signal.shape
    assert result["max_abs"] < 2e-5
    assert result["rms"] < 2e-6


def test_no_vocal_cells_all_receive_direct_or_silent_fallback():
    cfg = config()
    accompaniment = stereo_signal()
    vocal = np.zeros_like(accompaniment)
    candidates = [accompaniment, 0.5 * accompaniment]
    candidate_spectra = stft_stack(candidates, cfg)
    truth_spectra = stft_stack([accompaniment, vocal], cfg)
    grid, report = build_quadratic_grid(
        candidate_spectra, truth_spectra[0], truth_spectra[1], cfg
    )
    grid.validate()
    assert report["total_cells"] == sum(report["mode_counts"].values())
    assert report["mode_counts"].get("source_coordinates", 0) == 0
    assert all(
        "direct_fallback" in mode for mode in report["mode_counts"]
    )


def test_shared_stereo_weights_preserve_deterministic_channel_ratio():
    cfg = config()
    base = stereo_signal()
    # Give every candidate an exact 2:1 right/left ratio.
    base[:, 1] = 2.0 * base[:, 0]
    candidates = [base, 0.5 * base, 1.25 * base]
    spectra = stft_stack(candidates, cfg)
    truth = stft_stack([base, np.zeros_like(base)], cfg)
    _, report = build_quadratic_grid(
        spectra, truth[0], truth[1], cfg
    )
    time_cells = len(report["time_frame_ranges"])
    band_cells = len(report["frequency_bin_ranges"])
    weights = np.empty((time_cells, band_cells, 3), dtype=np.float64)
    for time_index in range(time_cells):
        for band_index in range(band_cells):
            weights[time_index, band_index] = (
                (0.2, 0.3, 0.5)
                if (time_index + band_index) % 2 == 0
                else (0.6, 0.1, 0.3)
            )
    output = render_spectral_route(
        spectra, weights, report, frames=len(base), config=cfg
    )
    assert np.max(np.abs(output[:, 1] - 2.0 * output[:, 0])) < 2e-5


def test_invalid_edges_are_refused_before_analysis():
    with pytest.raises(ValueError, match="Nyquist"):
        config(band_edges_hz=(0, 1_000, 3_000)).validate()
