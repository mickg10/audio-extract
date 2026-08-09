import numpy as np
import pytest

from audio_extract.oracle_routing_spectral_v2 import (
    CELL_MEASURE_REVISION,
    RoutingSpectralConfig,
    build_quadratic_grid,
    cell_measure_grid,
    frequency_bin_multiplicity,
)


def config():
    return RoutingSpectralConfig(
        sample_rate_hz=16,
        n_fft=16,
        hop_length=8,
        time_cell_seconds=1.0,
        band_edges_hz=(0, 4, 8),
    )


def test_even_rfft_multiplicity_counts_dc_and_nyquist_once():
    result = frequency_bin_multiplicity(9, config())
    assert result.tolist() == [
        1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0
    ]
    assert result.sum() == 16.0


def test_cell_measure_uses_time_span_channels_and_real_frequency_dof():
    result = cell_measure_grid(
        ((0, 2), (2, 3)),
        ((0, 4), (4, 9)),
        channels=2,
        frequency_bins=9,
        config=config(),
    )
    # First band: 1 + 2 + 2 + 2 = 7; second: 2 + 2 + 2 + 2 + 1 = 9.
    assert result.tolist() == [[28.0, 36.0], [14.0, 18.0]]


def test_quadratic_grid_carries_and_reports_exact_cell_measure():
    cfg = config()
    rng = np.random.default_rng(4)
    candidates = (
        rng.normal(size=(2, 2, 9, 3))
        + 1j * rng.normal(size=(2, 2, 9, 3))
    ).astype(np.complex64)
    accompaniment = np.ones((2, 9, 3), dtype=np.complex64)
    vocal = np.empty((2, 9, 3), dtype=np.complex64)
    vocal[0] = 1j
    vocal[1] = -1j
    grid, report = build_quadratic_grid(
        candidates, accompaniment, vocal, cfg
    )
    expected = np.asarray([[28.0, 36.0], [14.0, 18.0]])
    assert np.array_equal(grid.measure, expected)
    assert report["cell_measure_revision"] == CELL_MEASURE_REVISION
    assert np.array_equal(np.asarray(report["cell_measure"]), expected)
    assert np.asarray(report["normalized_cell_measure"]).sum() == pytest.approx(1.0)
    assert report["cell_measure_sum"] == pytest.approx(float(expected.sum()))
