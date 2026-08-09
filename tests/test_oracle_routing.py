import numpy as np
import pytest

from audio_extract.oracle_routing import (
    CellStatistics,
    RoutingConfig,
    best_whole_track,
    one_hot_weights,
    render_spectral_route,
    solve_convex_routing,
    solve_discrete_routing,
    source_coordinate_statistics,
    stft_stack,
    validate_basis,
)


def _synthetic_stats(unary):
    risk = np.asarray(unary, dtype=np.float64)
    q, b, k = risk.shape
    alpha = np.ones((q, b, k), dtype=np.complex128)
    beta = np.zeros_like(alpha)
    gram = np.zeros((q, b, k, k), dtype=np.complex128)
    return CellStatistics(
        alpha, beta, gram, np.ones((q, b)), np.ones((q, b)), np.ones((q, b)),
        np.ones((q, b), dtype=bool), risk,
        tuple((i, i + 1) for i in range(q)), tuple((i, i + 1) for i in range(b)),
    )


def test_validate_basis_rejects_grid_mismatch_and_nonfinite():
    a = np.zeros((100, 2), dtype=np.float32)
    v = np.ones_like(a)
    with pytest.raises(ValueError, match="grid"):
        validate_basis([a, a[:-1]], a, v)
    bad = a.copy(); bad[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_basis([a, bad], a, v)


def test_o1_and_o2_smooth_global_solution():
    # Candidate 0 wins three cells; a tiny isolated win by candidate 1 is not
    # worth two 0.05 switches.  The MILP must solve the global Potts objective.
    stats = _synthetic_stats([[[0.0, 1.0]], [[0.0, 1.0]],
                              [[0.02, 0.0]], [[0.0, 1.0]]])
    cfg = RoutingConfig(temporal_switch_penalty=0.05,
                        frequency_switch_penalty=0.05)
    o1, means = best_whole_track(stats)
    assert o1 == 0 and means[0] < means[1]
    o2 = solve_discrete_routing(stats, cfg)
    assert np.array_equal(o2.labels, np.zeros((4, 1), dtype=int))
    assert o2.temporal_switches == 0
    assert o2.mip_gap == 0.0


def test_o3_contains_exact_o1_and_o2_endpoints():
    stats = _synthetic_stats([[[0.0, 1.0], [1.0, 0.0]],
                              [[0.0, 1.0], [1.0, 0.0]]])
    cfg = RoutingConfig(o3_iterations=10)
    o2 = solve_discrete_routing(stats, cfg)
    o3 = solve_convex_routing(stats, cfg, o1_index=0, o2_labels=o2.labels)
    assert o3.objective <= o2.objective + 1e-9
    assert np.all(o3.weights >= 0)
    assert np.allclose(o3.weights.sum(-1), 1.0)


def test_spectral_statistics_mask_silent_source_and_render_exact_grid():
    sr = 8_000
    frames = 8_000
    t = np.arange(frames) / sr
    a = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype("float32")
    v = np.stack([0.1 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype("float32")
    candidates = [a + 0.5 * v, 0.9 * a + 0.1 * v]
    cfg = RoutingConfig(sample_rate_hz=sr, n_fft=256, hop_length=128,
                        tile_seconds=0.25,
                        band_edges_hz=(0, 250, 500, 1_000, 2_000, 4_000),
                        o3_iterations=5)
    spectra = stft_stack(candidates, cfg)
    truth = stft_stack([a, v], cfg)
    stats = source_coordinate_statistics(spectra, truth[0], truth[1], cfg)
    assert stats.available.any()
    o1, _ = best_whole_track(stats)
    labels = np.full(stats.available.shape, o1, dtype=int)
    weights = one_hot_weights(labels, len(candidates))
    output = render_spectral_route(spectra, weights, stats, frames=frames, config=cfg)
    assert output.shape == a.shape and output.dtype == np.float32
    # Constant one-hot routing is exactly the common STFT/ISTFT reconstruction
    # within normal floating synthesis tolerance.
    assert np.max(np.abs(output - candidates[o1])) < 2e-6

