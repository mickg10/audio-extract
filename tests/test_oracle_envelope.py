import numpy as np
import pytest

from audio_extract.oracle_envelope import (
    OracleEnvelopeConfig,
    project_simplex,
    reconstruct_convex,
    reconstruct_medoid,
    smooth_convex_weights,
    smooth_medoid_labels,
    source_coordinate_quadratics,
)


def config(**kwargs):
    values = dict(
        n_fft=16,
        hop_length=4,
        time_cell_seconds=1.0,
        band_edges_hz=(0.0, 22050.0),
        temporal_switch_penalty=0.1,
        frequency_switch_penalty=0.0,
        temporal_weight_smoothness=0.0,
        frequency_weight_smoothness=0.0,
        convex_max_iterations=500,
        convex_tolerance=1e-9,
    )
    values.update(kwargs)
    return OracleEnvelopeConfig(**values)


def test_simplex_projection_is_nonnegative_and_sums_to_one():
    values = np.array([[-1.0, 2.0, 0.0], [0.2, 0.3, 0.7]])
    projected = project_simplex(values)
    assert np.all(projected >= 0)
    assert np.allclose(projected.sum(axis=-1), 1.0)


def test_smooth_medoid_finds_complementary_time_route():
    unary = np.array([
        [[0.0, 10.0]],
        [[10.0, 0.0]],
    ])
    labels, report = smooth_medoid_labels(unary, config())
    assert labels.tolist() == [[0], [1]]
    assert report["temporal_switches"] == 1
    assert report["occupancy"] == [1, 1]


def test_smooth_convex_hull_interpolates_when_neither_vertex_is_exact():
    # Objective (2*w_1 - 1)^2, ignoring its constant:
    # G[1,1]=4 and c[1]=2. The simplex optimum is [0.5, 0.5].
    G = np.array([[[[0.0, 0.0], [0.0, 4.0]]]])
    c = np.array([[[0.0, 2.0]]])
    weights, report = smooth_convex_weights(G, c, config())
    assert weights[0, 0].sum() == pytest.approx(1.0)
    assert weights[0, 0, 0] == pytest.approx(0.5, abs=1e-5)
    assert weights[0, 0, 1] == pytest.approx(0.5, abs=1e-5)
    assert report["min_weight"] >= 0


def test_source_coordinate_unary_prefers_exact_accompaniment():
    # A and V occupy disjoint STFT bins, making their local coordinates identifiable.
    C, F, N = 2, 9, 4
    accompaniment = np.zeros((C, F, N), dtype=np.complex128)
    vocal = np.zeros_like(accompaniment)
    accompaniment[:, :4, :] = 1.0 + 0.2j
    vocal[:, 5:, :] = 0.7 - 0.1j
    candidates = np.stack([
        accompaniment,
        accompaniment + 0.5 * vocal,
    ])
    G, _, unary, info = source_coordinate_quadratics(
        candidates, accompaniment, vocal, 44100, config()
    )
    assert G.shape == (1, 1, 2, 2)
    assert np.linalg.eigvalsh(G[0, 0]).min() >= -1e-9
    assert unary[0, 0, 0] < unary[0, 0, 1]
    assert info["source_coordinate_cells"] == 1
    assert info["direct_fallback_cells"] == 0


def test_zero_vocal_cell_uses_exact_direct_fallback():
    C, F, N = 2, 9, 4
    accompaniment = np.ones((C, F, N), dtype=np.complex128)
    vocal = np.zeros_like(accompaniment)
    candidates = np.stack([
        accompaniment,
        0.5 * accompaniment,
    ])
    G, _, unary, info = source_coordinate_quadratics(
        candidates, accompaniment, vocal, 44100, config()
    )
    assert np.all(np.isfinite(G))
    assert unary[0, 0, 0] < unary[0, 0, 1]
    assert info["source_coordinate_cells"] == 0
    assert info["direct_fallback_cells"] == 1


def test_reconstruction_uses_one_stereo_candidate_or_convex_vector_per_cell():
    specs = np.zeros((2, 2, 4, 4), dtype=np.complex128)
    specs[0] = 1.0
    specs[1] = 3.0
    info = {
        "time_slices": [(0, 2), (2, 4)],
        "band_bin_slices": [(0, 2), (2, 4)],
    }
    labels = np.array([[0, 1], [1, 0]])
    medoid = reconstruct_medoid(specs, labels, info)
    assert np.all(medoid[:, 0:2, 0:2] == 1.0)
    assert np.all(medoid[:, 2:4, 0:2] == 3.0)
    assert np.all(medoid[:, 0:2, 2:4] == 3.0)
    assert np.all(medoid[:, 2:4, 2:4] == 1.0)

    weights = np.full((2, 2, 2), 0.5)
    convex = reconstruct_convex(specs, weights, info)
    assert np.all(convex == 2.0)
