import numpy as np
import pytest

from audio_extract.classical_surrogate_alignment import (
    MODE_DIRECT,
    MODE_SOURCE,
    SurrogateConfig,
    evaluate_surrogate,
    pairwise_false_safe,
    weighted_cvar,
)


def sources(frames=4096, scale=1.0):
    t = np.arange(frames, dtype=np.float64) / 44_100.0
    accompaniment = scale * np.column_stack((
        0.3 * np.sin(2 * np.pi * 220 * t),
        0.25 * np.sin(2 * np.pi * 330 * t),
    ))
    vocal = scale * np.column_stack((
        0.12 * np.sin(2 * np.pi * 610 * t),
        0.10 * np.sin(2 * np.pi * 680 * t),
    ))
    return accompaniment.astype("float64"), vocal.astype("float64")


def config(**kwargs):
    values = dict(
        sample_rate_hz=44_100,
        tile_seconds=0.02,
        hop_seconds=0.01,
        tail_fraction=0.10,
    )
    values.update(kwargs)
    return SurrogateConfig(**values)


def test_perfect_vocal_estimate_has_near_zero_source_risks():
    a, v = sources()
    report = evaluate_surrogate(v, a, v, config=config())
    assert report.source_coordinate_tiles == report.total_tiles
    assert report.direct_fallback_tiles == 0
    assert report.recall_cvar == pytest.approx(0.0, abs=1e-12)
    assert report.theft_cvar == pytest.approx(0.0, abs=1e-12)
    assert report.artifact_cvar == pytest.approx(0.0, abs=1e-12)


def test_missed_voice_increases_recall_without_theft():
    a, v = sources()
    report = evaluate_surrogate(0.5 * v, a, v, config=config())
    assert report.recall_cvar > 0.0
    assert report.theft_cvar == pytest.approx(0.0, abs=1e-10)


def test_stolen_orchestra_increases_theft():
    a, v = sources()
    report = evaluate_surrogate(v + 0.25 * a, a, v, config=config())
    assert report.theft_cvar > 0.0
    assert report.recall_cvar == pytest.approx(0.0, abs=1e-8)


def test_orthogonal_noise_increases_artifact():
    a, v = sources()
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.02, size=v.shape)
    report = evaluate_surrogate(v + noise, a, v, config=config())
    assert report.artifact_cvar > 0.0


def test_collinear_sources_use_direct_fallback_and_do_not_disappear():
    a, _ = sources()
    v = 0.5 * a
    estimate = 0.8 * v
    report = evaluate_surrogate(estimate, a, v, config=config())
    assert report.source_coordinate_tiles == 0
    assert report.direct_fallback_tiles == report.total_tiles
    assert report.direct_fallback_cvar is not None
    assert all(tile.mode == MODE_DIRECT for tile in report.tiles)


def test_silent_tile_with_routed_noise_uses_direct_fallback():
    frames = 2048
    a = np.zeros((frames, 2))
    v = np.zeros_like(a)
    estimate = np.zeros_like(a)
    estimate[100:200] = 0.1
    report = evaluate_surrogate(estimate, a, v, config=config())
    assert report.direct_fallback_tiles == report.total_tiles
    assert report.direct_fallback_max > 0


def test_weighted_cvar_exposes_one_catastrophic_tile():
    values = [0.0] * 9 + [10.0]
    weights = [1.0] * 10
    assert np.mean(values) == 1.0
    assert weighted_cvar(values, weights, 0.10) == pytest.approx(10.0)
    assert weighted_cvar(values, weights, 0.20) == pytest.approx(5.0)


def test_surrogate_is_scale_invariant_while_floors_are_inactive():
    reports = []
    for scale in (1.0, 1e-3, 1e3):
        a, v = sources(scale=scale)
        reports.append(
            evaluate_surrogate(0.7 * v + 0.1 * a, a, v, config=config())
        )
    for field in ("recall_cvar", "theft_cvar", "artifact_cvar"):
        values = [getattr(report, field) for report in reports]
        assert max(values) - min(values) < 1e-9


def test_mode_counts_form_complete_partition():
    a, v = sources(frames=5000)
    v[:1000] = 0.0
    report = evaluate_surrogate(v, a, v, config=config())
    assert (
        report.source_coordinate_tiles + report.direct_fallback_tiles
        == report.total_tiles
    )
    assert {tile.mode for tile in report.tiles} <= {
        MODE_SOURCE,
        MODE_DIRECT,
    }


def test_false_safe_detection_is_axis_specific():
    failures = pairwise_false_safe(
        {"recall_cvar": 1.0, "theft_cvar": 1.0},
        {"recall_cvar": 0.5, "theft_cvar": 1.1},
        {
            "retained_voice_db_p90": -10.0,
            "event_hole_db_p90": 2.0,
        },
        {
            "retained_voice_db_p90": -8.0,
            "event_hole_db_p90": 2.1,
        },
    )
    assert len(failures) == 1
    assert "retained_voice" in failures[0]


def test_invalid_config_and_grid_are_refused():
    with pytest.raises(ValueError):
        config(tail_fraction=1.1).validate()
    a, v = sources()
    with pytest.raises(ValueError, match="sample grids"):
        evaluate_surrogate(v[:-1], a, v, config=config())
