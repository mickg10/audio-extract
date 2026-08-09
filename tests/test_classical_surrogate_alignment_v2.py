import math

import numpy as np
import pytest

from audio_extract.classical_surrogate_alignment_v2 import (
    FALLBACK_REASONS,
    MODE_DIRECT,
    MODE_SOURCE,
    SurrogateConfigV2,
    evaluate_surrogate_v2,
    pairwise_false_safe_v2,
    partition_measure_weights,
    tile_ranges,
    weighted_cvar,
)


def _sources(frames=4096, scale=1.0):
    t = np.arange(frames, dtype=np.float64) / 44_100.0
    accompaniment = scale * np.column_stack(
        (
            0.30 * np.sin(2 * np.pi * 220 * t),
            0.25 * np.sin(2 * np.pi * 330 * t),
        )
    )
    vocal = scale * np.column_stack(
        (
            0.12 * np.sin(2 * np.pi * 610 * t),
            0.10 * np.sin(2 * np.pi * 680 * t),
        )
    )
    return accompaniment, vocal


def _config(**overrides):
    values = {
        "sample_rate_hz": 44_100,
        "tile_seconds": 0.02,
        "hop_seconds": 0.01,
        "tail_fraction": 0.10,
    }
    values.update(overrides)
    return SurrogateConfigV2(**values)


def test_partition_measure_covers_forced_final_tile_exactly_once():
    ranges = tile_ranges(frames=25, tile=10, hop=6)
    assert ranges == ((0, 10), (6, 16), (12, 22), (15, 25))
    measures = partition_measure_weights(25, ranges)
    assert all(value > 0 for value in measures)
    assert sum(measures) == pytest.approx(25.0, abs=1e-12)


def test_v2_perfect_estimate_and_source_mode_are_near_zero():
    accompaniment, vocal = _sources()
    report = evaluate_surrogate_v2(vocal, accompaniment, vocal, config=_config())
    assert report.source_coordinate_tiles == report.total_tiles
    assert report.direct_fallback_tiles == 0
    assert report.source_coordinate_measure_fraction == pytest.approx(1.0)
    assert report.recall_cvar == pytest.approx(0.0, abs=1e-12)
    assert report.theft_cvar == pytest.approx(0.0, abs=1e-12)
    assert report.artifact_cvar == pytest.approx(0.0, abs=1e-12)
    assert all(tile.mode == MODE_SOURCE for tile in report.tiles)


@pytest.mark.parametrize(
    ("reason", "make_sources"),
    (
        ("no_vocal", lambda a, v: (a, np.zeros_like(v))),
        ("vocal_only", lambda a, v: (np.zeros_like(a), v)),
        ("silent", lambda a, v: (np.zeros_like(a), np.zeros_like(v))),
        ("ill_conditioned", lambda a, v: (a, 0.5 * a)),
    ),
)
def test_v2_each_unidentifiable_basis_has_one_fallback_reason(reason, make_sources):
    accompaniment, vocal = _sources(frames=1024)
    accompaniment, vocal = make_sources(accompaniment, vocal)
    report = evaluate_surrogate_v2(
        0.8 * vocal,
        accompaniment,
        vocal,
        config=_config(tile_seconds=1.0, hop_seconds=1.0),
    )
    assert report.total_tiles == 1
    assert report.direct_fallback_tiles == 1
    assert report.tiles[0].mode == MODE_DIRECT
    assert report.tiles[0].fallback_reason == reason
    assert report.fallback_tile_counts[reason] == 1
    assert report.fallback_measure_fractions[reason] == pytest.approx(1.0)
    assert set(report.fallback_tile_counts) == set(FALLBACK_REASONS)


def test_v2_mode_and_measure_partitions_remain_complete():
    accompaniment, vocal = _sources(frames=5000)
    vocal[:1500] = 0
    report = evaluate_surrogate_v2(vocal, accompaniment, vocal, config=_config())
    assert report.source_coordinate_tiles + report.direct_fallback_tiles == report.total_tiles
    assert sum(tile.measure_frames for tile in report.tiles) == pytest.approx(report.frames)
    assert report.source_coordinate_measure_fraction + sum(
        report.fallback_measure_fractions.values()
    ) == pytest.approx(1.0)


def test_v2_risks_are_scale_invariant_while_relative_floor_is_active():
    reports = []
    for scale in (1e-3, 1.0, 1e3):
        accompaniment, vocal = _sources(scale=scale)
        reports.append(
            evaluate_surrogate_v2(
                0.7 * vocal + 0.1 * accompaniment,
                accompaniment,
                vocal,
                config=_config(),
            )
        )
    for field in ("recall_cvar", "theft_cvar", "artifact_cvar"):
        values = [getattr(report, field) for report in reports]
        assert max(values) - min(values) < 1e-9


def test_v2_weighted_cvar_and_false_safe_validation():
    assert weighted_cvar([0.0] * 9 + [10.0], [1.0] * 10, 0.10) == pytest.approx(10.0)
    failures = pairwise_false_safe_v2(
        {"recall_cvar": 1.0, "theft_cvar": 1.0},
        {"recall_cvar": 0.5, "theft_cvar": 1.1},
        {"retained_voice_db_p90": -10.0, "event_hole_db_p90": 2.0},
        {"retained_voice_db_p90": -8.0, "event_hole_db_p90": 2.1},
    )
    assert len(failures) == 1
    assert "retained_voice" in failures[0]
    with pytest.raises(ValueError, match="finite/nonnegative"):
        pairwise_false_safe_v2({}, {}, {}, {}, external_regression_limit_db=math.nan)


def test_v2_rejects_invalid_grid_config_and_nonfinite_samples():
    accompaniment, vocal = _sources()
    with pytest.raises(ValueError, match="sample grids"):
        evaluate_surrogate_v2(vocal[:-1], accompaniment, vocal, config=_config())
    invalid = vocal.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_surrogate_v2(invalid, accompaniment, vocal, config=_config())
    with pytest.raises(ValueError, match="max_condition"):
        _config(max_condition=1).validate()
    with pytest.raises(ValueError, match="unknown surrogate representation"):
        _config(representation="unknown").validate()
