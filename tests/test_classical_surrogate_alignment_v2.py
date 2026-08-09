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
    return accompaniment, vocal


def config(**kwargs):
    values = dict(
        sample_rate_hz=44_100,
        tile_seconds=0.02,
        hop_seconds=0.01,
        tail_fraction=0.10,
    )
    values.update(kwargs)
    return SurrogateConfigV2(**values)


def test_partition_measure_counts_every_frame_once_with_forced_final_tile():
    ranges = tile_ranges(101, tile=40, hop=25)
    assert ranges[-1] == (61, 101)
    weights = partition_measure_weights(101, ranges)
    assert sum(weights) == pytest.approx(101.0)
    assert all(weight > 0 for weight in weights)
    # Naive tile lengths would double count overlap.
    assert sum(stop - start for start, stop in ranges) > 101


def test_perfect_estimate_has_zero_source_risks_and_full_measure():
    a, v = sources()
    report = evaluate_surrogate_v2(v, a, v, config=config())
    assert report.source_coordinate_tiles == report.total_tiles
    assert report.source_coordinate_measure_fraction == pytest.approx(1.0)
    assert report.recall_cvar == pytest.approx(0.0, abs=1e-12)
    assert report.theft_cvar == pytest.approx(0.0, abs=1e-12)
    assert report.artifact_cvar == pytest.approx(0.0, abs=1e-12)


def test_recall_theft_and_artifact_axes_are_separate():
    a, v = sources()
    missed = evaluate_surrogate_v2(0.5 * v, a, v, config=config())
    stolen = evaluate_surrogate_v2(v + 0.25 * a, a, v, config=config())
    rng = np.random.default_rng(4)
    noisy = evaluate_surrogate_v2(
        v + rng.normal(0, 0.02, size=v.shape), a, v, config=config()
    )
    assert missed.recall_cvar > 0
    assert missed.theft_cvar == pytest.approx(0.0, abs=1e-10)
    assert stolen.theft_cvar > 0
    assert stolen.recall_cvar == pytest.approx(0.0, abs=1e-8)
    assert noisy.artifact_cvar > 0


@pytest.mark.parametrize(
    "kind,expected",
    (
        ("silent", "silent"),
        ("no_vocal", "no_vocal"),
        ("vocal_only", "vocal_only"),
        ("collinear", "ill_conditioned"),
    ),
)
def test_direct_fallback_reason_is_explicit(kind, expected):
    frames = 2048
    a, v = sources(frames)
    if kind == "silent":
        a[:] = 0; v[:] = 0
    elif kind == "no_vocal":
        v[:] = 0
    elif kind == "vocal_only":
        a[:] = 0
    elif kind == "collinear":
        v[:] = 0.5 * a
    estimate = 0.8 * v + 0.05 * a
    report = evaluate_surrogate_v2(estimate, a, v, config=config())
    assert report.source_coordinate_tiles == 0
    assert report.direct_fallback_tiles == report.total_tiles
    assert report.fallback_tile_counts[expected] == report.total_tiles
    assert report.fallback_measure_fractions[expected] == pytest.approx(1.0)
    assert report.direct_fallback_cvar_by_reason[expected] is not None
    assert all(tile.mode == MODE_DIRECT for tile in report.tiles)
    assert all(tile.fallback_reason == expected for tile in report.tiles)


def test_mode_measures_form_complete_partition():
    a, v = sources(frames=5000)
    v[:1500] = 0.0
    report = evaluate_surrogate_v2(v, a, v, config=config())
    assert report.source_coordinate_tiles + report.direct_fallback_tiles == report.total_tiles
    assert report.source_coordinate_measure_fraction + sum(
        report.fallback_measure_fractions.values()
    ) == pytest.approx(1.0)
    assert set(report.fallback_tile_counts) == set(FALLBACK_REASONS)
    assert {tile.mode for tile in report.tiles} <= {MODE_SOURCE, MODE_DIRECT}


def test_energy_floor_has_correct_scale_units():
    reports = []
    for scale in (1.0, 1e-3, 1e3):
        a, v = sources(scale=scale)
        reports.append(
            evaluate_surrogate_v2(
                0.7 * v + 0.1 * a, a, v, config=config()
            )
        )
    for field in ("recall_cvar", "theft_cvar", "artifact_cvar"):
        values = [getattr(report, field) for report in reports]
        assert max(values) - min(values) < 1e-9
    assert all(
        report.source_coordinate_tiles == reports[0].source_coordinate_tiles
        for report in reports
    )


def test_weighted_cvar_fractional_boundary_is_exact():
    assert weighted_cvar([10.0, 2.0], [1.0, 9.0], 0.15) == pytest.approx(
        (10.0 * 1.0 + 2.0 * 0.5) / 1.5
    )
    assert weighted_cvar([0.0] * 9 + [10.0], [1.0] * 10, 0.1) == 10.0


def test_false_safe_sign_is_lower_is_better_for_both_external_axes():
    failures = pairwise_false_safe_v2(
        {"recall_cvar": 1.0, "theft_cvar": 1.0},
        {"recall_cvar": 0.5, "theft_cvar": 0.5},
        {
            "retained_voice_db_p90": -10.0,
            "event_hole_db_p90": 2.0,
        },
        {
            "retained_voice_db_p90": -9.0,
            "event_hole_db_p90": 3.0,
        },
    )
    assert len(failures) == 2
    improved = pairwise_false_safe_v2(
        {"recall_cvar": 1.0, "theft_cvar": 1.0},
        {"recall_cvar": 0.5, "theft_cvar": 0.5},
        {
            "retained_voice_db_p90": -10.0,
            "event_hole_db_p90": 2.0,
        },
        {
            "retained_voice_db_p90": -11.0,
            "event_hole_db_p90": 1.0,
        },
    )
    assert improved == ()


def test_invalid_ranges_config_and_nonfinite_false_safe_are_refused():
    with pytest.raises(ValueError):
        partition_measure_weights(10, ((1, 5),))
    with pytest.raises(ValueError):
        config(representation="other").validate()
    with pytest.raises(ValueError, match="missing/non-finite"):
        pairwise_false_safe_v2(
            {"recall_cvar": 1.0},
            {"recall_cvar": 0.5},
            {"retained_voice_db_p90": -10.0},
            {"retained_voice_db_p90": float("nan")},
        )
