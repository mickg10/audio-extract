import numpy as np
import pytest

from audio_extract.classical_surrogate_alignment_v3 import (
    DEFAULT_TIME_SCALES,
    MODE_SOURCE,
    SurrogateConfigV3,
    evaluate_surrogate_v3,
    fit_complex_cell_v3,
    pairwise_false_safe_v3,
    partition_measure_weights,
    tile_ranges,
    weighted_cvar,
)


def sine_sources(frames=12_000, sample_rate=8_000, scale=1.0):
    time = np.arange(frames, dtype=np.float64) / sample_rate
    accompaniment = scale * np.column_stack((
        0.30 * np.sin(2 * np.pi * 220 * time)
        + 0.20 * np.sin(2 * np.pi * 2_400 * time),
        0.25 * np.sin(2 * np.pi * 330 * time)
        + 0.18 * np.sin(2 * np.pi * 2_700 * time),
    ))
    vocal = scale * np.column_stack((
        0.12 * np.sin(2 * np.pi * 900 * time),
        0.10 * np.sin(2 * np.pi * 1_100 * time),
    ))
    return accompaniment, vocal


def config(**kwargs):
    values = dict(
        sample_rate_hz=8_000,
        window_ms=40.0,
        stft_hop_ms=10.0,
        n_fft=512,
        bands_hz=((0, 500), (500, 1_500), (1_500, 3_900)),
        time_scales_seconds=(
            (0.5, 0.25), (0.125, 0.0625), (0.04, 0.02)
        ),
        tail_fraction=0.10,
    )
    values.update(kwargs)
    return SurrogateConfigV3(**values)


def external_limits():
    return {
        "retained_voice_db_p90": 0.5,
        "retained_voice_coef_p90": 0.05,
        "event_hole_db_p90": 0.5,
        "event_hole_db_max": 1.0,
        "alpha_error_p90": 0.05,
        "artifact_ratio_p90": 0.01,
    }


def test_partition_measure_and_cvar_are_measure_scale_invariant():
    ranges = tile_ranges(101, 40, 25)
    weights = partition_measure_weights(101, ranges)
    assert sum(weights) == pytest.approx(101.0)
    values = np.arange(100, dtype=np.float64)
    unit = np.ones(100)
    tiny = 1e-16 * unit
    assert weighted_cvar(values, unit, 0.10) == pytest.approx(94.5)
    assert weighted_cvar(values, tiny, 0.10) == pytest.approx(94.5)


def test_exact_complex_projection_recovers_coefficients_and_orthogonal_residue():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(32, 3)) + 1j * rng.normal(size=(32, 3))
    q, _ = np.linalg.qr(raw)
    a, v, residual = q[:, 0], q[:, 1], 0.03 * q[:, 2]
    estimate = 0.2 * a + (1.1 - 0.05j) * v + residual
    fit = fit_complex_cell_v3(
        estimate,
        a,
        v,
        source_fraction_floor=1e-12,
        max_condition=1e6,
        silent_reference_energy=1.0,
    )
    assert fit["mode"] == MODE_SOURCE
    assert fit["gamma_a"] == pytest.approx(0.2 + 0j, abs=1e-12)
    assert fit["gamma_v"] == pytest.approx(1.1 - 0.05j, abs=1e-12)
    assert fit["artifact_risk"] > 0
    assert fit["orthogonality_error"] < 1e-12


def test_perfect_estimate_is_zero_at_every_frozen_scale():
    a, v = sine_sources()
    report = evaluate_surrogate_v3(v, a, v, config=config())
    assert len(report.scale_reports) == len(DEFAULT_TIME_SCALES)
    for scale in report.scale_reports:
        assert scale.recall_cvar == pytest.approx(0.0, abs=1e-10)
        assert scale.theft_cvar == pytest.approx(0.0, abs=1e-10)
        assert scale.artifact_cvar == pytest.approx(0.0, abs=1e-10)
        assert scale.max_orthogonality_error is None or (
            scale.max_orthogonality_error < 1e-9
        )
        assert sum(scale.mode_counts.values()) == scale.total_cells
        assert sum(scale.mode_measure_fractions.values()) == pytest.approx(1.0)


def test_band_local_theft_detects_complete_loss_of_weak_low_band():
    frames = 16_000
    sr = 8_000
    time = np.arange(frames, dtype=np.float64) / sr
    low = 0.03 * np.column_stack((
        np.sin(2 * np.pi * 250 * time),
        np.sin(2 * np.pi * 300 * time),
    ))
    loud_high = np.column_stack((
        np.sin(2 * np.pi * 2_400 * time),
        0.8 * np.sin(2 * np.pi * 2_700 * time),
    ))
    accompaniment = low + loud_high
    vocal = 0.1 * np.column_stack((
        np.sin(2 * np.pi * 900 * time),
        np.sin(2 * np.pi * 1_100 * time),
    ))
    vocal_estimate = vocal + low
    report = evaluate_surrogate_v3(
        vocal_estimate, accompaniment, vocal, config=config()
    )
    assert max(
        scale.hole_axis_max or 0.0 for scale in report.scale_reports
    ) > 0.8


def test_fallback_reasons_feed_different_axes():
    frames = 12_000
    a, v = sine_sources(frames)
    zero = np.zeros_like(a)
    no_vocal = evaluate_surrogate_v3(
        0.25 * a, a, zero, config=config()
    )
    vocal_only = evaluate_surrogate_v3(
        0.5 * v, zero, v, config=config()
    )
    for scale in no_vocal.scale_reports:
        assert scale.mode_counts["no_vocal"] == scale.total_cells
        assert scale.hole_axis_max and scale.hole_axis_max > 0
        assert scale.voice_axis_max is None
    for scale in vocal_only.scale_reports:
        assert scale.mode_counts["vocal_only"] == scale.total_cells
        assert scale.voice_axis_max and scale.voice_axis_max > 0
        assert scale.hole_axis_max is None


def test_global_gain_does_not_change_mode_or_risks_down_to_tiny_scale():
    reports = []
    for scale in (1.0, 1e-6, 1e-12):
        a, v = sine_sources(scale=scale)
        reports.append(
            evaluate_surrogate_v3(
                0.7 * v + 0.1 * a, a, v, config=config()
            )
        )
    for scale_index in range(len(reports[0].scale_reports)):
        reference = reports[0].scale_reports[scale_index]
        for report in reports[1:]:
            current = report.scale_reports[scale_index]
            assert current.mode_counts == reference.mode_counts
            for field in (
                "recall_cvar",
                "theft_cvar",
                "artifact_cvar",
                "voice_axis_max",
                "hole_axis_max",
                "artifact_axis_max",
            ):
                assert getattr(current, field) == pytest.approx(
                    getattr(reference, field), rel=1e-8, abs=1e-12
                )


def test_multiscale_grid_is_frozen_and_short_scale_catches_brief_theft():
    a, v = sine_sources(frames=16_000)
    estimate = v.copy()
    estimate[4_000:4_400] += a[4_000:4_400]
    report = evaluate_surrogate_v3(estimate, a, v, config=config())
    identities = [
        (row.window_seconds, row.hop_seconds)
        for row in report.scale_reports
    ]
    assert identities == list(config().time_scales_seconds)
    fine = report.scale_reports[-1]
    coarse = report.scale_reports[0]
    assert fine.hole_axis_max is not None
    assert coarse.hole_axis_max is not None
    assert fine.hole_axis_max >= coarse.hole_axis_max


def _scaled_report(report, axis, factor):
    value = report.to_dict()
    for scale in value["scale_reports"]:
        for suffix in ("cvar", "max"):
            key = f"{axis}_axis_{suffix}"
            if scale[key] is not None:
                scale[key] *= factor
    return value


def test_false_safe_checks_every_exact_gate_and_fallback_axis():
    a, v = sine_sources()
    baseline = evaluate_surrogate_v3(
        0.5 * v, a, v, config=config()
    )
    candidate_voice = _scaled_report(baseline, "voice", 0.5)
    baseline_external = {
        "retained_voice_db_p90": -10.0,
        "retained_voice_coef_p90": 0.10,
        "event_hole_db_p90": 2.0,
        "event_hole_db_max": 3.0,
        "alpha_error_p90": 0.10,
        "artifact_ratio_p90": 0.02,
    }
    candidate_external = dict(baseline_external)
    candidate_external["retained_voice_coef_p90"] = 0.20
    failures = pairwise_false_safe_v3(
        baseline,
        candidate_voice,
        baseline_external,
        candidate_external,
        external_regression_limits=external_limits(),
    )
    assert any("retained_voice_coef_p90" in item for item in failures)

    no_vocal = evaluate_surrogate_v3(
        0.3 * a, a, np.zeros_like(v), config=config()
    )
    improved_hole = _scaled_report(no_vocal, "hole", 0.5)
    hole_external = dict(baseline_external)
    hole_candidate = dict(hole_external)
    hole_candidate["event_hole_db_max"] += 2.0
    failures = pairwise_false_safe_v3(
        no_vocal,
        improved_hole,
        hole_external,
        hole_candidate,
        external_regression_limits=external_limits(),
    )
    assert any("event_hole_db_max" in item for item in failures)


def test_surrogate_axis_must_improve_at_every_scale_not_cherry_pick():
    a, v = sine_sources()
    baseline = evaluate_surrogate_v3(
        0.5 * v, a, v, config=config()
    ).to_dict()
    candidate = _scaled_report(
        evaluate_surrogate_v3(
            0.5 * v, a, v, config=config()
        ),
        "voice",
        0.5,
    )
    candidate["scale_reports"][-1]["voice_axis_max"] = (
        baseline["scale_reports"][-1]["voice_axis_max"] * 2
    )
    external = {
        "retained_voice_db_p90": -10.0,
        "retained_voice_coef_p90": 0.10,
        "event_hole_db_p90": 2.0,
        "event_hole_db_max": 3.0,
        "alpha_error_p90": 0.10,
        "artifact_ratio_p90": 0.02,
    }
    regressed = dict(external)
    regressed["retained_voice_db_p90"] = -9.0
    assert pairwise_false_safe_v3(
        baseline,
        candidate,
        external,
        regressed,
        external_regression_limits=external_limits(),
    ) == ()


def test_invalid_configuration_and_missing_external_gate_are_refused():
    with pytest.raises(ValueError):
        config(time_scales_seconds=((0.1, 0.2),)).validate()
    a, v = sine_sources()
    report = evaluate_surrogate_v3(v, a, v, config=config())
    with pytest.raises(ValueError, match="cover every exact gate"):
        pairwise_false_safe_v3(
            report,
            report,
            {},
            {},
            external_regression_limits={},
        )
