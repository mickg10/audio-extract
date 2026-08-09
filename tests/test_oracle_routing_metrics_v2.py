import numpy as np
import pytest

pytest.importorskip("librosa")

from audio_extract.oracle_routing_metrics_v2 import (
    complete_work_metrics,
    hall_tail_metric,
    no_vocal_false_positive_energy_ratio,
    route_boundary_metrics,
    stereo_metrics,
    transient_metrics,
    worst_identifiable_event,
)

SR = 8_000


def exact_scene(seconds=4.0):
    frames = round(seconds * SR)
    time = np.arange(frames) / SR
    # Two phrase-like accompaniment envelopes with explicit decays.
    envelope = np.zeros(frames)
    envelope[:SR] = 1.0
    envelope[SR:2 * SR] = np.linspace(1.0, 0.02, SR)
    envelope[2 * SR:3 * SR] = 0.15
    envelope[3 * SR:] = np.linspace(0.8, 0.03, frames - 3 * SR)
    left = envelope * (
        0.55 * np.sin(2 * np.pi * 230 * time)
        + 0.20 * np.sin(2 * np.pi * 610 * time)
    )
    right = envelope * (
        0.45 * np.sin(2 * np.pi * 230 * time + 0.2)
        + 0.15 * np.sin(2 * np.pi * 880 * time)
    )
    accompaniment = np.column_stack((left, right)).astype("float32")
    vocal_envelope = np.zeros(frames)
    vocal_envelope[SR // 2:3 * SR // 2] = 0.35
    vocal_envelope[5 * SR // 2:7 * SR // 2] = 0.25
    vocal = np.column_stack((
        vocal_envelope * np.sin(2 * np.pi * 440 * time),
        vocal_envelope * np.sin(2 * np.pi * 440 * time + 0.05),
    )).astype("float32")
    return accompaniment, vocal


def test_exact_candidate_has_near_zero_complete_work_distortion():
    accompaniment, vocal = exact_scene()
    metrics, labels = complete_work_metrics(
        accompaniment, accompaniment, vocal, SR
    )
    assert metrics["event_hole_db_p90"] < 1e-4
    assert metrics["artifact_ratio_p90"] < 1e-4
    assert metrics["stereo_width_dev_db/v2"] < 1e-8
    assert metrics["interchannel_coherence_dev/v2"] < 1e-8
    assert metrics["hall_tail_dev_db/v2"] < 1e-8
    assert metrics["transient_loss_db/v2"] < 1e-8
    assert metrics["transient_excess_db/v2"] < 1e-8
    event = worst_identifiable_event(labels)
    assert event["available"]
    assert event["composite_risk"] < 1e-3


def test_half_level_candidate_exposes_hall_transient_and_hole_loss():
    accompaniment, vocal = exact_scene()
    candidate = 0.5 * accompaniment
    metrics, _ = complete_work_metrics(candidate, accompaniment, vocal, SR)
    assert metrics["event_hole_db_p90"] > 5.0
    assert metrics["hall_tail_dev_db/v2"] > 5.0
    assert metrics["transient_loss_db/v2"] > 5.0


def test_stereo_collapse_is_detected():
    accompaniment, _ = exact_scene()
    mono = accompaniment.mean(axis=1)
    candidate = np.column_stack((mono, mono)).astype("float32")
    metrics = stereo_metrics(candidate, accompaniment, SR)
    assert metrics["stereo_width_dev_db/v2"] > 1.0
    assert metrics["interchannel_coherence_dev/v2"] > 0.01


def test_no_vocal_false_positive_ratio_has_exact_interpretation():
    accompaniment, _ = exact_scene()
    assert no_vocal_false_positive_energy_ratio(
        accompaniment, accompaniment
    ) == pytest.approx(0.0)
    assert no_vocal_false_positive_energy_ratio(
        accompaniment, 0.5 * accompaniment
    ) == pytest.approx(0.25, rel=1e-6)


def test_route_boundary_metric_is_zero_frequency_residual_for_exact_target():
    accompaniment, _ = exact_scene()
    plan = np.zeros((3, 3, 2), dtype=np.float64)
    for time in range(3):
        for band in range(3):
            plan[time, band, (time + band) % 2] = 1.0
    result = route_boundary_metrics(
        accompaniment,
        accompaniment,
        SR,
        time_frame_ranges=((0, 100), (100, 200), (200, 251)),
        frequency_bin_ranges=((0, 10), (10, 20), (20, 33)),
        n_fft=64,
        hop_length=128,
        plan=plan,
    )
    assert result["frequency_boundary_count"] == 2
    assert result["max_frequency_ringing_ratio"] == pytest.approx(0.0)
    assert np.isfinite(result["max_time_boundary_jump_over_p99_derivative"])


def test_inserted_time_step_increases_boundary_ratio():
    accompaniment, _ = exact_scene()
    boundary = SR
    candidate = accompaniment.copy()
    candidate[boundary:] += 0.5
    plan = np.zeros((2, 2, 2), dtype=np.float64)
    plan[0, :, 0] = 1.0
    plan[1, :, 1] = 1.0
    result = route_boundary_metrics(
        candidate,
        accompaniment,
        SR,
        time_frame_ranges=((0, boundary // 16),
                           (boundary // 16, len(candidate) // 16 + 1)),
        frequency_bin_ranges=((0, 16), (16, 33)),
        n_fft=64,
        hop_length=16,
        plan=plan,
    )
    assert result["max_time_boundary_jump_over_p99_derivative"] > 1.0


def test_inactive_partition_edges_are_not_scored_as_route_boundaries():
    accompaniment, _ = exact_scene()
    plan = np.ones((2, 2, 1), dtype=np.float64)
    result = route_boundary_metrics(
        accompaniment,
        accompaniment,
        SR,
        time_frame_ranges=((0, 100), (100, 251)),
        frequency_bin_ranges=((0, 16), (16, 33)),
        n_fft=64,
        hop_length=128,
        plan=plan,
    )
    assert result["time_partition_boundary_count"] == 1
    assert result["frequency_partition_boundary_count"] == 1
    assert result["time_boundary_count"] == 0
    assert result["frequency_boundary_count"] == 0


def test_missing_decay_tail_is_refused_instead_of_using_quiet_fallback():
    constant = np.full((4 * SR, 2), 0.2, dtype="float32")
    with pytest.raises(ValueError, match="decay-tail evidence"):
        hall_tail_metric(constant, constant, SR)


def test_metric_grid_mismatch_is_refused():
    accompaniment, vocal = exact_scene()
    with pytest.raises(ValueError, match="grids differ"):
        complete_work_metrics(
            accompaniment[:-1], accompaniment, vocal, SR
        )


def test_hall_and_transient_metrics_have_event_coverage():
    accompaniment, _ = exact_scene()
    hall = hall_tail_metric(accompaniment, accompaniment, SR)
    transient = transient_metrics(accompaniment, accompaniment, SR)
    assert hall["hall_tail_frames/v2"] > 0
    assert transient["transient_event_frames/v2"] > 0
