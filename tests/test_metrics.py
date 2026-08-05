"""Metamorphic monotonicity tests (docs/v2 §benchmark).

A metric that does not move monotonically with its injected defect is not ready
to rank candidates. Each test builds a severity ladder and asserts the metric's
primary scalar is monotonic across it.
"""
import functools

import numpy as np

from audio_extract import fixtures as fx
from audio_extract import metrics as mx

SR = 44100
DUR = 3.0


@functools.lru_cache(maxsize=1)
def _stems():
    vocal, activity = fx.synth_vocal(SR, DUR)
    orch = fx.synth_orchestra(SR, DUR)
    n = min(len(vocal), len(orch), len(activity))
    return vocal[:n], orch[:n], activity[:n]


def _increasing(vals, label):
    assert all(b > a for a, b in zip(vals, vals[1:])), f"{label} not increasing: {vals}"


def _decreasing(vals, label):
    assert all(b < a for a, b in zip(vals, vals[1:])), f"{label} not decreasing: {vals}"


def test_pumping_monotonic():
    _, orch, activity = _stems()
    vals = [
        mx.pumping(fx.inject_pumping(orch, activity, d), SR)["pump_depth_db"]
        for d in (0.0, 3.0, 6.0, 12.0)
    ]
    _increasing(vals, "pumping")


def test_leakage_monotonic():
    vocal, orch, _ = _stems()
    vals = [
        mx.leakage(fx.add_vocal_bleed(orch, vocal, g), vocal, SR)["leakage"]
        for g in (-60.0, -24.0, -18.0, -12.0, -6.0)
    ]
    _increasing(vals, "leakage")


def test_brightness_deviation_monotonic():
    _, orch, _ = _stems()
    vals = [
        mx.brightness_deviation(fx.high_shelf(orch, SR, g), orch, SR)["brightness_deviation"]
        for g in (0.0, -3.0, -6.0, -12.0)
    ]
    _increasing(vals, "brightness_deviation")


def test_fullness_holes_monotonic():
    _, orch, _ = _stems()
    vals = [
        mx.fullness(fx.spectral_hole(orch, SR, 2000, 4000, d), orch, SR)["holes_db"]
        for d in (0.0, 3.0, 6.0, 12.0)
    ]
    _increasing(vals, "holes")


def test_hall_truncation_monotonic():
    _, orch, activity = _stems()
    vals = [
        mx.hall(fx.truncate_hall(orch, activity, SR, c), orch, SR, activity)["hall_damage_db"]
        for c in (0.0, 50.0, 150.0, 300.0)
    ]
    _increasing(vals, "hall_damage")


def test_stereo_collapse_monotonic():
    _, orch, _ = _stems()
    vals = [mx.stereo(fx.narrow_stereo(orch, a))["stereo_width"] for a in (0.0, 0.3, 0.6, 1.0)]
    _decreasing(vals, "stereo_width")


def _impulse_train():
    n = int(SR * DUR)
    rng = np.random.default_rng(2)
    sig = np.zeros(n)
    sig[:: int(0.1 * SR)] = 1.0  # an attack every 100 ms
    stereo = np.column_stack([sig, sig]) + 0.005 * rng.standard_normal((n, 2))
    return stereo


def test_transient_smear_monotonic():
    # Crest factor (peak/rms) is level-invariant and the robust transient measure.
    # On an attack train, temporal blur drives crest ∝ 1/√k — strictly decreasing.
    bed = _impulse_train()
    vals = [mx.transients(fx.smear_transients(bed, SR, b), SR)["crest_factor"]
            for b in (0.0, 1.0, 3.0, 8.0)]
    _decreasing(vals, "crest_factor")


def test_hard_checks():
    _, orch, _ = _stems()
    assert mx.hard_checks(orch, SR)["ok"] is True
    clipped = np.ones((1000, 2))
    assert "clipping" in mx.hard_checks(clipped, SR)["problems"]
    nanned = orch.copy(); nanned[0, 0] = np.nan
    assert "nan_or_inf" in mx.hard_checks(nanned, SR)["problems"]
    assert "wrong_sample_rate" in mx.hard_checks(orch, SR, expected_sr=48000)["problems"]


def test_gain_match_equalizes_rms():
    _, orch, _ = _stems()
    quiet = orch * 0.1
    matched = mx.gain_match(quiet, orch)
    r_matched = np.sqrt(np.mean(matched ** 2))
    r_ref = np.sqrt(np.mean(orch ** 2))
    assert abs(r_matched - r_ref) / r_ref < 1e-6


def test_measure_all_keys():
    vocal, orch, activity = _stems()
    res = mx.measure_all(orch, SR, reference=orch, vocal_ref=vocal, activity=activity)
    assert {"hard_checks", "pumping", "transients", "stereo", "leakage", "fullness", "brightness", "hall"} <= set(res)
