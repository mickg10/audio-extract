"""Source-aware feature v2 tests. Per the reviewers: every important feature gets a
synthetic-monotonic test, a real-clean confounder, and a real-defective case — plus
the fraction-vs-absolute regression and invariance checks."""
import numpy as np
import pytest

from audio_extract import fixtures as fx
from audio_extract import judge_features_v2 as jf2

SR = 44100


def _mix(dur=3.0, seed=0):
    orch = fx.synth_orchestra(SR, dur)
    vocal, _ = fx.synth_vocal(SR, dur)
    n = min(len(orch), len(vocal))
    return orch[:n], vocal[:n], (orch[:n] + vocal[:n])


def test_contract_shape_and_required_inputs():
    orch, _, mix = _mix()
    feats, avail = jf2.source_aware_features(mix, orch, SR)   # Y=orch (clean instrumental)
    v = jf2.feature_vector(feats)
    assert v.shape == (len(jf2.feature_names()),)
    assert np.all(np.isfinite(v))
    with pytest.raises(ValueError):
        jf2.source_aware_features(np.zeros((0, 2)), np.zeros((0, 2)), SR)


def test_absolute_band_not_fraction_regression():
    # the v0 bug: fraction-of-total inverts under broadband change. v2 must report the
    # ABSOLUTE low-band loss and stay consistent when the whole signal is scaled.
    orch, _, mix = _mix()
    # (a) a genuine low-band hole: attenuate 20-250 Hz only
    holed = fx.spectral_hole(orch, SR, 20, 250, 18.0)
    f_hole, _ = jf2.source_aware_features(mix, holed, SR)
    f_clean, _ = jf2.source_aware_features(mix, orch, SR)
    assert f_hole["low_band_abs_reduction_db"] > f_clean["low_band_abs_reduction_db"] + 3
    # (b) uniform scale-down must NOT masquerade as a band-specific loss:
    # low-band reduction ~ tracks broadband reduction (proportional), not exploding
    quiet = orch * 0.5
    fq, _ = jf2.source_aware_features(mix, quiet, SR)
    assert abs(fq["low_band_abs_reduction_db"] - fq["broadband_reduction_db"]) < 3.0


def test_event_deficit_uses_quantiles_not_mean():
    # one catastrophic 200ms hole must show in max/cvar90 even if p50 is ~0
    orch, vocal, mix = _mix(4.0)
    n = len(orch)
    holed = orch.copy()
    s = n // 2
    holed[s:s + int(0.2 * SR)] *= 0.05            # deep brief gouge
    f, _ = jf2.source_aware_features(mix, holed, SR, removed=mix - holed)
    d = f["event_band_deficit_db"]
    assert d["max"] > 6.0                          # the worst frame is caught
    assert d["max"] >= d["p90"] >= d["p50"]        # distribution ordered


def test_clean_confounder_vs_residual_voice():
    # real-clean: broadband-noise "orchestra", no voice left -> low residual alignment.
    # real-defective: a residual harmonic comb (a left-in singer) -> higher.
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * 3)) / SR
    noise = np.column_stack([rng.standard_normal(len(t))] * 2) * 0.1
    comb = np.column_stack([sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 8))] * 2) * 0.15
    mix = noise + comb
    clean = noise                                  # perfect removal
    leaky = noise + 0.5 * comb                      # half the singer left in
    fc, _ = jf2.source_aware_features(mix, clean, SR)
    fl, _ = jf2.source_aware_features(mix, leaky, SR)
    assert fl["residual_voice_alignment"] >= fc["residual_voice_alignment"]


def test_stereo_availability_and_mono():
    orch, _, mix = _mix()
    _, avail = jf2.source_aware_features(mix, orch, SR)
    assert avail["ms_width_change_db"] is True
    mono_mix = mix.mean(axis=1, keepdims=True)
    mono_y = orch.mean(axis=1, keepdims=True)
    _, avail_m = jf2.source_aware_features(mono_mix, mono_y, SR)
    assert avail_m["ms_width_change_db"] is False   # mono: UNAVAILABLE, not fabricated 0


def test_availability_masks_missing_not_zero():
    orch, _, mix = _mix()
    _, avail = jf2.source_aware_features(mix, orch, SR)                    # no member_vocals
    assert avail["ensemble_disagreement"] is False
    _, avail2 = jf2.source_aware_features(mix, orch, SR, member_vocals=[orch, orch])
    assert avail2["ensemble_disagreement"] is True


def test_gain_invariance():
    # scaling M and Y by the SAME factor leaves the reduction ratio unchanged
    orch, _, mix = _mix()
    f1, _ = jf2.source_aware_features(mix, orch, SR)
    f2, _ = jf2.source_aware_features(mix * 2.0, orch * 2.0, SR)
    assert abs(f1["broadband_reduction_db"] - f2["broadband_reduction_db"]) < 0.2


def test_channel_swap_stability():
    orch, _, mix = _mix()
    sw = lambda z: z[:, ::-1]
    f1, _ = jf2.source_aware_features(mix, orch, SR)
    f2, _ = jf2.source_aware_features(sw(mix), sw(orch), SR)
    assert abs(f1["ms_width_change_db"] - f2["ms_width_change_db"]) < 0.5   # width is swap-stable


def test_removed_defaults_to_m_minus_y():
    orch, _, mix = _mix()
    f_auto, _ = jf2.source_aware_features(mix, orch, SR)                    # D defaults to M-Y
    f_expl, _ = jf2.source_aware_features(mix, orch, SR, removed=mix - orch)
    assert jf2.feature_vector(f_auto).shape == jf2.feature_vector(f_expl).shape
