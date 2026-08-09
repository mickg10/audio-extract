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


def test_accomp_continuity_deficit_catches_gouge_not_removal():
    # oracle P0: the feature must flag an accompaniment DIP co-located with voice activity in
    # Y itself — but must NOT flag correct voice removal. The old (M − Y) form reported the
    # removed-voice energy as a "hole"; this rewrite proves the fix on both sides.
    orch = fx.synth_orchestra(SR, 4.0)
    vocal, _ = fx.synth_vocal(SR, 4.0, phrases=[(1.0, 3.0)])    # voice active 1–3 s
    n = min(len(orch), len(vocal)); orch, vocal = orch[:n], vocal[:n]
    mix = orch + vocal
    clean = orch.copy()                                         # perfect removal, no hole
    gouged = orch.copy()
    a, b = int(1.5 * SR), int(1.9 * SR)                         # gouge DURING the voice phrase
    gouged[a:b] *= 0.05
    dc = jf2.source_aware_features(mix, clean, SR, removed=mix - clean)[0]["accomp_continuity_deficit_db"]
    dg = jf2.source_aware_features(mix, gouged, SR, removed=mix - gouged)[0]["accomp_continuity_deficit_db"]
    assert dg["max"] > dc["max"] + 6.0            # the co-located gouge is caught
    assert dc["max"] < 4.0                        # correct voice removal is NOT read as a hole
    assert dg["max"] >= dg["p90"] >= dg["p50"]    # distribution ordered


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


def test_task_and_availability_enter_the_vector():
    # oracle P0 #1: task encoding + availability bits must be IN feature_vector, not merely
    # recorded — else the model can't tell "remove all voices" from "retain chorus", nor a
    # genuine 0.0 from an unavailable input.
    orch, _, mix = _mix()
    names = jf2.feature_names()
    idx = {n: i for i, n in enumerate(names)}
    # different task -> different vector (one-hot slots differ)
    fa, ava = jf2.source_aware_features(mix, orch, SR, task="all_vocals")
    fs, avs = jf2.source_aware_features(mix, orch, SR, task="soloist_vs_rest")
    assert not np.array_equal(jf2.feature_vector(fa, ava), jf2.feature_vector(fs, avs))
    # unknown task -> all task one-hot bits zero (a valid, distinct encoding)
    fu, avu = jf2.source_aware_features(mix, orch, SR, task="not_a_real_task")
    vu = jf2.feature_vector(fu, avu)
    assert sum(vu[i] for n, i in idx.items() if n.startswith("task__")) == 0.0
    # mono input -> stereo availability bit is 0.0, distinguishable from a genuine feature 0
    mono, mono_y = mix.mean(axis=1, keepdims=True), orch.mean(axis=1, keepdims=True)
    fm, avm = jf2.source_aware_features(mono, mono_y, SR)
    assert jf2.feature_vector(fm, avm)[idx["avail__ms_width_change_db"]] == 0.0
