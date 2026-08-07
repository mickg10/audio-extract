"""Reference-free judge features: the extractor must (a) fire on residual voice /
artifacts and (b) stay quiet on a clean instrumental — i.e. the features carry
signal the trained judge can map to the exact-reference labels."""
import numpy as np

from audio_extract import fixtures as fx
from audio_extract import judge_features as jf

SR = 44100


def test_feature_vector_shape_and_order():
    orch = fx.synth_orchestra(SR, 2.0)
    feats = jf.reference_free_features(orch, SR)
    v = jf.feature_vector(feats)
    assert v.shape == (len(jf.FEATURE_NAMES),)
    assert set(feats) == set(jf.FEATURE_NAMES)
    assert np.all(np.isfinite(v))


def test_residual_voice_raises_voice_features():
    # controlled: broadband-noise "orchestra" (low harmonic salience) vs the same
    # with a sustained harmonic comb (a residual singer at F0=220 + partials).
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * 3.0)) / SR
    noise = rng.standard_normal(len(t)) * 0.1
    orch = np.column_stack([noise, noise]).astype(np.float64)
    comb = sum(np.sin(2 * np.pi * 220.0 * k * t) / k for k in range(1, 8)) * 0.15
    contaminated = orch + np.column_stack([comb, comb])
    fc = jf.reference_free_features(orch, SR)
    fk = jf.reference_free_features(contaminated, SR)
    # a residual harmonic singer must raise harmonic salience and/or voiced prob
    assert (fk["residual_voice_harmonic_salience"] > fc["residual_voice_harmonic_salience"]
            or fk["singing_voice_prob_on_instrumental"] > fc["singing_voice_prob_on_instrumental"] + 0.05)


def test_reduction_features_relative_to_mixture():
    orch = fx.synth_orchestra(SR, 2.0)
    vocal, _ = fx.synth_vocal(SR, 2.0)
    n = min(len(orch), len(vocal))
    mix = orch[:n] + vocal[:n]
    # perfect instrumental (== orch) vs an over-subtracted one (half energy)
    good = jf.reference_free_features(orch[:n], SR, mixture=mix)
    over = jf.reference_free_features(orch[:n] * 0.5, SR, mixture=mix)
    assert over["broadband_reduction_db"] > good["broadband_reduction_db"]  # more removed


def test_ensemble_disagreement_feature():
    orch = fx.synth_orchestra(SR, 2.0)
    vocal, _ = fx.synth_vocal(SR, 2.0)
    n = min(len(orch), len(vocal))
    agree = [vocal[:n], vocal[:n], vocal[:n]]                 # identical estimates
    disagree = [vocal[:n], orch[:n], 0.5 * (vocal[:n] + orch[:n])]  # divergent
    fa = jf.reference_free_features(orch[:n], SR, member_vocals=agree)
    fd = jf.reference_free_features(orch[:n], SR, member_vocals=disagree)
    assert fd["ensemble_member_disagreement"] > fa["ensemble_member_disagreement"]
    assert fa["ensemble_member_disagreement"] < 1e-6          # perfect agreement -> ~0


def test_all_features_present_and_named():
    # the input contract is stable + append-only
    assert len(jf.FEATURE_NAMES) == len(set(jf.FEATURE_NAMES))
    feats = jf.reference_free_features(fx.synth_orchestra(SR, 1.0), SR)
    assert jf.feature_vector(feats).dtype == np.float64
