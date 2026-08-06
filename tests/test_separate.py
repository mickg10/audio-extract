import numpy as np

from audio_extract.separate import _normalize_stem_name, _residual_primary


def test_stem_tag_beats_model_name_collision():
    # Regression (real-opera GPU run): model names containing "vocal" made BOTH
    # stems normalize to "vocals", with glob order deciding which audio won.
    assert _normalize_stem_name("canonical.f32_(Instrumental)_Kim_Vocal_2") == "instrumental"
    assert _normalize_stem_name("canonical.f32_(Vocals)_Kim_Vocal_2") == "vocals"
    assert _normalize_stem_name("mix_(Instrumental)_vocals_mel_band_roformer") == "instrumental"
    assert _normalize_stem_name("mix_(Vocals)_vocals_mel_band_roformer") == "vocals"
    # no tag -> whole-name fallback still works
    assert _normalize_stem_name("vocals") == "vocals"
    assert _normalize_stem_name("something_inst") == "instrumental"


def test_residual_primary_prefers_vocals_no_truthiness_error():
    # Regression: `_stem("vocals") or _stem("instrumental")` raised
    # "truth value of an array is ambiguous" for multi-element stems.
    v = np.zeros((100, 2))
    i = np.ones((100, 2))
    assert _residual_primary({"vocals": v, "instrumental": i}) is v
    assert _residual_primary({"instrumental": i}) is i
    assert _residual_primary({}) is None
