import numpy as np
import soundfile as sf

from audio_extract.separate import Separator, _normalize_stem_name, _residual_primary


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


def test_owned_separator_scratch_is_removed_after_read(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    class Backend:
        def separate(self, _audio_path):
            sf.write(scratch / "mix_(Vocals)_model.wav", np.zeros((32, 2), dtype="float32"),
                     44_100, subtype="FLOAT")

    separator = Separator.__new__(Separator)
    separator._out = scratch
    separator._owns_out = True
    separator._sep = Backend()
    separator.model_filename = "model.ckpt"
    separator.model_sha256 = "sha256:model"
    separator.overlap = 8
    result = separator.separate_file(tmp_path / "unused.wav")
    assert result.stems["vocals"].shape == (32, 2)
    assert not scratch.exists()
