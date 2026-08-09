import hashlib
import json

import numpy as np
import soundfile as sf

from audio_extract.separate import (
    SepOutput,
    Separator,
    _normalize_stem_name,
    _residual_primary,
    render_candidate,
    render_channel_map_candidate,
    render_ensemble_candidate,
    render_residual_candidate,
)
from audio_extract.storage import TrackLayout


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


def test_mono_channel_map_is_explicit_parent_for_separator_and_ensemble(tmp_path, monkeypatch):
    frames, sr = 2048, 44_100
    mono = np.linspace(-0.25, 0.25, frames, dtype="float32").reshape(-1, 1)
    layout = TrackLayout(tmp_path, "mono").ensure()
    sf.write(layout.source_dir / "canonical.f32.wav", mono, sr, subtype="FLOAT")
    source = {
        "track_id": "mono",
        "input_pcm_sha256": "sha256:mono-source",
        "sample_rate_hz": sr,
        "channel_layout": ["FC"],
        "frames": frames,
        "sample_format": "float32-le-interleaved",
    }
    (layout.source_dir / "source.json").write_text(json.dumps(source))

    mapped = render_channel_map_candidate(layout, source, code_commit="test")
    mapped_again = render_channel_map_candidate(layout, source, code_commit="test")
    assert mapped_again["recipe_id"] == mapped["recipe_id"]
    assert mapped_again["cached"] is True
    mapped_dir = layout.candidate_dir(mapped["recipe_id"])
    stereo, reopened_sr = sf.read(mapped_dir / "output.f32.wav", dtype="float32",
                                  always_2d=True)
    original, _ = sf.read(layout.source_dir / "canonical.f32.wav", dtype="float32",
                          always_2d=True)
    assert reopened_sr == sr and stereo.shape == (frames, 2)
    assert np.array_equal(stereo[:, 0], mono[:, 0])
    assert np.array_equal(stereo[:, 1], mono[:, 0])
    assert original.shape == (frames, 1) and np.array_equal(original, mono)
    map_recipe = json.loads((mapped_dir / "recipe.json").read_text())
    assert map_recipe["operation"]["type"] == "channel_map"
    assert map_recipe["effective_config"]["matrix_ppm"] == [[1_000_000], [1_000_000]]

    factors = {"a.ckpt": 0.1, "b.ckpt": 0.2, "c.ckpt": 0.3}

    class FakeSeparator:
        def __init__(self, model_filename, **_kwargs):
            self.model_filename = model_filename
            self.model_sha256 = "sha256:" + hashlib.sha256(model_filename.encode()).hexdigest()
            self.overlap = 8

        def separate_file(self, audio_path):
            samples, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
            assert samples.shape == (frames, 2)
            return SepOutput(
                {"vocals": samples * factors[self.model_filename]}, int(sample_rate),
                self.model_filename, self.model_sha256, self.overlap,
            )

    monkeypatch.setattr("audio_extract.separate.Separator", FakeSeparator)
    vocal_ids = []
    for model_name in factors:
        record = render_candidate(
            layout, source, model_filename=model_name, target="vocals",
            construction="native_primary", overlap=8, code_commit="test",
            model_dir=tmp_path / "models", input_recipe_id=mapped["recipe_id"],
        )
        assert record["parents"] == [mapped["recipe_id"]]
        vocal_ids.append(record["recipe_id"])
        recipe = json.loads((layout.candidate_dir(record["recipe_id"]) / "recipe.json").read_text())
        assert recipe["input_pcm"]["parent_recipe_ids"] == [mapped["recipe_id"]]

    residual = render_residual_candidate(
        layout, source, vocal_recipe_id=vocal_ids[0], code_commit="test",
        mixture_recipe_id=mapped["recipe_id"],
    )
    assert residual["parents"] == [mapped["recipe_id"], vocal_ids[0]]
    residual_recipe = json.loads(
        (layout.candidate_dir(residual["recipe_id"]) / "recipe.json").read_text()
    )
    assert residual_recipe["input_pcm"]["parent_recipe_ids"] == [mapped["recipe_id"]]
    residual_audio, _ = sf.read(
        layout.candidate_dir(residual["recipe_id"]) / "output.f32.wav",
        dtype="float32", always_2d=True,
    )
    assert np.allclose(residual_audio, stereo * 0.9, atol=2e-7)

    ensemble = render_ensemble_candidate(
        layout, source, member_recipe_ids=vocal_ids, algo="median",
        code_commit="test", mixture_recipe_id=mapped["recipe_id"],
    )
    assert ensemble["parents"][0] == mapped["recipe_id"]
    output, _ = sf.read(
        layout.candidate_dir(ensemble["recipe_id"]) / "output.f32.wav",
        dtype="float32", always_2d=True,
    )
    assert output.shape == (frames, 2)
    assert np.allclose(output, stereo * 0.8, atol=2e-7)
