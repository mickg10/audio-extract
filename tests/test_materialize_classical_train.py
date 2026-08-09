import json

import numpy as np
import pytest
import soundfile as sf

from audio_extract.materialize_classical_train import (
    GridMismatch,
    materialize_cantoria,
    materialize_exact_truth,
)


def _write(path, audio, sr=44100):
    sf.write(path, np.asarray(audio, dtype="float32"), sr, subtype="FLOAT")


def test_cantoria_materialization_is_float32_stereo_and_exact(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    source.mkdir()
    rng = np.random.default_rng(2)
    vocal = rng.normal(0, 0.01, (4096, 1)).astype("float32")
    accompaniment = rng.normal(0, 0.02, (4096, 1)).astype("float32")
    _write(source / "Cantoria_X_MixOrgan.wav", accompaniment + vocal)
    _write(source / "Cantoria_X_Mix.wav", vocal)

    result = materialize_cantoria(source, output, "X")
    assert result["status"] == "materialized"
    arrays = []
    for role in "MAV":
        info = sf.info(output / "cantoria_X" / f"{role}.f32.wav")
        audio, sr = sf.read(output / "cantoria_X" / f"{role}.f32.wav",
                            dtype="float32", always_2d=True)
        assert info.subtype == "FLOAT"
        assert sr == 44100
        assert audio.shape == (4096, 2)
        arrays.append(audio)
    assert np.max(np.abs(arrays[0] - arrays[1] - arrays[2])) < 1e-7
    recipe = json.loads((output / "cantoria_X" / "recipe.json").read_text())
    assert recipe["sample_format"] == "float32-le-interleaved"
    assert any(n.get("expression") == "M-V" for n in recipe["operations"])


def test_mismatched_frames_are_refused_not_truncated(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "Cantoria_X_MixOrgan.wav", np.zeros((100, 1), dtype="float32"))
    _write(source / "Cantoria_X_Mix.wav", np.zeros((99, 1), dtype="float32"))
    with pytest.raises(GridMismatch, match="source grid mismatch"):
        materialize_cantoria(source, tmp_path / "out", "X")


def test_existing_immutable_materialization_is_verified(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    source.mkdir()
    _write(source / "Cantoria_X_MixOrgan.wav", np.ones((128, 1), dtype="float32") * 0.2)
    _write(source / "Cantoria_X_Mix.wav", np.ones((128, 1), dtype="float32") * 0.1)
    materialize_cantoria(source, output, "X")
    again = materialize_cantoria(source, output, "X")
    assert again["status"] == "verified_existing"


def test_exact_truth_materialization_preserves_roles_without_transform(tmp_path):
    truth = tmp_path / "truth" / "opera"
    output = tmp_path / "out"
    truth.mkdir(parents=True)
    rng = np.random.default_rng(4)
    accompaniment = rng.normal(0, 0.01, (256, 2)).astype("float32")
    vocal = rng.normal(0, 0.01, (256, 2)).astype("float32")
    _write(truth / "mix_with_voice.wav", accompaniment + vocal)
    _write(truth / "orchestra_only.wav", accompaniment)
    _write(truth / "voice_ref.wav", vocal)
    result = materialize_exact_truth(tmp_path / "truth", output, "opera")
    assert result["status"] == "materialized"
    recipe = json.loads((output / "opera" / "recipe.json").read_text())
    assert recipe["task"] == "featured_soloist_vs_rest"
    assert recipe["operations"] == [{"operation": "role_map", "mapping": {
        "mix_with_voice": "M", "orchestra_only": "A", "voice_ref": "V"
    }}]
