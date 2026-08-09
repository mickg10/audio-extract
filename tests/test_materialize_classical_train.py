import json
import hashlib

import numpy as np
import pytest
import soundfile as sf

from audio_extract.materialize_classical_train import (
    GridMismatch,
    materialize_cantolopera_pair,
    materialize_cantoria,
    materialize_exact_truth,
)
from audio_extract.cantolopera_training import SELECTION_SCHEMA, TASK


def _write(path, audio, sr=44100):
    sf.write(path, np.asarray(audio, dtype="float32"), sr, subtype="FLOAT")


def _sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert recipe["task"] == "soloist_vs_rest"
    assert recipe["task_roles"] == {
        "removed": ["featured_soloists"],
        "retained": ["orchestra", "chorus", "non_target_soloists"],
    }
    assert recipe["operations"] == [{"operation": "role_map", "mapping": {
        "mix_with_voice": "M", "orchestra_only": "A", "voice_ref": "V"
    }}]
    assert set(recipe["demucs_full_track_affine"]) == {"M", "A", "V"}
    for role, affine in recipe["demucs_full_track_affine"].items():
        assert affine["implementation"] == "demucs-full-track-affine/v1"
        assert np.isfinite(float(affine["mean"]))
        assert np.isfinite(float(affine["scale"])) and float(affine["scale"]) > 0


def test_cantolopera_pair_is_explicit_aligned_float_child(tmp_path):
    source = tmp_path / "staged-audio"
    output = tmp_path / "materialized"
    source.mkdir()
    rng = np.random.default_rng(9)
    accompaniment = rng.normal(0, 0.02, (4800, 2)).astype("float32")
    vocal = rng.normal(0, 0.005, (4800, 2)).astype("float32")
    mixture = accompaniment + vocal
    mixture_path = source / "opera_voice.wav"
    accompaniment_path = source / "opera_orchestra.wav"
    _write(mixture_path, mixture, sr=48000)
    _write(accompaniment_path, accompaniment, sr=48000)
    source_hashes = {_sha(mixture_path), _sha(accompaniment_path)}
    selected = {
        "schema": SELECTION_SCHEMA,
        "pair_id": "opera",
        "task": TASK,
        "group_id": "cantolopera:component:test",
        "split": "train",
        "audit_sha256": "sha256:" + "a" * 64,
        "selector_code_commit": "1" * 40,
        "files": {
            "voice": {
                "path": "/different-host/opera_voice.wav",
                "container_sha256": _sha(mixture_path),
            },
            "orchestra": {
                "path": "/different-host/opera_orchestra.wav",
                "container_sha256": _sha(accompaniment_path),
            },
        },
        "assessment": {
            "eligible": True,
            "cohort": "tier_a",
            "transform": {
                "schema": "audio-extract/cantolopera-fixed-alignment/v1",
                "mixture_to_orchestra": {
                    "delay_samples": 0,
                    "fractional_samples": "0",
                    "polarity": 1,
                    "channel_swap": False,
                },
                "orchestra_gain": "1",
                "source_rate_hz": 48000,
                "output_rate_hz": 44100,
                "resampler": "soxr_vhq",
                "derive_vocal": "V=M-A",
            },
        },
    }
    result = materialize_cantolopera_pair(
        selected,
        source,
        output,
        selection_sha256="sha256:" + "b" * 64,
    )
    assert result["status"] == "materialized"
    assert result["M_eq_A_plus_V_db"] < -130
    arrays = {}
    for role in "MAV":
        path = output / "opera" / f"{role}.f32.wav"
        info = sf.info(path)
        arrays[role], rate = sf.read(path, dtype="float32", always_2d=True)
        assert (info.frames, rate, info.channels, info.subtype) == (4410, 44100, 2, "FLOAT")
    assert np.max(np.abs(arrays["M"] - arrays["A"] - arrays["V"])) < 1e-7
    assert {_sha(mixture_path), _sha(accompaniment_path)} == source_hashes
    recipe = json.loads((output / "opera" / "recipe.json").read_text())
    assert recipe["task"] == TASK
    assert recipe["integrity_class"] == "same_take_paired_target"
    assert [node["operation"] for node in recipe["operations"]] == [
        "align", "fixed_gain", "resample", "derive_source"
    ]
    assert recipe["operations"][2]["backend"] == "python-soxr"
    again = materialize_cantolopera_pair(
        selected,
        source,
        output,
        selection_sha256="sha256:" + "b" * 64,
    )
    assert again["status"] == "verified_existing"


def test_cantolopera_materializer_rejects_non_tier_a(tmp_path):
    selected = {
        "schema": SELECTION_SCHEMA,
        "pair_id": "bad",
        "task": TASK,
        "assessment": {"eligible": False, "cohort": "tier_b"},
    }
    with pytest.raises(ValueError, match="non-Tier-A"):
        materialize_cantolopera_pair(
            selected, tmp_path, tmp_path / "out", selection_sha256="sha256:" + "c" * 64
        )
