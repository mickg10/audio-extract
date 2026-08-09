import json

import numpy as np
import soundfile as sf

from audio_extract import identity, recipe as recipe_mod
from audio_extract.storage import TrackLayout
from tools.oracle_routing_envelope import _write_removed_vocal


def test_removed_vocal_is_exact_immutable_child_of_route(tmp_path):
    frames = 4096
    t = np.arange(frames, dtype=np.float64) / 44100.0
    accompaniment = np.column_stack([
        0.2 * np.sin(2 * np.pi * 220 * t),
        0.2 * np.sin(2 * np.pi * 330 * t),
    ]).astype("float32")
    vocal = np.column_stack([
        0.1 * np.sin(2 * np.pi * 660 * t),
        0.08 * np.sin(2 * np.pi * 550 * t),
    ]).astype("float32")
    mixture = accompaniment + vocal
    source = {
        "input_pcm_sha256": identity.artifact_pcm_sha256(
            mixture, 44100, ["FL", "FR"], frames
        ),
        "sample_rate_hz": 44100,
        "channel_layout": ["FL", "FR"],
        "frames": frames,
    }
    layout = TrackLayout(tmp_path / "lib", "opera").ensure()
    parent_recipe = {
        "schema": recipe_mod.SCHEMA,
        "canon": recipe_mod.CANON,
        "input_pcm": {
            "sha256": source["input_pcm_sha256"],
            "sample_rate_hz": 44100,
            "channel_layout": ["FL", "FR"],
            "frames": frames,
            "sample_format": "float32-le-interleaved",
        },
        "operation": {"type": "phrase_route", "target": "instrumental"},
        "model": {
            "model_id": "test-route",
            "weights_sha256": "0" * 64,
            "adapter": "test",
            "adapter_revision": "test/v1",
        },
        "effective_config": {"alignment": "source-grid-exact"},
        "software": {"audio_extract_commit": "test"},
    }
    parent_id = identity.recipe_id(parent_recipe)
    temporary = tmp_path / "parent.f32.wav"
    sf.write(temporary, accompaniment, 44100, subtype="FLOAT")
    parent_pcm = identity.artifact_pcm_sha256(
        accompaniment, 44100, ["FL", "FR"], frames
    )
    parent_dir = layout.write_candidate(
        parent_id, parent_recipe, {"parents": []}, temporary, parent_pcm
    )
    parent_path = parent_dir / "output.f32.wav"
    parent = {
        "recipe_id": parent_id,
        "path": str(parent_path),
        "artifact_pcm_sha256": parent_pcm,
        "container_sha256": "sha256:" + __import__("hashlib").sha256(
            parent_path.read_bytes()
        ).hexdigest(),
    }

    first = _write_removed_vocal(
        layout=layout, source=source, mixture=mixture,
        accompaniment=accompaniment, parent=parent, code_commit="test",
    )
    child = layout.candidate_dir(first["recipe_id"])
    removed, sr = sf.read(
        child / "output.f32.wav", dtype="float32", always_2d=True
    )
    assert sr == 44100
    assert np.array_equal(removed, mixture.astype("float32") - accompaniment)
    recipe = json.loads((child / "recipe.json").read_text())
    assert identity.recipe_id(recipe) == first["recipe_id"]
    assert recipe["model"]["members"] == [parent_id]
    assert (child / "COMPLETE").is_file()

    second = _write_removed_vocal(
        layout=layout, source=source, mixture=mixture,
        accompaniment=accompaniment, parent=parent, code_commit="test",
    )
    assert second["cached"] is True
    assert second["artifact_pcm_sha256"] == first["artifact_pcm_sha256"]
