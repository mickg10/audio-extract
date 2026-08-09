import json
import math

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.resample import render_resample_candidate
from audio_extract.storage import TrackLayout


def _parent(layout, tmp_path, *, frames=10_001, sample_rate=44_100):
    t = np.arange(frames, dtype="float64") / sample_rate
    mono = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    audio = np.column_stack((mono, mono * 0.75)).astype("float32")
    parent_id = "sha256:parent"
    recipe = {
        "input_pcm": {
            "sha256": "sha256:source",
            "sample_rate_hz": sample_rate,
            "channel_layout": ["FL", "FR"],
            "frames": frames,
            "sample_format": "float32-le-interleaved",
        },
        "operation": {
            "type": "mixture_minus_source",
            "target": "instrumental",
            "construction": "waveform_ensemble",
        },
    }
    temporary = tmp_path / "parent.wav"
    sf.write(temporary, audio, sample_rate, subtype="FLOAT")
    pcm = identity.artifact_pcm_sha256(audio, sample_rate, ["FL", "FR"], frames)
    layout.write_candidate(parent_id, recipe, {}, temporary, pcm)
    return parent_id, audio


def test_resample_candidate_is_explicit_48k_float_child_and_cached(tmp_path):
    layout = TrackLayout(tmp_path, "track").ensure()
    parent_id, parent_audio = _parent(layout, tmp_path)
    first = render_resample_candidate(
        layout, parent_id, target_rate_hz=48_000,
        algo="scipy_polyphase", code_commit="test",
    )
    second = render_resample_candidate(
        layout, parent_id, target_rate_hz=48_000,
        algo="scipy_polyphase", code_commit="test",
    )
    assert second["recipe_id"] == first["recipe_id"]
    assert first["cached"] is False and second["cached"] is True
    assert first["parents"] == [parent_id]

    candidate_dir = layout.candidate_dir(first["recipe_id"])
    info = sf.info(candidate_dir / "output.f32.wav")
    expected_frames = math.ceil(len(parent_audio) * 160 / 147)
    assert (info.frames, info.samplerate, info.channels, info.subtype) == (
        expected_frames, 48_000, 2, "FLOAT"
    )
    recipe = json.loads((candidate_dir / "recipe.json").read_text())
    assert recipe["operation"] == {"type": "resample", "target": "48000_hz"}
    assert recipe["input_pcm"]["parent_recipe_ids"] == [parent_id]
    assert recipe["effective_config"]["resampler"] == "scipy_polyphase"
    assert recipe["effective_config"]["frame_count_policy"] == "ceil(input_frames*up/down)"

    reopened, rate = sf.read(candidate_dir / "output.f32.wav", dtype="float32",
                             always_2d=True)
    actual = identity.artifact_pcm_sha256(
        reopened, rate, ["FL", "FR"], len(reopened)
    )
    assert actual == (candidate_dir / "output.pcm.sha256").read_text().strip()
    original, original_rate = sf.read(
        layout.candidate_dir(parent_id) / "output.f32.wav", dtype="float32",
        always_2d=True,
    )
    assert original_rate == 44_100 and np.array_equal(original, parent_audio)


def test_resample_candidate_refuses_redundant_rate(tmp_path):
    layout = TrackLayout(tmp_path, "track").ensure()
    parent_id, _ = _parent(layout, tmp_path)
    with pytest.raises(ValueError, match="identical"):
        render_resample_candidate(layout, parent_id, target_rate_hz=44_100)
