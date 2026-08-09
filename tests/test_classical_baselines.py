import json

import numpy as np
import soundfile as sf

from audio_extract import identity
from audio_extract.classical_baselines import _ensure_exact_source, _verified_row
from audio_extract.separate import render_residual_candidate
from audio_extract.storage import TrackLayout


def test_baseline_row_reopens_hashes_and_preserves_parent(tmp_path):
    frames = 8192
    t = np.arange(frames, dtype=np.float32) / 44_100
    mix = np.column_stack((np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)))
    truth = tmp_path / "truth.wav"
    sf.write(truth, mix, 44_100, subtype="FLOAT")
    layout = TrackLayout(tmp_path / "lib", "bologna_verdi")
    source = _ensure_exact_source(layout, truth)
    assert _ensure_exact_source(layout, truth)["input_pcm_sha256"] == source["input_pcm_sha256"]

    vocal = (mix * 0.25).astype("float32")
    parent_id = "sha256:" + "a" * 64
    parent_recipe = {
        "operation": {"type": "separate", "construction": "native_primary",
                      "target": "vocals"},
        "model": {"executed_bundle_id": "bundle-one"},
    }
    parent_tmp = tmp_path / "vocal.wav"
    sf.write(parent_tmp, vocal, 44_100, subtype="FLOAT")
    layout.write_candidate(
        parent_id, parent_recipe, {}, parent_tmp,
        identity.artifact_pcm_sha256(vocal, 44_100, ["FL", "FR"], frames),
    )
    record = render_residual_candidate(
        layout, source, vocal_recipe_id=parent_id, code_commit="abc"
    )
    row = _verified_row(
        layout, "bologna_verdi", "residual_mdx23c", record,
        {"mdx": {"bundle_sha256": "bundle-one"}}, "abc",
    )
    assert row["subtype"] == "FLOAT"
    assert row["frames"] == frames
    assert row["parent_candidate_recipe_ids"] == [parent_id]
    assert row["executed_model_bundle_hashes"] == ["bundle-one"]
    assert row["artifact_pcm_sha256"] == (
        layout.candidate_dir(record["recipe_id"]) / "output.pcm.sha256"
    ).read_text().strip()
    assert json.loads(
        (layout.candidate_dir(record["recipe_id"]) / "recipe.json").read_text()
    )["model"]["members"] == [parent_id]
