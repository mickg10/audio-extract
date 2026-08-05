import json

import numpy as np
import soundfile as sf

from audio_extract import delivery as dv
from audio_extract.storage import TrackLayout

SR = 44100


def test_global_gain_peak_normalizes():
    x = (np.random.default_rng(0).standard_normal((2000, 2)) * 0.2).astype(np.float32)
    gained, applied_db = dv.global_gain(x, target_dbfs=-5.0)
    peak_dbfs = 20 * np.log10(np.max(np.abs(gained)))
    assert abs(peak_dbfs - (-5.0)) < 1e-3


def test_tpdf_dither_lands_on_integer_grid(tmp_path):
    x = np.linspace(-0.5, 0.5, 4000).reshape(-1, 1).astype(np.float32)
    y = dv.tpdf_dither_float(x, 16, seed=0)
    p = tmp_path / "q.wav"
    sf.write(str(p), y, SR, subtype="PCM_16")
    back, _ = sf.read(str(p), dtype="float64", always_2d=True)
    grid = back * 32768.0
    assert np.allclose(grid, np.round(grid), atol=1e-3)   # on the 16-bit grid
    assert np.max(np.abs(back)) <= 1.0


def test_tpdf_dither_deterministic():
    x = np.linspace(-0.3, 0.3, 500).reshape(-1, 1)
    assert np.array_equal(dv.tpdf_dither_float(x, 16, seed=7), dv.tpdf_dither_float(x, 16, seed=7))
    assert not np.array_equal(dv.tpdf_dither_float(x, 16, seed=1), dv.tpdf_dither_float(x, 16, seed=2))


def test_delivery_node_id_stable_and_sensitive():
    a = dv.delivery_node_id("global_gain", ["sha256:p"], {"target_micro_dbfs": -5000, "mode": "peak"}, "abc")
    b = dv.delivery_node_id("global_gain", ["sha256:p"], {"target_micro_dbfs": -5000, "mode": "peak"}, "abc")
    c = dv.delivery_node_id("global_gain", ["sha256:p"], {"target_micro_dbfs": -3000, "mode": "peak"}, "abc")
    assert a == b and a != c and a.startswith("sha256:")


def _make_run(tmp_path):
    layout = TrackLayout(tmp_path, "trk").ensure()
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal((SR, 2)) * 0.1).astype(np.float32)
    recipe_id = "sha256:cand0000"
    cdir = layout.candidate_dir(recipe_id)
    cdir.mkdir(parents=True, exist_ok=True)
    sf.write(str(cdir / "output.f32.wav"), audio, SR, subtype="FLOAT")
    source_record = {"track_id": "trk", "source_blob_sha256": "sha256:src",
                     "input_pcm_sha256": "sha256:in", "sample_rate_hz": SR,
                     "channel_layout": ["FL", "FR"], "frames": SR}
    (layout.source_dir / "source.json").write_text(json.dumps(source_record))
    return layout, source_record, recipe_id


def test_deliver_produces_child_dag(tmp_path):
    layout, source_record, recipe_id = _make_run(tmp_path)
    report = dv.deliver(layout, source_record, recipe_id, target_dbfs=-5.0, code_commit="test")
    rdir = layout.render_dir(recipe_id)
    assert (rdir / "master_gain.f32.wav").exists()
    assert (rdir / "delivery_pcm24.wav").exists()
    assert (rdir / "delivery_pcm16.wav").exists()
    assert (rdir / "repro.json").exists()

    # float master is FLOAT subtype; pcm24 is PCM_24
    assert sf.info(str(rdir / "master_gain.f32.wav")).subtype == "FLOAT"
    assert sf.info(str(rdir / "delivery_pcm24.wav")).subtype == "PCM_24"

    ops = {n["operation"] for n in report["delivery_nodes"]}
    assert {"global_gain", "dither_quantize"} <= ops
    # every delivery node parents back toward the finalist candidate
    gain = next(n for n in report["delivery_nodes"] if n["operation"] == "global_gain")
    assert gain["parents"] == [recipe_id]
    dith = [n for n in report["delivery_nodes"] if n["operation"] == "dither_quantize"]
    assert all(n["parents"] == [gain["recipe_id"]] for n in dith)   # dither off the gain master, not the raw candidate
    assert report["finalist_candidate"] == recipe_id
