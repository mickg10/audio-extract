import json
from pathlib import Path

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
    by_op = {n["operation"]: n for n in report["delivery_nodes"]}
    assert {"global_gain", "dither_quantize", "encode_delivery"} <= set(by_op)

    # each child lives under its OWN node-id dir (per-node paths in the report)
    gain = by_op["global_gain"]
    assert Path(gain["path"]).exists() and sf.info(gain["path"]).subtype == "FLOAT"
    pcm24 = next(n for n in report["delivery_nodes"]
                 if n["operation"] == "dither_quantize" and n["sample_format"] == "pcm_s24")
    assert Path(pcm24["path"]).exists() and sf.info(pcm24["path"]).subtype == "PCM_24"
    assert (layout.render_dir(recipe_id) / "repro.json").exists()

    # lineage: gain off the candidate; dithers off the gain master
    assert gain["parents"] == [recipe_id]
    assert all(n["parents"] == [gain["recipe_id"]]
               for n in report["delivery_nodes"] if n["operation"] == "dither_quantize")
    assert report["finalist_candidate"] == recipe_id


def test_deliver_no_overwrite_on_reparam(tmp_path):
    # Different target level -> different node ids -> different dirs; nothing overwritten.
    layout, source_record, recipe_id = _make_run(tmp_path)
    r1 = dv.deliver(layout, source_record, recipe_id, target_dbfs=-5.0, code_commit="t")
    r2 = dv.deliver(layout, source_record, recipe_id, target_dbfs=-3.0, code_commit="t")
    g1 = next(n for n in r1["delivery_nodes"] if n["operation"] == "global_gain")
    g2 = next(n for n in r2["delivery_nodes"] if n["operation"] == "global_gain")
    assert g1["recipe_id"] != g2["recipe_id"] and g1["path"] != g2["path"]
    assert Path(g1["path"]).exists() and Path(g2["path"]).exists()
