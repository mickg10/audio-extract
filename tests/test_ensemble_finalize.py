"""Ensemble construction (oracle §5) + finalist revalidation→delivery (v3 §11)."""
import json

import numpy as np
import pytest
import soundfile as sf

from audio_extract import fixtures as fx
from audio_extract import identity
from audio_extract.cli_autonomous import finalize_and_deliver, revalidate_finalist
from audio_extract.separate import (
    _stft_geometric_median,
    render_ensemble_candidate,
    render_residual_candidate,
)
from audio_extract.storage import TrackLayout

SR = 44100


def _run_with_vocal_members(tmp_path):
    """A run whose store holds 3 native-vocal candidates: two good (one delayed),
    one corrupted — the median should reject the corruption."""
    dur = 4.0
    vocal, _ = fx.synth_vocal(SR, dur)
    orch = fx.synth_orchestra(SR, dur)
    n = min(len(vocal), len(orch))
    vocal, orch = vocal[:n], orch[:n]
    mix = vocal + orch

    layout = TrackLayout(tmp_path, "trk").ensure()
    sf.write(str(layout.source_dir / "canonical.f32.wav"), mix.astype("float32"), SR, subtype="FLOAT")
    (layout.source_dir / "source.json").write_text(json.dumps({
        "track_id": "trk", "sample_rate_hz": SR, "channel_layout": ["FL", "FR"],
        "frames": n, "input_pcm_sha256": "sha256:in",
        "source_blob_sha256": "sha256:src"}))

    members = {}
    variants = {
        "v_good": vocal,
        "v_delayed": np.vstack([np.zeros((25, 2)), vocal])[:n],          # 25-sample delay
        "v_corrupt": vocal + 0.5 * fx.synth_orchestra(SR, dur)[:n],      # steals orchestra
    }
    for name, arr in variants.items():
        rid = f"sha256:{name}"
        recipe = {"operation": {"type": "separate", "construction": "native_primary",
                                "target": "vocals"}, "model": {"model_id": name}}
        pcm = identity.artifact_pcm_sha256(arr.astype("float32"), SR, ["FL", "FR"], n)
        tmp = tmp_path / f"{name}.wav"
        sf.write(str(tmp), arr.astype("float32"), SR, subtype="FLOAT")
        layout.write_candidate(rid, recipe, {"e": 1}, tmp, pcm)
        members[name] = rid
    return layout, mix, vocal, orch, members


def test_median_ensemble_rejects_corruption_and_aligns(tmp_path, monkeypatch):
    layout, mix, vocal, orch, members = _run_with_vocal_members(tmp_path)
    src = json.loads((layout.source_dir / "source.json").read_text())
    rec = render_ensemble_candidate(layout, src,
                                    member_recipe_ids=list(members.values()),
                                    algo="median", code_commit="t")
    out, _ = sf.read(str(layout.candidate_dir(rec["recipe_id"]) / "output.f32.wav"),
                     dtype="float64", always_2d=True)
    n = min(len(out), len(orch))
    # median of {good, aligned-delayed, corrupt} ≈ good vocal -> residual ≈ orchestra
    err_ens = np.sqrt(np.mean((out[:n] - orch[:n]) ** 2))
    # compare against the corrupt member's residual (mix - corrupt vocal)
    corrupt, _ = sf.read(str(layout.candidate_dir(members["v_corrupt"]) / "output.f32.wav"),
                         dtype="float64", always_2d=True)
    err_corrupt = np.sqrt(np.mean(((mix[:n] - corrupt[:n]) - orch[:n]) ** 2))
    assert err_ens < 0.5 * err_corrupt
    assert rec["parents"] == sorted(members.values())
    delays = {a["recipe_id"]: a["delay"] for a in rec["alignments"]}
    assert abs(delays[members["v_delayed"]]) >= 20      # the delay was detected
    from audio_extract import alignment
    monkeypatch.setattr(
        alignment, "estimate_alignment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache realigned")),
    )
    cached = render_ensemble_candidate(
        layout, src, member_recipe_ids=list(reversed(list(members.values()))),
        algo="median", code_commit="t",
    )
    assert cached["recipe_id"] == rec["recipe_id"] and cached["cached"] is True


def test_ensemble_refuses_non_vocal_members(tmp_path):
    layout, mix, vocal, orch, members = _run_with_vocal_members(tmp_path)
    src = json.loads((layout.source_dir / "source.json").read_text())
    # forge an instrumental-role member
    rid = "sha256:not_vocal"
    tmp = tmp_path / "nv.wav"
    sf.write(str(tmp), orch.astype("float32"), SR, subtype="FLOAT")
    layout.write_candidate(rid, {"operation": {"type": "separate",
                                               "construction": "mixture_minus_primary",
                                               "target": "vocals"}}, {},
                           tmp, identity.artifact_pcm_sha256(orch.astype("float32"), SR,
                                                             ["FL", "FR"], len(orch)))
    with pytest.raises(ValueError, match="not a native vocal"):
        render_ensemble_candidate(layout, src,
                                  member_recipe_ids=[members["v_good"], members["v_delayed"], rid],
                                  algo="median", code_commit="t")


def test_single_vocal_residual_is_exact_parented_and_cached(tmp_path):
    layout, mix, vocal, orch, members = _run_with_vocal_members(tmp_path)
    src = json.loads((layout.source_dir / "source.json").read_text())
    first = render_residual_candidate(
        layout, src, vocal_recipe_id=members["v_good"], code_commit="t"
    )
    second = render_residual_candidate(
        layout, src, vocal_recipe_id=members["v_good"], code_commit="t"
    )
    assert first["recipe_id"] == second["recipe_id"]
    assert first["parents"] == [members["v_good"]]
    assert first["cached"] is False and second["cached"] is True
    cdir = layout.candidate_dir(first["recipe_id"])
    info = sf.info(cdir / "output.f32.wav")
    assert (info.frames, info.samplerate, info.channels, info.subtype) == (
        len(mix), SR, 2, "FLOAT"
    )
    recipe = json.loads((cdir / "recipe.json").read_text())
    assert recipe["model"]["members"] == [members["v_good"]]
    reopened, _ = sf.read(cdir / "output.f32.wav", dtype="float32", always_2d=True)
    assert identity.artifact_pcm_sha256(reopened, SR, ["FL", "FR"], len(reopened)) == (
        cdir / "output.pcm.sha256"
    ).read_text().strip()


def test_mean_canonicalizes_member_weight_pairs(tmp_path):
    layout, _mix, _vocal, _orch, members = _run_with_vocal_members(tmp_path)
    src = json.loads((layout.source_dir / "source.json").read_text())
    ids = [members["v_good"], members["v_delayed"]]
    first = render_ensemble_candidate(
        layout, src, member_recipe_ids=ids, weights=[0.25, 0.75],
        algo="mean", code_commit="t",
    )
    second = render_ensemble_candidate(
        layout, src, member_recipe_ids=list(reversed(ids)), weights=[0.75, 0.25],
        algo="mean", code_commit="t",
    )
    assert first["recipe_id"] == second["recipe_id"]
    assert second["cached"] is True


def test_stft_geometric_median_is_lr_ms_equivariant():
    rng = np.random.default_rng(9)
    stack = rng.normal(0, 0.05, size=(3, 4096, 2)).astype("float32")
    stack[2, 1200:1600] += np.array([0.8, -0.3], dtype="float32")
    lr = _stft_geometric_median(stack, n_fft=256, hop=64, max_iter=20, tol=1e-5)
    basis = np.array([[1, 1], [1, -1]], dtype="float64") / np.sqrt(2)
    ms = np.einsum("knc,cd->knd", stack, basis)
    ms_result = _stft_geometric_median(ms, n_fft=256, hop=64, max_iter=20, tol=1e-5)
    roundtrip = np.einsum("nd,dc->nc", ms_result, basis.T)
    assert np.max(np.abs(lr - roundtrip)) < 2e-4


def _passages_for(layout, n):
    layout.passages_dir.mkdir(parents=True, exist_ok=True)
    (layout.passages_dir / "passages.v1.json").write_text(json.dumps({
        "schema": "audio-extract/passages/v1",
        "passages": [{"passage_id": "p_0", "start_sample": 0, "end_sample": n // 2,
                      "tags": ["voice_dominant"], "features": {}, "detectors": {}}]}))


def test_revalidate_and_deliver_clean_finalist(tmp_path):
    layout, mix, vocal, orch, members = _run_with_vocal_members(tmp_path)
    n = mix.shape[0]
    _passages_for(layout, n)
    src = json.loads((layout.source_dir / "source.json").read_text())
    rec = render_ensemble_candidate(layout, src,
                                    member_recipe_ids=list(members.values()),
                                    algo="median", code_commit="t")
    res = finalize_and_deliver(layout, SR, rec["recipe_id"], code_commit="t")
    assert res["status"] == "delivered"
    assert res["revalidation"]["ok"] and res["revalidation"]["passages_checked"] == 1
    ops = {nd["operation"] for nd in res["delivery"]["delivery_nodes"]}
    assert {"global_gain", "dither_quantize", "encode_delivery"} <= ops


def test_revalidation_rejects_corrupted_finalist(tmp_path):
    layout, mix, vocal, orch, members = _run_with_vocal_members(tmp_path)
    n = mix.shape[0]
    _passages_for(layout, n)
    # a "finalist" with NaNs in the decisive passage must not ship
    bad = (mix - vocal).copy()
    bad[100:200] = np.nan
    rid = "sha256:bad_final"
    tmp = tmp_path / "bad.wav"
    sf.write(str(tmp), np.nan_to_num(bad, nan=2.0).astype("float32"), SR, subtype="FLOAT")
    layout.write_candidate(rid, {"operation": {"type": "mixture_minus_source",
                                               "construction": "mixture_minus_primary",
                                               "target": "vocals"}}, {}, tmp,
                           identity.artifact_pcm_sha256(
                               np.nan_to_num(bad, nan=2.0).astype("float32"), SR,
                               ["FL", "FR"], n))
    reval = revalidate_finalist(layout, SR, rid)
    assert reval["ok"] is False and reval["failures"]
    res = finalize_and_deliver(layout, SR, rid, code_commit="t")
    assert res["status"] == "revalidation_failed"
    assert "delivery" not in res                       # nothing shipped
