"""End-to-end WP12 test: build challenges from a synthetic run, run two synthetic
'models' over them, and let the selector produce a real terminal decision —
no separators, no GPU, fully deterministic."""
import json

import numpy as np
import soundfile as sf

from audio_extract import cli_autonomous as auto
from audio_extract import fixtures as fx
from audio_extract.storage import TrackLayout

SR = 44100


def _make_run(tmp_path):
    """Synthetic run: 8s orchestra; vocal only in the second half -> the first
    ~3.6s is a GENUINE no-vocal control; provisional vocal saved."""
    dur = 8.0
    orch = fx.synth_orchestra(SR, dur)
    vocal, _ = fx.synth_vocal(SR, dur, phrases=[(0.60 * dur, 0.90 * dur)])
    n = min(len(orch), len(vocal))
    mix = orch[:n] + vocal[:n]

    layout = TrackLayout(tmp_path, "trk").ensure()
    sf.write(str(layout.source_dir / "canonical.f32.wav"), mix.astype("float32"), SR, subtype="FLOAT")
    sf.write(str(layout.source_dir / "provisional_vocal.f32.wav"),
             vocal[:n].astype("float32"), SR, subtype="FLOAT")
    (layout.source_dir / "source.json").write_text(json.dumps(
        {"track_id": "trk", "sample_rate_hz": SR, "channel_layout": ["FL", "FR"],
         "frames": n, "input_pcm_sha256": "sha256:x"}))
    control_end = int(0.55 * dur * SR)
    (layout.passages_dir).mkdir(parents=True, exist_ok=True)
    (layout.passages_dir / "passages.v1.json").write_text(json.dumps({
        "schema": "audio-extract/passages/v1",
        "passages": [
            {"passage_id": "p_ctl", "start_sample": 0, "end_sample": control_end,
             "tags": ["no_vocal_control"],
             "features": {"vocal_energy_ratio": 0.01}, "detectors": {}},
            {"passage_id": "p_bad", "start_sample": 0, "end_sample": n,
             "tags": ["no_vocal_control"],
             "features": {"vocal_energy_ratio": 0.55}, "detectors": {}},  # gated out
        ]}))
    return layout


def _make_factory(layout):
    """Two synthetic 'models': 'oracle' recognizes built mixtures and subtracts the
    EXACT injected vocal (perfect separator; extracts nothing from controls);
    'bad' claims everything as vocals (residual = silence, total theft)."""
    def oracle(audio):
        a = np.asarray(audio, dtype=np.float64)
        for cdir in (layout.root / "challenges").glob("sha256_*"):
            m, _ = sf.read(str(cdir / "mixture.f32.wav"), dtype="float64", always_2d=True)
            if m.shape == a.shape and np.allclose(m, a, atol=1e-4):
                t, _ = sf.read(str(cdir / "target.f32.wav"), dtype="float64", always_2d=True)
                return {"vocals": a - t}
        return {"vocals": np.zeros_like(a)}    # controls: extract nothing

    def factory(name):
        return oracle if name == "oracle" else (lambda audio: {"vocals": np.asarray(audio).copy()})

    return factory


def test_full_autonomous_flow(tmp_path):
    layout = _make_run(tmp_path)

    # build: only the GENUINE control is used
    built = auto.build_challenges(layout, SR, count=4)
    assert built["built"] >= 2
    assert built["controls"] == ["p_ctl"]          # the contaminated one was gated out

    # run: two synthetic models over the cases + theft assay
    res = auto.run_challenges(layout, SR, ["oracle", "bad"], _make_factory(layout))
    assert res["ran"] == built["built"]
    assert res["matrix"]["oracle"]["theft_mean"] < 0.01   # theft assay separates them
    assert res["matrix"]["bad"]["theft_mean"] > 0.9

    # oracle's residual == exact target -> huge SI-SDR; bad's residual == silence
    g_case = next(iter(res["matrix"]["oracle"]["cases"].values()))
    b_case = next(iter(res["matrix"]["bad"]["cases"].values()))
    assert g_case["si_sdr_db"] > 30 > b_case["si_sdr_db"]

    # select: the robust selector certifies 'oracle'; 'bad' fails the theft gate
    decision = auto.select_autonomous(layout, n_boot=50)
    assert decision["status"] == "final"
    assert decision["candidate_id"] == "oracle"
    assert decision["candidates"]["bad"]["gates_passed"] is False
    assert decision["decision_id"].startswith("sd_")

    # run report: consolidated §17.3 shape
    rep = auto.run_report(layout)
    assert rep["status"] == "final"
    assert rep["candidate"]["id"] == "oracle"
    assert rep["challenge_summary"]["cases"] >= 2


def test_recipe_spec_bakeoff(tmp_path):
    """oracle §2: distinct recipes (residual vs native vs ensemble) are evaluated as
    distinct artifacts, keyed by recipe id, each scored as it would be delivered."""
    layout = _make_run(tmp_path)
    built = auto.build_challenges(layout, SR, count=4)
    assert built["built"] >= 2
    fac = _make_factory(layout)

    def factory(name):
        # 'oracle' knows the exact injected vocal; 'native_perfect' returns a perfect
        # instrumental stem; 'bad' claims everything as vocals.
        if name == "native_perfect":
            def est(mix):
                m = np.asarray(mix, dtype=np.float64)
                for cdir in (layout.root / "challenges").glob("sha256_*"):
                    import soundfile as _sf
                    mm, _ = _sf.read(str(cdir / "mixture.f32.wav"), dtype="float64", always_2d=True)
                    if mm.shape == m.shape and np.allclose(mm, m, atol=1e-4):
                        tt, _ = _sf.read(str(cdir / "target.f32.wav"), dtype="float64", always_2d=True)
                        return {"instrumental": tt}
                return {"instrumental": m}
            return est
        return fac(name)

    specs = [
        {"kind": "residual", "models": ["oracle"]},
        {"kind": "native", "models": ["native_perfect"]},
        {"kind": "ensemble_residual", "models": ["oracle", "bad"], "algo": "median"},
    ]
    res = auto.run_challenges(layout, SR, recipe_specs=specs, estimator_factory=factory)
    assert res["ran"] == built["built"]
    keys = set(res["matrix"])
    assert len(keys) == 3                            # three distinct recipe ids
    ids = [auto.recipe_spec_id(s) for s in specs]
    assert set(ids) == keys                          # keyed by recipe id, deterministic
    # residual-oracle and native-perfect both recover the target well
    for rid in ids[:2]:
        c = next(iter(res["matrix"][rid]["cases"].values()))
        assert c["si_sdr_db"] > 25


def _voc_orch():
    v, _ = fx.synth_vocal(SR, 2.0); o = fx.synth_orchestra(SR, 2.0)
    n = min(len(v), len(o)); return v[:n], o[:n]


def test_geometric_median_rejects_outlier_member():
    # 3 members: two agree, one is a wild outlier. Geometric median follows the two;
    # the mean gets dragged toward the outlier. (gpt56's robustness rationale.)
    good, orch = _voc_orch()
    n = len(good)
    members = {"a": good, "b": good * 0.98,
               "outlier": good + 2.0 * orch[:n]}                # member steals orchestra
    def fac(name):
        return lambda mix: {"vocals": members[name]}
    geo = auto.compose_accompaniment(
        {"kind": "geometric_median", "models": ["a", "b", "outlier"]}, fac)(orch[:n] + good)
    mean = auto.compose_accompaniment(
        {"kind": "convex_fusion", "models": ["a", "b", "outlier"]}, fac)(orch[:n] + good)
    # residual should recover the orchestra; geo-median (rejecting the outlier vocal) is closer
    err_geo = np.sqrt(np.mean((geo[:n] - orch[:n]) ** 2))
    err_mean = np.sqrt(np.mean((mean[:n] - orch[:n]) ** 2))
    assert err_geo < err_mean


def test_convex_fusion_projects_to_simplex():
    good, orch = _voc_orch()
    n = len(good)
    def fac(name):
        return lambda mix: {"vocals": good if name == "a" else good * 0.5}
    # negative / unnormalized weights are clipped + normalized to the simplex
    out = auto.compose_accompaniment(
        {"kind": "convex_fusion", "models": ["a", "b"], "weights": [-1.0, 3.0]}, fac)(orch[:n] + good)
    assert out.shape[0] == n and np.all(np.isfinite(out))


def test_recipe_spec_id_stable_and_distinct():
    a = {"kind": "residual", "models": ["mdx23c"], "overlap": 8}
    b = {"kind": "native", "models": ["mdx23c"], "overlap": 8}
    c = {"kind": "residual", "models": ["mdx23c"], "overlap": 4}
    assert auto.recipe_spec_id(a) == auto.recipe_spec_id(dict(a))
    assert len({auto.recipe_spec_id(a), auto.recipe_spec_id(b), auto.recipe_spec_id(c)}) == 3


def test_weighted_mean_id_binds_weight_to_member(tmp_path=None):
    # oracle §2 identity bug: [MDX,Mel]+[.8,.2] and [Mel,MDX]+[.8,.2] are DIFFERENT
    # weighted means and must get different ids (member order carries the weight).
    s1 = {"kind": "ensemble_residual", "algo": "mean", "models": ["MDX", "Mel"], "weights": [0.8, 0.2]}
    s2 = {"kind": "ensemble_residual", "algo": "mean", "models": ["Mel", "MDX"], "weights": [0.8, 0.2]}
    assert auto.recipe_spec_id(s1) != auto.recipe_spec_id(s2)
    # but the same semantic mapping (MDX=.8, Mel=.2) written either way collides
    s3 = {"kind": "ensemble_residual", "algo": "mean", "models": ["Mel", "MDX"], "weights": [0.2, 0.8]}
    assert auto.recipe_spec_id(s1) == auto.recipe_spec_id(s3)
    # median is weight-symmetric -> member order irrelevant
    m1 = {"kind": "ensemble_residual", "algo": "median", "models": ["A", "B", "C"]}
    m2 = {"kind": "ensemble_residual", "algo": "median", "models": ["C", "A", "B"]}
    assert auto.recipe_spec_id(m1) == auto.recipe_spec_id(m2)


def test_native_without_instrumental_raises():
    # §2: a 'native' recipe on a vocals-only model must RAISE, not silently residual
    est = auto.compose_accompaniment({"kind": "native", "models": ["vocals_only"]},
                                     lambda n: (lambda mix: {"vocals": np.asarray(mix)}))
    import pytest
    with pytest.raises(auto.UnsupportedConstructionError):
        est(fx.synth_orchestra(SR, 1.0))


def test_build_declines_without_genuine_controls(tmp_path):
    layout = _make_run(tmp_path)
    # poison the passages file: only contaminated controls remain
    pj = layout.passages_dir / "passages.v1.json"
    doc = json.loads(pj.read_text())
    for p in doc["passages"]:
        p["features"]["vocal_energy_ratio"] = 0.9
    pj.write_text(json.dumps(doc))
    built = auto.build_challenges(layout, SR)
    assert built["built"] == 0 and "genuine" in built["reason"]
