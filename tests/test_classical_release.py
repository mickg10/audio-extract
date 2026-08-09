import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract.classical_release import (
    ClassicalReleaseError,
    ScreeningPolicy,
    package_work,
    pareto_front,
    rank_rows,
    screening,
)


def policy():
    return ScreeningPolicy(
        policy_id="test",
        retained_voice_coef_p90_max=0.12,
        event_hole_db_p90_max=18.0,
        event_hole_db_max_max=30.0,
        alpha_error_p90_max=0.75,
        stereo_width_dev_db_max=0.5,
        coherence_dev_max=0.1,
        artifact_ratio_p90_max=None,
    )


def metrics(**overrides):
    value = {
        "retained_voice_coef_p90": 0.05,
        "retained_voice_db_p90": -20.0,
        "event_hole_db_p90": 10.0,
        "event_hole_db_max": 20.0,
        "alpha_error_p90": 0.2,
        "artifact_ratio_p90": 0.1,
        "scale_dependent_sdr_db": 20.0,
        "stereo_width_dev_db/v2": 0.05,
        "interchannel_coherence_dev/v2": 0.01,
        "erb_envelope_dist_db/v2": 1.0,
    }
    value.update(overrides)
    return value


def row(name, values, pol=None):
    pol = pol or policy()
    return {
        "candidate": name,
        "metrics": values,
        "screening": screening(values, pol),
    }


def test_screening_and_ranking_are_feasibility_first():
    good = row("good", metrics(artifact_ratio_p90=0.2, scale_dependent_sdr_db=18.0))
    cleaner = row("cleaner", metrics(artifact_ratio_p90=0.1, scale_dependent_sdr_db=22.0))
    bad = row("bad", metrics(retained_voice_coef_p90=0.3, event_hole_db_p90=4.0))
    ranked = rank_rows([bad, good, cleaner])
    assert [item["candidate"] for item in ranked] == ["cleaner", "good", "bad"]
    assert not bad["screening"]["passed"]
    assert {failure["metric"] for failure in bad["screening"]["failures"]} == {
        "retained_voice_coef_p90"
    }


def test_pareto_front_keeps_tradeoffs_and_removes_dominated_candidate():
    voice_best = row("voice_best", metrics(retained_voice_coef_p90=0.01,
                                            event_hole_db_p90=15.0))
    hole_best = row("hole_best", metrics(retained_voice_coef_p90=0.08,
                                          event_hole_db_p90=5.0))
    dominated = row("dominated", metrics(retained_voice_coef_p90=0.09,
                                          event_hole_db_p90=16.0,
                                          artifact_ratio_p90=0.3))
    assert set(pareto_front([voice_best, hole_best, dominated])) == {
        "voice_best", "hole_best"
    }


def _write(path: Path, audio: np.ndarray, sr=44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype("float32"), sr, subtype="FLOAT")


def test_package_work_emits_primary_alternate_targets_and_removed_vocal(tmp_path):
    frames = 2048
    t = np.arange(frames) / 44100.0
    accompaniment = np.column_stack([
        0.2 * np.sin(2 * np.pi * 220 * t),
        0.2 * np.sin(2 * np.pi * 330 * t),
    ]).astype("float32")
    vocal = np.column_stack([
        0.1 * np.sin(2 * np.pi * 660 * t),
        0.1 * np.sin(2 * np.pi * 660 * t),
    ]).astype("float32")
    mixture = accompaniment + vocal
    primary = accompaniment + 0.02 * vocal
    alternate = accompaniment + 0.05 * vocal

    source = tmp_path / "source"
    primary_path = source / "primary.f32.wav"
    alternate_path = source / "alternate.f32.wav"
    mix_path = source / "mix.f32.wav"
    a_path = source / "a.f32.wav"
    v_path = source / "v.f32.wav"
    for path, audio in (
        (primary_path, primary), (alternate_path, alternate),
        (mix_path, mixture), (a_path, accompaniment), (v_path, vocal),
    ):
        _write(path, audio)

    report = {
        "work_id": "opera",
        "policy_id": "test",
        "status": "exact_benchmark_qualified",
        "primary": "primary",
        "alternate": "alternate",
        "pareto_front": ["primary", "alternate"],
        "candidates": [
            {"candidate": "primary", "path": str(primary_path),
             "metrics": metrics(), "screening": screening(metrics(), policy())},
            {"candidate": "alternate", "path": str(alternate_path),
             "metrics": metrics(retained_voice_coef_p90=0.07),
             "screening": screening(metrics(retained_voice_coef_p90=0.07), policy())},
        ],
        "truth": {
            "mixture": str(mix_path), "accompaniment": str(a_path),
            "vocal": str(v_path), "frames": frames, "sample_rate_hz": 44100,
        },
    }
    output = tmp_path / "release"
    manifest = package_work(report, output)
    work = output / "opera"
    for name in (
        "accompaniment.primary.f32.wav", "accompaniment.alternate.f32.wav",
        "exact-orchestra-target.f32.wav", "exact-voice-target.f32.wav",
        "removed-vocal.primary.f32.wav", "manifest.json",
    ):
        assert (work / name).exists()
    removed, sr = sf.read(work / "removed-vocal.primary.f32.wav",
                          dtype="float32", always_2d=True)
    assert sr == 44100
    assert np.allclose(removed, mixture - primary, atol=1e-7)
    assert manifest["population_risk_claim"] is None
    assert json.loads((work / "manifest.json").read_text())["primary"]["candidate"] == "primary"


def test_package_refuses_to_replace_different_existing_file(tmp_path):
    frames = 16
    zero = np.zeros((frames, 2), dtype="float32")
    one = np.ones((frames, 2), dtype="float32")
    source = tmp_path / "source"
    for name, audio in (("primary.wav", zero), ("alternate.wav", zero),
                        ("mix.wav", zero), ("a.wav", zero), ("v.wav", zero)):
        _write(source / name, audio)
    report = {
        "work_id": "opera", "policy_id": "test", "status": "engineering_preview",
        "primary": "primary", "alternate": "alternate", "pareto_front": [],
        "candidates": [
            {"candidate": "primary", "path": str(source / "primary.wav"),
             "metrics": metrics(), "screening": screening(metrics(), policy())},
            {"candidate": "alternate", "path": str(source / "alternate.wav"),
             "metrics": metrics(), "screening": screening(metrics(), policy())},
        ],
        "truth": {"mixture": str(source / "mix.wav"),
                  "accompaniment": str(source / "a.wav"),
                  "vocal": str(source / "v.wav"), "frames": frames,
                  "sample_rate_hz": 44100},
    }
    output = tmp_path / "release"
    _write(output / "opera" / "accompaniment.primary.f32.wav", one)
    with pytest.raises(ClassicalReleaseError, match="replace differing"):
        package_work(report, output)


def test_policy_rejects_nonpositive_limit():
    data = {
        "policy_id": "bad",
        "retained_voice_coef_p90_max": 0.0,
        "event_hole_db_p90_max": 18.0,
        "event_hole_db_max_max": 30.0,
        "alpha_error_p90_max": 0.75,
        "stereo_width_dev_db_max": 0.5,
        "coherence_dev_max": 0.1,
        "artifact_ratio_p90_max": None,
    }
    with pytest.raises(ValueError, match="retained_voice_coef"):
        ScreeningPolicy.from_mapping(data)
