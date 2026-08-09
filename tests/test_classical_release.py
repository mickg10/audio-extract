import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.classical_release import (
    CandidateRow,
    ClassicalReleaseError,
    ScreeningPolicy,
    package_work,
    pareto_front,
    rank_rows,
    screening,
    verify_candidate,
)


def policy():
    return ScreeningPolicy(
        policy_id="test",
        retained_voice_coef_p90_max=0.12,
        retained_voice_db_p90_max=-18.0,
        event_hole_db_p90_max=18.0,
        event_hole_db_max_max=30.0,
        alpha_error_p90_max=0.75,
        stereo_width_dev_db_max=0.5,
        coherence_dev_max=0.1,
        min_identifiable_tiles=10,
        min_identifiable_fraction=0.05,
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
        "_available_tiles": 100,
        "_total_tiles": 200,
        "identifiable_fraction": 0.5,
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


def test_screening_requires_audible_voice_and_exact_label_coverage():
    poor = metrics(
        retained_voice_db_p90=-10.0,
        _available_tiles=2,
        _total_tiles=200,
        identifiable_fraction=0.01,
    )
    result = screening(poor, policy())
    assert not result["passed"]
    assert {failure["metric"] for failure in result["failures"]} == {
        "retained_voice_db_p90", "_available_tiles", "identifiable_fraction"
    }


def test_pareto_front_keeps_tradeoffs_and_removes_dominated_candidate():
    voice_best = row("voice_best", metrics(retained_voice_coef_p90=0.01,
                                            retained_voice_db_p90=-30.0,
                                            event_hole_db_p90=15.0))
    hole_best = row("hole_best", metrics(retained_voice_coef_p90=0.08,
                                          retained_voice_db_p90=-19.0,
                                          event_hole_db_p90=5.0))
    dominated = row("dominated", metrics(retained_voice_coef_p90=0.09,
                                          retained_voice_db_p90=-18.5,
                                          event_hole_db_p90=16.0,
                                          artifact_ratio_p90=0.3))
    assert set(pareto_front([voice_best, hole_best, dominated])) == {
        "voice_best", "hole_best"
    }


def _write(path: Path, audio: np.ndarray, sr=44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype("float32"), sr, subtype="FLOAT")


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_record(path: Path, audio: np.ndarray, name: str) -> dict:
    return {
        "candidate": name,
        "recipe_id": "sha256:" + ("1" if name == "primary" else "2") * 64,
        "path": str(path),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            audio.astype("float32"), 44100, ["FL", "FR"], len(audio)
        ),
        "container_sha256": _sha(path),
        "frames": len(audio),
        "sample_rate_hz": 44100,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
        "metrics": metrics(),
        "screening": screening(metrics(), policy()),
    }


def _package_fixture(tmp_path: Path):
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
    paths = {
        "primary": source / "primary.f32.wav",
        "alternate": source / "alternate.f32.wav",
        "mixture": source / "mix.f32.wav",
        "accompaniment": source / "a.f32.wav",
        "vocal": source / "v.f32.wav",
    }
    for name, audio in (
        ("primary", primary), ("alternate", alternate),
        ("mixture", mixture), ("accompaniment", accompaniment), ("vocal", vocal),
    ):
        _write(paths[name], audio)
    primary_row = _candidate_record(paths["primary"], primary, "primary")
    alternate_row = _candidate_record(paths["alternate"], alternate, "alternate")
    alternate_row["metrics"] = metrics(retained_voice_coef_p90=0.07)
    alternate_row["screening"] = screening(alternate_row["metrics"], policy())
    report = {
        "work_id": "opera", "policy_id": "test", "status": "final",
        "release_scope": "exact_benchmark_qualified",
        "primary": "primary", "alternate": "alternate",
        "pareto_front": ["primary", "alternate"],
        "artifact_aliases": {"primary": [], "alternate": []},
        "candidates": [primary_row, alternate_row],
        "truth": {
            "mixture": str(paths["mixture"]),
            "accompaniment": str(paths["accompaniment"]),
            "vocal": str(paths["vocal"]),
            "frames": frames, "sample_rate_hz": 44100,
        },
    }
    return report, paths, mixture, primary


def test_package_work_copies_and_inventories_every_artifact(tmp_path):
    report, paths, mixture, primary = _package_fixture(tmp_path)
    output = tmp_path / "release"
    manifest = package_work(report, output)
    work = output / "opera"
    expected_names = {
        "accompaniment.primary.f32.wav", "accompaniment.alternate.f32.wav",
        "exact-orchestra-target.f32.wav", "exact-voice-target.f32.wav",
        "removed-vocal.primary.f32.wav", "manifest.json", "report.json",
        "report.md", "COMPLETE",
    }
    assert expected_names <= {path.name for path in work.iterdir()}
    assert not os.path.samefile(paths["primary"], work / "accompaniment.primary.f32.wav")
    assert not os.path.samefile(paths["accompaniment"], work / "exact-orchestra-target.f32.wav")
    removed, sr = sf.read(
        work / "removed-vocal.primary.f32.wav", dtype="float32", always_2d=True
    )
    assert sr == 44100
    assert np.allclose(removed, mixture - primary, atol=1e-7)
    assert manifest["population_risk_claim"] is None
    assert manifest["status"] == "final"
    assert set(manifest["packaged_artifacts"]) == {
        "accompaniment_primary", "accompaniment_alternate",
        "exact_orchestra_target", "exact_voice_target", "removed_vocal_primary",
    }
    for record in manifest["packaged_artifacts"].values():
        assert record["container_sha256"].startswith("sha256:")
        assert record["artifact_pcm_sha256"].startswith("sha256:")
        assert record["frames"] == len(mixture)
        assert record["sample_rate_hz"] == 44100
    stored = json.loads((work / "manifest.json").read_text())
    assert stored["primary"]["candidate"] == "primary"
    lineage = work / "lineage" / "removed-vocal.primary"
    recipe = json.loads((lineage / "recipe.json").read_text())
    assert identity.recipe_id(recipe) == stored["removed_vocal_recipe_id"]
    assert recipe["model"]["members"] == [stored["primary"]["recipe_id"]]
    assert (lineage / "COMPLETE").is_file()


def test_package_refuses_stale_source_bytes_even_when_destination_absent(tmp_path):
    report, paths, _, _ = _package_fixture(tmp_path)
    _write(paths["primary"], np.ones((2048, 2), dtype="float32"))
    with pytest.raises(ClassicalReleaseError, match="source changed since scoring"):
        package_work(report, tmp_path / "release")


def test_package_refuses_to_replace_different_existing_file(tmp_path):
    report, _, _, _ = _package_fixture(tmp_path)
    output = tmp_path / "release"
    _write(output / "opera" / "accompaniment.primary.f32.wav",
           np.ones((2048, 2), dtype="float32"))
    with pytest.raises(ClassicalReleaseError, match="packaged artifact mismatch"):
        package_work(report, output)


def test_verify_candidate_rejects_truth_grid_sample_rate_mismatch(tmp_path):
    audio = np.zeros((100, 2), dtype="float32")
    path = tmp_path / "candidate.wav"
    _write(path, audio, sr=48000)
    row = CandidateRow(
        work_id="work", candidate="candidate",
        recipe_id="sha256:" + "1" * 64, path=str(path),
        artifact_pcm_sha256=identity.artifact_pcm_sha256(
            audio, 48000, ["FL", "FR"], len(audio)
        ),
        container_sha256=_sha(path), sr_hz=48000, frames=len(audio),
        channels=("FL", "FR"), subtype="FLOAT",
    )
    with pytest.raises(ClassicalReleaseError, match="truth-grid mismatch"):
        verify_candidate(row, expected_sr=44100, expected_frames=len(audio))


def test_policy_rejects_nonpositive_limit():
    data = {
        "policy_id": "bad",
        "retained_voice_coef_p90_max": 0.0,
        "retained_voice_db_p90_max": -18.0,
        "event_hole_db_p90_max": 18.0,
        "event_hole_db_max_max": 30.0,
        "alpha_error_p90_max": 0.75,
        "stereo_width_dev_db_max": 0.5,
        "coherence_dev_max": 0.1,
        "min_identifiable_tiles": 10,
        "min_identifiable_fraction": 0.05,
        "artifact_ratio_p90_max": None,
    }
    with pytest.raises(ValueError, match="retained_voice_coef"):
        ScreeningPolicy.from_mapping(data)
