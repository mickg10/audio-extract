import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.classical_release import _audio_record, package_work
from tools.verify_classical_release import VerificationError, verify_work


def _write(path: Path, audio: np.ndarray, sr=44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype("float32"), sr, subtype="FLOAT")


def _candidate(path: Path, audio: np.ndarray, name: str, recipe_digit: str):
    return {
        "candidate": name,
        "recipe_id": "sha256:" + recipe_digit * 64,
        "path": str(path),
        "container_sha256": "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            audio, 44100, ["FL", "FR"], len(audio)
        ),
        "frames": len(audio),
        "sample_rate_hz": 44100,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
        "metrics": {
            "retained_voice_coef_p90": 0.05,
            "retained_voice_db_p90": -25.0,
            "event_hole_db_p90": 2.0,
            "event_hole_db_max": 4.0,
            "alpha_error_p90": 0.1,
            "artifact_ratio_p90": 0.02,
            "scale_dependent_sdr_db": 20.0,
            "stereo_width_dev_db/v2": 0.1,
            "interchannel_coherence_dev/v2": 0.01,
            "erb_envelope_dist_db/v2": 0.2,
            "_available_tiles": 10,
            "_total_tiles": 10,
        },
        "screening": {"passed": True, "checks": [], "failures": [], "critical_max": 0.0},
    }


def fixture(tmp_path: Path):
    frames = 4096
    t = np.arange(frames, dtype=np.float64) / 44100.0
    a = np.column_stack([
        0.2 * np.sin(2 * np.pi * 220 * t),
        0.2 * np.sin(2 * np.pi * 330 * t),
    ]).astype("float32")
    v = np.column_stack([
        0.1 * np.sin(2 * np.pi * 660 * t),
        0.08 * np.sin(2 * np.pi * 550 * t),
    ]).astype("float32")
    m = a + v
    p = a + 0.02 * v
    alt = a + 0.05 * v
    src = tmp_path / "source"
    paths = {
        "mixture": src / "M.f32.wav",
        "accompaniment": src / "A.f32.wav",
        "vocal": src / "V.f32.wav",
        "primary": src / "primary.f32.wav",
        "alternate": src / "alternate.f32.wav",
    }
    for role, audio in (
        ("mixture", m), ("accompaniment", a), ("vocal", v),
        ("primary", p), ("alternate", alt),
    ):
        _write(paths[role], audio)
    truth_records = {
        role: _audio_record(paths[role])
        for role in ("mixture", "accompaniment", "vocal")
    }
    report = {
        "work_id": "opera",
        "policy_id": "test",
        "status": "final",
        "release_scope": "exact_benchmark_qualified",
        "primary": "primary",
        "alternate": "alternate",
        "pareto_front": ["primary", "alternate"],
        "artifact_aliases": {"primary": [], "alternate": []},
        "candidates": [
            _candidate(paths["primary"], p, "primary", "1"),
            _candidate(paths["alternate"], alt, "alternate", "2"),
        ],
        "truth": {
            "mixture": str(paths["mixture"]),
            "accompaniment": str(paths["accompaniment"]),
            "vocal": str(paths["vocal"]),
            "frames": frames,
            "sample_rate_hz": 44100,
            "records": truth_records,
        },
    }
    package_root = tmp_path / "release"
    package_work(report, package_root)
    return report, paths, package_root, m, p


def test_verifier_accepts_complete_package_and_exact_lineage(tmp_path):
    report, _, package_root, _, _ = fixture(tmp_path)
    result = verify_work(report, package_root)
    assert result["work_id"] == "opera"
    assert result["status"] == "final"
    assert result["primary"] == "primary"
    assert result["alternate"] == "alternate"
    assert result["removed_vocal_lineage"]["recipe_id"].startswith("sha256:")
    assert set(result["packaged_artifacts"]) == {
        "accompaniment_primary", "accompaniment_alternate",
        "exact_orchestra_target", "exact_voice_target", "removed_vocal_primary",
    }


@pytest.mark.parametrize("role", ["mixture", "accompaniment", "vocal"])
def test_verifier_refuses_truth_mutation_after_scoring(tmp_path, role):
    report, paths, package_root, _, _ = fixture(tmp_path)
    _write(paths[role], np.ones((4096, 2), dtype="float32"))
    with pytest.raises(VerificationError, match=f"truth/{role}"):
        verify_work(report, package_root)


def test_verifier_refuses_candidate_mutation_after_scoring(tmp_path):
    report, paths, package_root, _, _ = fixture(tmp_path)
    _write(paths["primary"], np.ones((4096, 2), dtype="float32"))
    with pytest.raises(VerificationError, match="candidate/primary"):
        verify_work(report, package_root)


def test_verifier_refuses_package_mutation(tmp_path):
    report, _, package_root, _, _ = fixture(tmp_path)
    _write(
        package_root / "opera" / "accompaniment.primary.f32.wav",
        np.ones((4096, 2), dtype="float32"),
    )
    with pytest.raises(VerificationError, match="accompaniment_primary"):
        verify_work(report, package_root)


def test_verifier_refuses_removed_lineage_parent_change(tmp_path):
    report, _, package_root, _, _ = fixture(tmp_path)
    recipe_path = package_root / "opera" / "lineage" / "removed-vocal.primary" / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["model"]["members"] = ["sha256:" + "9" * 64]
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    with pytest.raises(VerificationError, match="recipe ID mismatch|primary recipe parent"):
        verify_work(report, package_root)


def test_verifier_refuses_byte_identical_primary_and_alternate(tmp_path):
    report, _, package_root, _, _ = fixture(tmp_path)
    report["candidates"][1]["artifact_pcm_sha256"] = report["candidates"][0][
        "artifact_pcm_sha256"
    ]
    with pytest.raises(VerificationError, match="artifact_pcm_sha256 mismatch|byte-identical"):
        verify_work(report, package_root)
