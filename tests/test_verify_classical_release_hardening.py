"""Adversarial mutation coverage for the independent release verifier."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.classical_release import _audio_record, package_work
from tools.verify_classical_release import VerificationError, run, verify_work


def _write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype("float32"), 44100, subtype="FLOAT")


def _candidate(path: Path, audio: np.ndarray, name: str, digit: str) -> dict:
    return {
        "candidate": name,
        "recipe_id": "sha256:" + digit * 64,
        "path": str(path),
        "container_sha256": "sha256:" + __import__("hashlib").sha256(
            path.read_bytes()
        ).hexdigest(),
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
        "screening": {
            "passed": True,
            "checks": [],
            "failures": [],
            "critical_max": 0.0,
        },
    }


def _package_fixture(tmp_path: Path):
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
    primary = accompaniment + 0.02 * vocal
    alternate = accompaniment + 0.05 * vocal
    source = tmp_path / "source"
    paths = {
        "mixture": source / "M.f32.wav",
        "accompaniment": source / "A.f32.wav",
        "vocal": source / "V.f32.wav",
        "primary": source / "primary.f32.wav",
        "alternate": source / "alternate.f32.wav",
    }
    for role, audio in (
        ("mixture", mixture),
        ("accompaniment", accompaniment),
        ("vocal", vocal),
        ("primary", primary),
        ("alternate", alternate),
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
            _candidate(paths["primary"], primary, "primary", "1"),
            _candidate(paths["alternate"], alternate, "alternate", "2"),
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
    return report, paths, package_root


def _manifest(package_root: Path) -> Path:
    return package_root / "opera" / "manifest.json"


def _rewrite_manifest(package_root: Path, document: dict) -> None:
    _manifest(package_root).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )


def test_top_level_run_binds_report_and_manifest_commit(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    top = {
        "schema": "audio-extract/classical-exact-release-report/v1",
        "status": "final",
        "release_scope": "exact_linear_reference_engineering_release",
        "population_risk_claim": None,
        "code_commit": "unknown",
        "works": [report],
    }
    report_path = tmp_path / "release-report.json"
    report_path.write_text(json.dumps(top, indent=2, sort_keys=True) + "\n")
    result = run(report_path, package_root, None)
    assert result["schema"] == (
        "audio-extract/classical-release-verification/v2"
    )
    assert result["status"] == "verified"


def test_manifest_artifact_record_mutation_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    document = json.loads(_manifest(package_root).read_text())
    document["packaged_artifacts"]["accompaniment_primary"][
        "container_sha256"
    ] = "sha256:" + "0" * 64
    _rewrite_manifest(package_root, document)
    with pytest.raises(
        VerificationError,
        match=r"accompaniment_primary/manifest container_sha256 mismatch",
    ):
        verify_work(report, package_root)


@pytest.mark.parametrize(
    "bad_path", ["/tmp/outside.f32.wav", "../outside.f32.wav"]
)
def test_package_path_escape_is_refused(tmp_path, bad_path):
    report, _, package_root = _package_fixture(tmp_path)
    document = json.loads(_manifest(package_root).read_text())
    document["packaged_artifacts"]["accompaniment_primary"]["path"] = bad_path
    _rewrite_manifest(package_root, document)
    with pytest.raises(VerificationError, match="escapes work directory"):
        verify_work(report, package_root)


def test_symlinked_package_artifact_is_refused(tmp_path):
    report, paths, package_root = _package_fixture(tmp_path)
    linked = package_root / "opera" / "linked.f32.wav"
    linked.symlink_to(paths["primary"])
    document = json.loads(_manifest(package_root).read_text())
    document["packaged_artifacts"]["accompaniment_primary"]["path"] = linked.name
    _rewrite_manifest(package_root, document)
    with pytest.raises(VerificationError, match="symlink"):
        verify_work(report, package_root)


def test_manifest_primary_row_mutation_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    document = json.loads(_manifest(package_root).read_text())
    document["primary"]["recipe_id"] = "sha256:" + "9" * 64
    _rewrite_manifest(package_root, document)
    with pytest.raises(
        VerificationError, match="manifest/report primary candidate mismatch"
    ):
        verify_work(report, package_root)


def test_packaged_report_json_mutation_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    path = package_root / "opera" / "report.json"
    document = json.loads(path.read_text())
    document["primary"] = "alternate"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with pytest.raises(VerificationError, match="packaged report.json mismatch"):
        verify_work(report, package_root)


def test_packaged_report_markdown_mutation_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    (package_root / "opera" / "report.md").write_text("# incorrect\n")
    with pytest.raises(VerificationError, match="report.md lacks required facts"):
        verify_work(report, package_root)


def test_removed_pcm_marker_mutation_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    path = (
        package_root
        / "opera"
        / "lineage"
        / "removed-vocal.primary"
        / "output.pcm.sha256"
    )
    path.write_text("sha256:" + "0" * 64 + "\n")
    with pytest.raises(VerificationError, match="PCM marker mismatch"):
        verify_work(report, package_root)


def test_removed_execution_output_mutation_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    path = (
        package_root
        / "opera"
        / "lineage"
        / "removed-vocal.primary"
        / "execution.json"
    )
    document = json.loads(path.read_text())
    document["output"]["frames"] += 1
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with pytest.raises(VerificationError, match="execution output mismatch"):
        verify_work(report, package_root)


def test_symlinked_lineage_metadata_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    path = (
        package_root
        / "opera"
        / "lineage"
        / "removed-vocal.primary"
        / "recipe.json"
    )
    external = tmp_path / "external-recipe.json"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(VerificationError, match="symlink"):
        verify_work(report, package_root)


def test_nonempty_complete_marker_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    (package_root / "opera" / "COMPLETE").write_text("not complete\n")
    with pytest.raises(VerificationError, match="COMPLETE.*not empty"):
        verify_work(report, package_root)


def test_unsafe_work_id_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    mutated = dict(report)
    mutated["work_id"] = "../opera"
    with pytest.raises(VerificationError, match="unsafe work ID"):
        verify_work(mutated, package_root)


def test_release_manifest_commit_mismatch_is_refused(tmp_path):
    report, _, package_root = _package_fixture(tmp_path)
    with pytest.raises(VerificationError, match="code commit mismatch"):
        verify_work(
            report,
            package_root,
            expected_code_commit="deadbeef",
        )
