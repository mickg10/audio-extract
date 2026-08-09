import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract.oracle_routing_run_contract_v2 import (
    CANONICAL_RESOLUTIONS,
    RUN_INPUT_SCHEMA,
    RunContractError,
    build_truth_manifest,
    load_run_input_anchor,
    resolution_key,
    semantic_sha256,
    stable_file,
    validate_resolutions,
    verify_disjoint_manifest_groups,
    verify_truth_manifest,
)


def _write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype="float32"), 44_100, subtype="FLOAT")


def _truth_tree(root: Path, works=("opera",)) -> None:
    frames = 4096
    time = np.arange(frames, dtype=np.float64) / 44_100.0
    for offset, work in enumerate(works):
        accompaniment = np.column_stack((
            0.15 * np.sin(2 * np.pi * (220 + offset) * time),
            0.14 * np.sin(2 * np.pi * (330 + offset) * time),
        )).astype("float32")
        vocal = np.column_stack((
            0.05 * np.sin(2 * np.pi * (660 + offset) * time),
            0.04 * np.sin(2 * np.pi * (550 + offset) * time),
        )).astype("float32")
        mixture = accompaniment + vocal
        _write(root / work / "mix_with_voice.wav", mixture)
        _write(root / work / "orchestra_only.wav", accompaniment)
        _write(root / work / "voice_ref.wav", vocal)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_record(path: Path) -> dict[str, str]:
    _, record = stable_file(path)
    return {"path": record.path, "sha256": record.container_sha256}


def test_resolution_identity_is_exact_not_formatted():
    assert validate_resolutions(CANONICAL_RESOLUTIONS) == CANONICAL_RESOLUTIONS
    assert resolution_key(2.0) == "2.0"
    assert resolution_key(1) == "1.0"
    assert resolution_key(0.5) == "0.5"
    for value in (
        (2.04, 1.04, 0.54),
        (1.0, 2.0, 0.5),
        (2.0, 1.0),
        (2.0, 1.0, 0.5, 0.25),
        (2.0, True, 0.5),
    ):
        with pytest.raises(RunContractError, match="resolution"):
            validate_resolutions(value)
    with pytest.raises(RunContractError, match="outside preregistration"):
        resolution_key(1.04)


def test_manifest_groups_reject_different_hardlink_names(tmp_path):
    voiced = tmp_path / "voiced.jsonl"
    alias = tmp_path / "no-vocal.jsonl"
    voiced.write_text("voiced\n")
    os.link(voiced, alias)
    groups = {
        "voiced": [_file_record(voiced)],
        "no_vocal": [_file_record(alias)],
    }
    with pytest.raises(RunContractError, match="one physical file"):
        verify_disjoint_manifest_groups(groups)


def test_truth_manifest_reopens_all_roles_and_reproves_identity(tmp_path):
    truth = tmp_path / "truth"
    _truth_tree(truth, ("one", "two"))
    manifest = build_truth_manifest(truth, ("one", "two"))
    assert manifest["schema"].endswith("truth-manifest/v2")
    assert verify_truth_manifest(manifest) == semantic_sha256(manifest)
    for work in ("one", "two"):
        assert set(manifest["records"][work]["roles"]) == {
            "mixture", "accompaniment", "vocal"
        }
        assert manifest["records"][work]["mixture_identity_max_abs"] <= 2e-5


@pytest.mark.parametrize(
    "filename",
    ("mix_with_voice.wav", "orchestra_only.wav", "voice_ref.wav"),
)
def test_truth_manifest_rejects_post_freeze_role_mutation(tmp_path, filename):
    truth = tmp_path / "truth"
    _truth_tree(truth)
    manifest = build_truth_manifest(truth, ("opera",))
    path = truth / "opera" / filename
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio[100, 0] += 0.01
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    with pytest.raises(RunContractError, match="truth artifact changed"):
        verify_truth_manifest(manifest)


def test_truth_manifest_refuses_symlinked_role(tmp_path):
    truth = tmp_path / "truth"
    _truth_tree(truth)
    original = truth / "opera" / "voice_ref.wav"
    external = tmp_path / "external.wav"
    external.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(external)
    with pytest.raises(RunContractError, match="symlinked truth"):
        build_truth_manifest(truth, ("opera",))


def _anchored_fixture(tmp_path):
    truth = tmp_path / "truth"
    _truth_tree(truth)
    truth_value = build_truth_manifest(truth, ("opera",))
    truth_path = tmp_path / "truth-manifest-v2.json"
    _write_json(truth_path, truth_value)
    _, truth_file = stable_file(truth_path)

    voiced = tmp_path / "voiced.jsonl"
    no_vocal = tmp_path / "no-vocal.jsonl"
    voiced.write_text("voiced\n")
    no_vocal.write_text("no-vocal\n")
    run_config = {
        "resolutions_seconds": [2.0, 1.0, 0.5],
        "spectral": {"n_fft": 2048, "hop_length": 1024},
    }
    value = {
        "schema": RUN_INPUT_SCHEMA,
        "code_commit": "a" * 40,
        "works": ["opera"],
        "run_config": run_config,
        "source_manifest_groups": {
            "voiced": [_file_record(voiced)],
            "no_vocal": [_file_record(no_vocal)],
        },
        "truth_manifest": {
            "path": truth_file.path,
            "sha256": truth_file.container_sha256,
        },
    }
    run_path = tmp_path / "run-inputs-v2.json"
    _write_json(run_path, value)
    return run_path, run_config, truth_file.container_sha256


def test_run_input_anchor_binds_preexisting_bytes_config_work_and_truth(tmp_path):
    run_path, run_config, truth_sha = _anchored_fixture(tmp_path)
    anchor = load_run_input_anchor(
        run_path,
        expected_code_commit="a" * 40,
        expected_works=("opera",),
        expected_run_config=run_config,
        expected_truth_manifest_sha256=truth_sha,
    )
    assert anchor.sha256.startswith("sha256:")
    assert anchor.resolutions_seconds == CANONICAL_RESOLUTIONS
    assert anchor.truth_manifest_sha256 == truth_sha


def test_run_input_anchor_refuses_posthoc_config_or_commit(tmp_path):
    run_path, run_config, truth_sha = _anchored_fixture(tmp_path)
    with pytest.raises(RunContractError, match="code commit"):
        load_run_input_anchor(
            run_path,
            expected_code_commit="b" * 40,
            expected_works=("opera",),
            expected_run_config=run_config,
            expected_truth_manifest_sha256=truth_sha,
        )
    changed = dict(run_config)
    changed["spectral"] = {"n_fft": 4096, "hop_length": 1024}
    with pytest.raises(RunContractError, match="configuration"):
        load_run_input_anchor(
            run_path,
            expected_code_commit="a" * 40,
            expected_works=("opera",),
            expected_run_config=changed,
            expected_truth_manifest_sha256=truth_sha,
        )


def test_run_input_anchor_refuses_hardlinked_source_groups(tmp_path):
    run_path, run_config, truth_sha = _anchored_fixture(tmp_path)
    value = json.loads(run_path.read_text())
    voiced_path = Path(value["source_manifest_groups"]["voiced"][0]["path"])
    no_vocal_path = Path(value["source_manifest_groups"]["no_vocal"][0]["path"])
    no_vocal_path.unlink()
    os.link(voiced_path, no_vocal_path)
    value["source_manifest_groups"]["no_vocal"][0] = _file_record(no_vocal_path)
    _write_json(run_path, value)
    with pytest.raises(RunContractError, match="one physical file"):
        load_run_input_anchor(
            run_path,
            expected_code_commit="a" * 40,
            expected_works=("opera",),
            expected_run_config=run_config,
            expected_truth_manifest_sha256=truth_sha,
        )
