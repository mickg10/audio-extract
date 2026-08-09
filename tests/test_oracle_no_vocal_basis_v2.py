import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.oracle_no_vocal_basis_v2 import (
    NoVocalBasisError,
    build_complete_basis,
    ensemble_accompaniment,
    write_strict_manifest,
)
from audio_extract.oracle_routing_basis_v2 import (
    deduplicate_basis,
    verify_basis,
)
from audio_extract.oracle_routing_runner_v2 import REQUIRED_ALIASES


SR = 44_100


def file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_float(path: Path, audio: np.ndarray) -> tuple[str, str]:
    sf.write(path, audio.astype("float32"), SR, subtype="FLOAT")
    reopened, _ = sf.read(path, dtype="float32", always_2d=True)
    return (
        identity.artifact_pcm_sha256(
            reopened, SR, ["FL", "FR"], len(reopened)
        ),
        file_sha(path),
    )


def test_median_and_uniform_use_implied_vocal_relationship():
    source = np.asarray([
        [1.0, 0.5], [0.7, -0.2], [-0.4, 0.3], [0.2, 0.9]
    ])
    vocals = {
        "residual_mdx23c": np.asarray([
            [0.1, 0.0], [0.2, 0.1], [0.0, -0.1], [0.3, 0.2]
        ]),
        "residual_melband": np.asarray([
            [0.0, 0.1], [0.1, 0.0], [0.2, 0.1], [0.1, 0.3]
        ]),
        "residual_bs_roformer": np.asarray([
            [0.2, 0.2], [0.0, 0.2], [0.1, 0.0], [0.2, 0.1]
        ]),
    }
    residuals = {name: source - vocal for name, vocal in vocals.items()}
    median = ensemble_accompaniment(source, residuals, "median")
    uniform = ensemble_accompaniment(source, residuals, "uniform_mean")
    vocal_stack = np.stack(list(vocals.values()))
    assert np.allclose(median, source - np.median(vocal_stack, axis=0))
    assert np.allclose(uniform, source - np.mean(vocal_stack, axis=0))


def test_geometric_median_of_identical_members_is_exact_member():
    frames = 4096
    time = np.arange(frames) / SR
    source = np.column_stack((
        0.5 * np.sin(2 * np.pi * 210 * time),
        0.4 * np.sin(2 * np.pi * 310 * time),
    ))
    vocal = np.column_stack((
        0.1 * np.sin(2 * np.pi * 440 * time),
        0.1 * np.sin(2 * np.pi * 440 * time + 0.1),
    ))
    residual = source - vocal
    result = ensemble_accompaniment(
        source,
        {name: residual for name in (
            "residual_mdx23c",
            "residual_melband",
            "residual_bs_roformer",
        )},
        "stft_geometric_median",
    )
    assert result.shape == source.shape
    assert np.max(np.abs(result - residual.astype("float32"))) < 2e-5


def fixture(tmp_path: Path):
    frames = 4096
    time = np.arange(frames) / SR
    source = np.column_stack((
        0.4 * np.sin(2 * np.pi * 230 * time),
        0.3 * np.sin(2 * np.pi * 330 * time + 0.2),
    )).astype("float32")
    source_path = tmp_path / "orchestra_only.wav"
    source_pcm, _ = write_float(source_path, source)
    source, _ = sf.read(source_path, dtype="float32", always_2d=True)

    names = (
        "htdemucs_04573f0d",
        "htdemucs_955717e8",
        "mdx23c",
        "melband",
        "bs_roformer",
    )
    # Keep the three ensemble members identical in this integration fixture so
    # median, geometric median, and uniform mean have an unambiguous exact answer.
    voice = np.column_stack((
        0.03 * np.sin(2 * np.pi * 440 * time),
        0.02 * np.sin(2 * np.pi * 440 * time + 0.1),
    )).astype("float32")
    members = []
    for index, name in enumerate(names):
        candidate = source - (
            voice if name in {"mdx23c", "melband", "bs_roformer"}
            else (index + 1) * 0.25 * voice
        )
        path = tmp_path / f"{name}.wav"
        artifact, container = write_float(path, candidate)
        members.append({
            "name": name,
            "recipe_id": "sha256:" + format(index + 1, "x") * 64,
            "parent_vocal_recipe_id": "sha256:" + "a" * 64,
            "path": str(path),
            "artifact_pcm_sha256": artifact,
            "container_sha256": container,
            "executed_bundle_hash": "sha256:" + "b" * 64,
            "cached": False,
        })
    manifest = tmp_path / "control.json"
    manifest.write_text(json.dumps({
        "schema": "audio-extract/oracle-no-vocal-basis/v1",
        "status": "complete",
        "work_id": "aalto_mozart_dry",
        "source_pcm_sha256": source_pcm,
        "members": members,
    }, indent=2) + "\n")
    return source_path, manifest


def test_build_complete_basis_writes_eight_strict_aliases_and_replays(tmp_path):
    source, manifest = fixture(tmp_path)
    output = tmp_path / "generated"
    rows = build_complete_basis(
        control_manifest=manifest,
        source_audio=source,
        output_root=output,
        code_commit="test-commit",
    )
    assert len(rows) == 8
    assert {row.candidate_name for row in rows} == set(REQUIRED_ALIASES)
    verified = verify_basis(rows)
    deduplicated = deduplicate_basis(
        verified, required_aliases=REQUIRED_ALIASES
    )
    aliases = {alias for row in deduplicated for alias in row.aliases}
    assert aliases == set(REQUIRED_ALIASES)

    strict = tmp_path / "no-vocal-basis-v2.jsonl"
    write_strict_manifest(strict, rows)
    write_strict_manifest(strict, rows)
    assert len(strict.read_text().splitlines()) == 8

    replay = build_complete_basis(
        control_manifest=manifest,
        source_audio=source,
        output_root=output,
        code_commit="test-commit",
    )
    assert replay == rows


def test_source_identity_mismatch_is_refused(tmp_path):
    source, manifest = fixture(tmp_path)
    audio, _ = sf.read(source, dtype="float32", always_2d=True)
    audio[0, 0] += 0.1
    sf.write(source, audio, SR, subtype="FLOAT")
    with pytest.raises(NoVocalBasisError, match="source identity"):
        build_complete_basis(
            control_manifest=manifest,
            source_audio=source,
            output_root=tmp_path / "generated",
            code_commit="test-commit",
        )
