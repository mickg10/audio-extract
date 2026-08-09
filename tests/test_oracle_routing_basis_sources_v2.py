import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.oracle_routing_basis_sources_v2 import (
    assemble_basis,
    load_audited_candidates_v2,
    normalize_audited_candidates_v2_row,
    normalize_sha256,
)
from audio_extract.oracle_routing_basis_v2 import BasisValidationError


def file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def audited_row(path: Path, *, candidate="median_mdx_mel_bs", recipe="1"):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return {
        "work_id": "work",
        "candidate": candidate,
        "recipe_id": "sha256:" + recipe * 64,
        "path": str(path),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            audio, sr, ["FL", "FR"], len(audio)
        ),
        "container_sha256": file_sha(path),
        "frames": len(audio),
        "sr_hz": sr,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
        "parent_candidate_recipe_ids": ["sha256:" + "a" * 64],
        # This is the exact legacy shape in datasets/candidates-v2.jsonl.
        "executed_model_bundle_hashes": ["b" * 64, "sha256:" + "c" * 64],
    }


def test_normalize_sha256_accepts_prefixed_and_legacy_bare_hex():
    assert normalize_sha256("a" * 64, "x") == "sha256:" + "a" * 64
    assert normalize_sha256("sha256:" + "B" * 64, "x") == (
        "sha256:" + "b" * 64
    )
    with pytest.raises(BasisValidationError):
        normalize_sha256("not-a-hash", "x")


def test_audited_row_normalizes_bare_bundle_id(tmp_path):
    path = tmp_path / "candidate.wav"
    sf.write(path, np.zeros((64, 2), dtype="float32"), 44_100, subtype="FLOAT")
    normalized = normalize_audited_candidates_v2_row(audited_row(path))
    assert normalized["candidate_name"] == "median_mdx_mel_bs"
    assert normalized["executed_model_bundle_hashes"] == [
        "sha256:" + "b" * 64,
        "sha256:" + "c" * 64,
    ]


def test_load_and_assemble_audited_manifest(tmp_path):
    path = tmp_path / "candidate.wav"
    signal = np.column_stack((
        np.linspace(-0.1, 0.1, 96),
        np.linspace(0.1, -0.1, 96),
    )).astype("float32")
    sf.write(path, signal, 44_100, subtype="FLOAT")
    manifest = tmp_path / "candidates-v2.jsonl"
    manifest.write_text(json.dumps(audited_row(path)) + "\n")
    declarations = load_audited_candidates_v2(manifest)
    assert declarations[0].executed_model_bundle_hashes == (
        "sha256:" + "b" * 64,
        "sha256:" + "c" * 64,
    )
    verified, deduplicated = assemble_basis(
        audited_candidate_manifests=(manifest,),
        required_aliases=("median_mdx_mel_bs",),
    )
    assert len(verified) == len(deduplicated) == 1
    assert deduplicated[0].canonical_name == "median_mdx_mel_bs"


def test_missing_audited_field_is_refused(tmp_path):
    path = tmp_path / "candidate.wav"
    sf.write(path, np.zeros((32, 2), dtype="float32"), 44_100, subtype="FLOAT")
    row = audited_row(path)
    del row["container_sha256"]
    with pytest.raises(BasisValidationError, match="lacks fields"):
        normalize_audited_candidates_v2_row(row)
