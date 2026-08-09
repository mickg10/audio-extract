import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.oracle_routing_basis_v2 import (
    BasisValidationError,
    CandidateDeclaration,
    basis_report,
    deduplicate_basis,
    load_declarations,
    verify_basis,
    verify_candidate,
    write_jsonl,
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def write_float(path: Path, audio: np.ndarray, sr: int = 44_100) -> dict:
    sf.write(path, audio.astype("float32"), sr, subtype="FLOAT")
    reopened, actual_sr = sf.read(path, dtype="float32", always_2d=True)
    return {
        "artifact": identity.artifact_pcm_sha256(
            reopened, actual_sr, ["FL", "FR"], len(reopened)
        ),
        "container": sha_file(path),
        "frames": len(reopened),
    }


def declaration(path: Path, facts: dict, *, name: str, recipe: str) -> CandidateDeclaration:
    return CandidateDeclaration(
        work_id="work",
        candidate_name=name,
        recipe_id="sha256:" + recipe * 64,
        path=str(path),
        artifact_pcm_sha256=facts["artifact"],
        container_sha256=facts["container"],
        frames=facts["frames"],
        parent_recipe_ids=("sha256:" + "a" * 64,),
        executed_model_bundle_hashes=("sha256:" + "b" * 64,),
    )


def test_verify_and_deduplicate_preserves_median_alias(tmp_path):
    rng = np.random.default_rng(4)
    audio = rng.normal(0, 0.02, size=(512, 2)).astype("float32")
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    facts = write_float(first, audio)
    second.write_bytes(first.read_bytes())
    second_facts = {
        "artifact": facts["artifact"],
        "container": sha_file(second),
        "frames": facts["frames"],
    }
    rows = verify_basis((
        declaration(
            first, facts, name="residual_mdx23c", recipe="1"
        ),
        declaration(
            second, second_facts, name="median_mdx_mel_bs", recipe="2"
        ),
    ))
    deduplicated = deduplicate_basis(
        rows, required_aliases=("median_mdx_mel_bs",)
    )
    assert len(deduplicated) == 1
    assert deduplicated[0].canonical_name == "median_mdx_mel_bs"
    assert set(deduplicated[0].aliases) == {
        "median_mdx_mel_bs", "residual_mdx23c"
    }
    report = basis_report(rows, deduplicated)
    assert report["status"] == "pass"
    assert report["duplicates_collapsed"] == 1


def test_verify_candidate_refuses_container_mismatch(tmp_path):
    audio = np.zeros((256, 2), dtype="float32")
    path = tmp_path / "audio.wav"
    facts = write_float(path, audio)
    row = declaration(path, facts, name="candidate", recipe="3")
    broken = CandidateDeclaration(
        **{**row.__dict__, "container_sha256": "sha256:" + "0" * 64}
    )
    with pytest.raises(BasisValidationError, match="container identity mismatch"):
        verify_candidate(broken)


def test_verify_candidate_refuses_integer_subtype(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros((128, 2), dtype="float32"), 44_100, subtype="PCM_16")
    row = CandidateDeclaration(
        work_id="work", candidate_name="candidate",
        recipe_id="sha256:" + "4" * 64, path=str(path),
        artifact_pcm_sha256="sha256:" + "5" * 64,
        container_sha256=sha_file(path), frames=128,
    )
    with pytest.raises(BasisValidationError, match="published grid mismatch"):
        verify_candidate(row)


def test_verify_basis_refuses_mixed_grids(tmp_path):
    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    facts = [
        write_float(paths[0], np.zeros((128, 2), dtype="float32")),
        write_float(paths[1], np.zeros((129, 2), dtype="float32")),
    ]
    rows = [
        verify_candidate(declaration(
            paths[0], facts[0], name="a", recipe="6"
        )),
        verify_candidate(declaration(
            paths[1], facts[1], name="b", recipe="7"
        )),
    ]
    with pytest.raises(BasisValidationError, match="grids differ within work"):
        verify_basis(tuple(row.declaration for row in rows))


def test_required_alias_is_enforced(tmp_path):
    path = tmp_path / "audio.wav"
    facts = write_float(path, np.zeros((64, 2), dtype="float32"))
    rows = verify_basis((declaration(
        path, facts, name="residual_mdx23c", recipe="8"
    ),))
    with pytest.raises(BasisValidationError, match="lacks required"):
        deduplicate_basis(rows, required_aliases=("median_mdx_mel_bs",))


def test_load_audited_candidates_v2_shape_and_write_immutable_jsonl(tmp_path):
    path = tmp_path / "audio.wav"
    facts = write_float(path, np.zeros((96, 2), dtype="float32"))
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps({
        "work_id": "work",
        "candidate": "median_mdx_mel_bs",
        "recipe_id": "sha256:" + "9" * 64,
        "path": str(path),
        "artifact_pcm_sha256": facts["artifact"],
        "container_sha256": facts["container"],
        "frames": facts["frames"],
        "sr_hz": 44_100,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
        "parent_candidate_recipe_ids": ["sha256:" + "a" * 64],
        "executed_model_bundle_hashes": ["sha256:" + "b" * 64],
    }) + "\n")
    declarations = load_declarations((manifest,))
    verified = verify_basis(declarations)
    deduplicated = deduplicate_basis(
        verified, required_aliases=("median_mdx_mel_bs",)
    )
    output = tmp_path / "basis-v2.jsonl"
    write_jsonl(output, deduplicated)
    write_jsonl(output, deduplicated)
    assert json.loads(output.read_text())["canonical_name"] == "median_mdx_mel_bs"


def test_host_prefixed_path_requires_explicit_local_alias(tmp_path):
    path = tmp_path / "audio.wav"
    facts = write_float(path, np.zeros((32, 2), dtype="float32"))
    row = declaration(path, facts, name="candidate", recipe="c")
    remote = CandidateDeclaration(
        **{**row.__dict__, "path": f"research6:{path}"}
    )
    with pytest.raises(BasisValidationError, match="is not this execution host"):
        verify_candidate(remote)
    verified = verify_candidate(
        remote, allowed_host_aliases=("research6",)
    )
    assert Path(verified.resolved_path) == path.resolve()
