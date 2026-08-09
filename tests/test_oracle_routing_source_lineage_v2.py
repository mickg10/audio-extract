import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import canon, identity, recipe as recipe_mod
from audio_extract.oracle_routing_basis_v2 import (
    CandidateDeclaration,
    VerifiedCandidate,
)
from audio_extract.oracle_routing_source_lineage_v2 import (
    NO_VOCAL_ENSEMBLE_SCHEMA,
    SourceLineageError,
    audit_basis_source_lineage,
    expected_source_from_audio,
    verify_candidate_source_lineage,
)


def write_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype("float32"), 44_100, subtype="FLOAT")


def sha_file(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(
    directory: Path,
    *,
    work_id: str,
    name: str,
    recipe: dict,
    recipe_id: str,
    frames: int,
) -> VerifiedCandidate:
    output = directory / "output.f32.wav"
    write_audio(output, np.zeros((frames, 2), dtype="float32"))
    (directory / "recipe.json").write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\n"
    )
    audio, _ = sf.read(output, dtype="float32", always_2d=True)
    declaration = CandidateDeclaration(
        work_id=work_id,
        candidate_name=name,
        recipe_id=recipe_id,
        path=str(output),
        artifact_pcm_sha256=identity.artifact_pcm_sha256(
            audio, 44_100, ["FL", "FR"], len(audio)
        ),
        container_sha256=sha_file(output),
        frames=frames,
    )
    return VerifiedCandidate(declaration, str(output.resolve()))


def canonical_recipe(source, *, source_sha=None, frames=None):
    return {
        "schema": recipe_mod.SCHEMA,
        "canon": recipe_mod.CANON,
        "input_pcm": {
            "sha256": source_sha or source.artifact_pcm_sha256,
            "sample_rate_hz": 44_100,
            "channel_layout": ["FL", "FR"],
            "frames": frames or source.frames,
            "sample_format": "float32-le-interleaved",
        },
        "operation": {
            "type": "separate",
            "target": "vocals",
            "construction": "native_primary",
        },
        "model": {
            "model_id": "test",
            "weights_sha256": "1" * 64,
            "adapter": "tests",
            "adapter_revision": "v1",
        },
        "effective_config": {},
        "software": {"audio_extract_commit": "abc123"},
    }


def fixture(tmp_path: Path):
    frames = 256
    source_path = tmp_path / "truth" / "mix_with_voice.wav"
    t = np.arange(frames, dtype=np.float32) / 44_100.0
    audio = np.column_stack((
        0.1 * np.sin(2 * np.pi * 220 * t),
        0.1 * np.sin(2 * np.pi * 330 * t),
    ))
    write_audio(source_path, audio)
    source = expected_source_from_audio(
        source_path, work_id="work", role="voiced_mixture"
    )
    recipe = canonical_recipe(source)
    row = candidate(
        tmp_path / "candidate",
        work_id="work",
        name="model",
        recipe=recipe,
        recipe_id=identity.recipe_id(recipe),
        frames=frames,
    )
    return source, row


def test_canonical_recipe_binds_exact_source_pcm_and_grid(tmp_path):
    source, row = fixture(tmp_path)
    result = verify_candidate_source_lineage(row, source)
    assert result.source_pcm_sha256 == source.artifact_pcm_sha256
    assert result.recipe_id == row.declaration.recipe_id
    assert result.recipe_schema == recipe_mod.SCHEMA
    assert result.source_role == "voiced_mixture"


def test_recipe_source_pcm_mismatch_is_refused(tmp_path):
    source, row = fixture(tmp_path)
    recipe_path = Path(row.resolved_path).parent / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["input_pcm"]["sha256"] = "sha256:" + "0" * 64
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))
    broken = VerifiedCandidate(
        CandidateDeclaration(
            **{
                **row.declaration.__dict__,
                "recipe_id": identity.recipe_id(recipe),
            }
        ),
        row.resolved_path,
    )
    with pytest.raises(SourceLineageError, match="source PCM mismatch"):
        verify_candidate_source_lineage(broken, source)


def test_recipe_identity_mismatch_is_refused(tmp_path):
    source, row = fixture(tmp_path)
    broken = VerifiedCandidate(
        CandidateDeclaration(
            **{
                **row.declaration.__dict__,
                "recipe_id": "sha256:" + "9" * 64,
            }
        ),
        row.resolved_path,
    )
    with pytest.raises(SourceLineageError, match="recipe identity mismatch"):
        verify_candidate_source_lineage(broken, source)


def test_missing_or_symlinked_recipe_is_refused(tmp_path):
    source, row = fixture(tmp_path)
    recipe_path = Path(row.resolved_path).parent / "recipe.json"
    payload = recipe_path.read_bytes()
    recipe_path.unlink()
    with pytest.raises(SourceLineageError, match="lacks immutable sibling"):
        verify_candidate_source_lineage(row, source)

    external = tmp_path / "external.json"
    external.write_bytes(payload)
    recipe_path.symlink_to(external)
    with pytest.raises(SourceLineageError, match="symlinked recipe"):
        verify_candidate_source_lineage(row, source)


def test_recipe_source_grid_mismatch_is_refused(tmp_path):
    source, row = fixture(tmp_path)
    recipe_path = Path(row.resolved_path).parent / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["input_pcm"]["frames"] += 1
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))
    broken = VerifiedCandidate(
        CandidateDeclaration(
            **{
                **row.declaration.__dict__,
                "recipe_id": identity.recipe_id(recipe),
            }
        ),
        row.resolved_path,
    )
    with pytest.raises(SourceLineageError, match="source grid mismatch"):
        verify_candidate_source_lineage(broken, source)


def test_custom_no_vocal_ensemble_recipe_is_verified(tmp_path):
    frames = 128
    source_path = tmp_path / "truth" / "orchestra_only.wav"
    write_audio(source_path, np.full((frames, 2), 0.01, dtype="float32"))
    source = expected_source_from_audio(
        source_path,
        work_id="aalto_mozart_dry",
        role="no_vocal_accompaniment",
    )
    recipe = {
        "schema": NO_VOCAL_ENSEMBLE_SCHEMA,
        "work_id": "aalto_mozart_dry",
        "source_pcm_sha256": source.artifact_pcm_sha256,
        "source_grid": {
            "sample_rate_hz": 44_100,
            "channels": ["FL", "FR"],
            "frames": frames,
            "subtype": "FLOAT",
        },
        "candidate_name": "median_mdx_mel_bs",
        "operation": {
            "type": "mixture_minus_vocal_ensemble",
            "algorithm": "median",
        },
        "parents": [],
        "effective_config": {},
        "software": {"audio_extract_commit": "abc123"},
    }
    recipe_id = identity.blob_sha256(canon.canonicalize(recipe))
    row = candidate(
        tmp_path / "candidate",
        work_id="aalto_mozart_dry",
        name="median_mdx_mel_bs",
        recipe=recipe,
        recipe_id=recipe_id,
        frames=frames,
    )
    result = verify_candidate_source_lineage(row, source)
    assert result.recipe_schema == NO_VOCAL_ENSEMBLE_SCHEMA
    assert result.source_role == "no_vocal_accompaniment"


def test_audit_requires_exact_complete_work_set(tmp_path):
    source, row = fixture(tmp_path)
    report = audit_basis_source_lineage(
        [row], {"work": source}, role="voiced_mixture"
    )
    assert report["status"] == "pass"
    assert report["candidate_count"] == 1

    extra_path = tmp_path / "extra.wav"
    write_audio(extra_path, np.zeros((256, 2), dtype="float32"))
    extra = expected_source_from_audio(
        extra_path, work_id="extra", role="voiced_mixture"
    )
    with pytest.raises(SourceLineageError, match="work sets differ"):
        audit_basis_source_lineage(
            [row], {"work": source, "extra": extra},
            role="voiced_mixture",
        )
