"""Complete the certified Aalto no-vocal candidate basis.

The existing no-vocal renderer publishes the two HTDemucs and three individual
MDX/Mel/BS residual accompaniments. The certified routing comparison also needs
the current waveform median champion, STFT geometric median, and uniform convex
fusion on the same orchestra-only source.

This module verifies all five singles, derives the three ensemble accompaniments
from their exact residual relationship, writes immutable FLOAT artifacts, and
emits strict basis-v2 rows. No alignment, resampling, truncation, normalization,
or lossy conversion is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import canon, identity
from .oracle_routing_basis_v2 import CandidateDeclaration
from .separate import _stft_geometric_median

CONTROL_SCHEMA = "audio-extract/oracle-no-vocal-basis/v1"
ENSEMBLE_RECIPE_SCHEMA = "audio-extract/oracle-no-vocal-ensemble/v2"

SINGLE_NAME_MAP = {
    "htdemucs_04573f0d": "htdemucs_04573f0d",
    "htdemucs_955717e8": "htdemucs_955717e8",
    "mdx23c": "residual_mdx23c",
    "melband": "residual_melband",
    "bs_roformer": "residual_bs_roformer",
}
ENSEMBLE_PARENTS = (
    "residual_mdx23c",
    "residual_melband",
    "residual_bs_roformer",
)


class NoVocalBasisError(RuntimeError):
    """The control basis or generated ensemble violates an exact invariant."""


@dataclass(frozen=True)
class ControlMember:
    name: str
    recipe_id: str
    parent_vocal_recipe_id: str
    path: str
    artifact_pcm_sha256: str
    container_sha256: str
    executed_bundle_hash: str


@dataclass(frozen=True)
class VerifiedControlMember:
    member: ControlMember
    strict_name: str
    resolved_path: str
    audio: np.ndarray



def _sha(value: Any, name: str) -> str:
    result = str(value or "").strip().lower()
    if not result.startswith("sha256:"):
        result = "sha256:" + result
    if len(result) != 71:
        raise NoVocalBasisError(f"{name} must be sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise NoVocalBasisError(f"{name} must be sha256:<64 hex>") from exc
    return result


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _strict_audio(path: Path, *, frames: int | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    if info.samplerate != 44_100 or info.channels != 2 or info.subtype != "FLOAT":
        raise NoVocalBasisError(
            f"expected 44.1-kHz stereo FLOAT: {path}: {info}"
        )
    if frames is not None and info.frames != frames:
        raise NoVocalBasisError(
            f"frame mismatch for {path}: {info.frames} != {frames}"
        )
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape != (info.frames, 2) or not np.all(np.isfinite(audio)):
        raise NoVocalBasisError(f"invalid decoded audio: {path}")
    return audio, int(sample_rate)


def _pcm(audio: np.ndarray) -> str:
    return identity.artifact_pcm_sha256(
        np.asarray(audio, dtype="float32"), 44_100, ["FL", "FR"], len(audio)
    )


def load_control_manifest(path: Path) -> tuple[str, str, tuple[ControlMember, ...]]:
    value = json.loads(path.read_text())
    if value.get("schema") != CONTROL_SCHEMA or value.get("status") != "complete":
        raise NoVocalBasisError(f"invalid no-vocal control manifest: {path}")
    work_id = str(value.get("work_id") or "").strip()
    source_pcm = _sha(value.get("source_pcm_sha256"), "source_pcm_sha256")
    if not work_id:
        raise NoVocalBasisError("control manifest lacks work_id")
    result = []
    seen = set()
    for raw in value.get("members") or ():
        name = str(raw.get("name") or "").strip()
        if name not in SINGLE_NAME_MAP or name in seen:
            raise NoVocalBasisError(f"unexpected/duplicate control member {name!r}")
        seen.add(name)
        result.append(ControlMember(
            name=name,
            recipe_id=_sha(raw.get("recipe_id"), f"{name}.recipe_id"),
            parent_vocal_recipe_id=_sha(
                raw.get("parent_vocal_recipe_id"),
                f"{name}.parent_vocal_recipe_id",
            ),
            path=str(raw.get("path") or "").strip(),
            artifact_pcm_sha256=_sha(
                raw.get("artifact_pcm_sha256"),
                f"{name}.artifact_pcm_sha256",
            ),
            container_sha256=_sha(
                raw.get("container_sha256"), f"{name}.container_sha256"
            ),
            executed_bundle_hash=_sha(
                raw.get("executed_bundle_hash"),
                f"{name}.executed_bundle_hash",
            ),
        ))
    if seen != set(SINGLE_NAME_MAP):
        raise NoVocalBasisError(
            f"control manifest member set {sorted(seen)} != "
            f"{sorted(SINGLE_NAME_MAP)}"
        )
    return work_id, source_pcm, tuple(result)


def verify_control_members(
    members: Sequence[ControlMember],
    *,
    frames: int,
) -> tuple[VerifiedControlMember, ...]:
    result = []
    for member in members:
        path = Path(member.path).resolve()
        if not path.is_file():
            raise NoVocalBasisError(f"control member is missing: {member.path}")
        audio, _ = _strict_audio(path, frames=frames)
        if _sha_file(path) != member.container_sha256:
            raise NoVocalBasisError(f"container changed for {member.name}")
        if _pcm(audio) != member.artifact_pcm_sha256:
            raise NoVocalBasisError(f"decoded PCM changed for {member.name}")
        result.append(VerifiedControlMember(
            member=member,
            strict_name=SINGLE_NAME_MAP[member.name],
            resolved_path=str(path),
            audio=audio,
        ))
    return tuple(result)


def ensemble_accompaniment(
    source: np.ndarray,
    residual_members: Mapping[str, np.ndarray],
    algorithm: str,
) -> np.ndarray:
    """Construct one ensemble by combining the implied vocal estimates."""

    source = np.asarray(source, dtype="float64")
    if source.ndim != 2 or source.shape[1] != 2 or not np.all(np.isfinite(source)):
        raise NoVocalBasisError("source must be finite stereo audio")
    if set(residual_members) != set(ENSEMBLE_PARENTS):
        raise NoVocalBasisError(
            f"ensemble residual member set changed: {sorted(residual_members)}"
        )
    accompaniments = []
    for name in ENSEMBLE_PARENTS:
        value = np.asarray(residual_members[name], dtype="float64")
        if value.shape != source.shape or not np.all(np.isfinite(value)):
            raise NoVocalBasisError(f"invalid ensemble member {name}")
        accompaniments.append(value)
    implied_vocals = np.stack([source - value for value in accompaniments])
    if algorithm == "median":
        vocal = np.median(implied_vocals, axis=0)
    elif algorithm == "uniform_mean":
        vocal = np.mean(implied_vocals, axis=0)
    elif algorithm == "stft_geometric_median":
        vocal = _stft_geometric_median(implied_vocals)
    else:
        raise NoVocalBasisError(f"unknown no-vocal ensemble algorithm {algorithm!r}")
    accompaniment = source - np.asarray(vocal, dtype="float64")
    if accompaniment.shape != source.shape or not np.all(np.isfinite(accompaniment)):
        raise NoVocalBasisError(f"invalid generated ensemble {algorithm}")
    return accompaniment.astype("float32")


def _recipe(
    *,
    work_id: str,
    source_pcm_sha256: str,
    source_frames: int,
    name: str,
    algorithm: str,
    parents: Sequence[VerifiedControlMember],
    code_commit: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if algorithm == "stft_geometric_median":
        config = {
            "n_fft": 2048,
            "hop_length": 512,
            "window": "hann",
            "center": True,
            "pad_mode": "constant",
            "whitening": "identity",
            "solver": {
                "name": "weiszfeld",
                "max_iter": 40,
                "tolerance": "0.0001",
            },
        }
    elif algorithm == "uniform_mean":
        config = {"weights_ppm": [333_333, 333_333, 333_334]}
    return {
        "schema": ENSEMBLE_RECIPE_SCHEMA,
        "work_id": work_id,
        "source_pcm_sha256": source_pcm_sha256,
        "source_grid": {
            "sample_rate_hz": 44_100,
            "channels": ["FL", "FR"],
            "frames": source_frames,
            "subtype": "FLOAT",
        },
        "candidate_name": name,
        "operation": {
            "type": "mixture_minus_vocal_ensemble",
            "algorithm": algorithm,
        },
        "parents": [{
            "candidate_name": parent.strict_name,
            "recipe_id": parent.member.recipe_id,
            "artifact_pcm_sha256": parent.member.artifact_pcm_sha256,
            "executed_bundle_hash": parent.member.executed_bundle_hash,
        } for parent in parents],
        "effective_config": config,
        "software": {"audio_extract_commit": code_commit},
    }


def _verify_generated(
    directory: Path,
    recipe: Mapping[str, Any],
    audio: np.ndarray,
) -> tuple[str, str]:
    if json.loads((directory / "recipe.json").read_text()) != recipe:
        raise NoVocalBasisError(f"generated recipe mismatch: {directory}")
    reopened, _ = _strict_audio(directory / "output.f32.wav", frames=len(audio))
    if not np.array_equal(reopened, np.asarray(audio, dtype="float32")):
        raise NoVocalBasisError(f"generated FLOAT mismatch: {directory}")
    artifact = _pcm(reopened)
    container = _sha_file(directory / "output.f32.wav")
    if (directory / "output.pcm.sha256").read_text().strip() != artifact:
        raise NoVocalBasisError(f"generated PCM sidecar mismatch: {directory}")
    if (directory / "container.sha256").read_text().strip() != container:
        raise NoVocalBasisError(f"generated container sidecar mismatch: {directory}")
    if not (directory / "COMPLETE").is_file():
        raise NoVocalBasisError(f"generated candidate is incomplete: {directory}")
    return artifact, container


def write_ensemble(
    output_root: Path,
    *,
    recipe: Mapping[str, Any],
    audio: np.ndarray,
) -> tuple[str, Path, str, str]:
    recipe_id = identity.blob_sha256(canon.canonicalize(recipe))
    directory = output_root / recipe["candidate_name"] / recipe_id
    if directory.is_dir():
        artifact, container = _verify_generated(directory, recipe, audio)
        return recipe_id, directory / "output.f32.wav", artifact, container
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=directory.parent))
    try:
        sf.write(temporary / "output.f32.wav", audio, 44_100, subtype="FLOAT")
        reopened, _ = _strict_audio(
            temporary / "output.f32.wav", frames=len(audio)
        )
        if not np.array_equal(reopened, np.asarray(audio, dtype="float32")):
            raise NoVocalBasisError("generated FLOAT reopen mismatch")
        (temporary / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "output.pcm.sha256").write_text(_pcm(reopened) + "\n")
        (temporary / "container.sha256").write_text(
            _sha_file(temporary / "output.f32.wav") + "\n"
        )
        (temporary / "COMPLETE").write_text("complete\n")
        try:
            os.replace(temporary, directory)
        except OSError:
            if not directory.is_dir():
                raise
        artifact, container = _verify_generated(directory, recipe, audio)
        return recipe_id, directory / "output.f32.wav", artifact, container
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_complete_basis(
    *,
    control_manifest: Path,
    source_audio: Path,
    output_root: Path,
    code_commit: str,
    work_id_override: str | None = None,
) -> tuple[CandidateDeclaration, ...]:
    manifest_work, source_pcm, members = load_control_manifest(control_manifest)
    work_id = work_id_override or manifest_work
    source, _ = _strict_audio(source_audio)
    if _pcm(source) != source_pcm:
        raise NoVocalBasisError(
            f"source identity {_pcm(source)} != control manifest {source_pcm}"
        )
    verified = verify_control_members(members, frames=len(source))
    result = []
    for item in verified:
        result.append(CandidateDeclaration(
            work_id=work_id,
            candidate_name=item.strict_name,
            recipe_id=item.member.recipe_id,
            path=item.resolved_path,
            artifact_pcm_sha256=item.member.artifact_pcm_sha256,
            container_sha256=item.member.container_sha256,
            frames=len(source),
            parent_recipe_ids=(item.member.parent_vocal_recipe_id,),
            executed_model_bundle_hashes=(item.member.executed_bundle_hash,),
        ))
    parent_map = {item.strict_name: item.audio for item in verified}
    ensemble_parents = tuple(
        next(item for item in verified if item.strict_name == name)
        for name in ENSEMBLE_PARENTS
    )
    specifications = (
        ("median_mdx_mel_bs", "median"),
        ("geomedian_mdx_mel_bs", "stft_geometric_median"),
        ("convex_fusion_uniform", "uniform_mean"),
    )
    for name, algorithm in specifications:
        audio = ensemble_accompaniment(
            source,
            {parent: parent_map[parent] for parent in ENSEMBLE_PARENTS},
            algorithm,
        )
        recipe = _recipe(
            work_id=work_id,
            source_pcm_sha256=source_pcm,
            source_frames=len(source),
            name=name,
            algorithm=algorithm,
            parents=ensemble_parents,
            code_commit=code_commit,
        )
        recipe_id, path, artifact, container = write_ensemble(
            output_root, recipe=recipe, audio=audio
        )
        result.append(CandidateDeclaration(
            work_id=work_id,
            candidate_name=name,
            recipe_id=recipe_id,
            path=str(path.resolve()),
            artifact_pcm_sha256=artifact,
            container_sha256=container,
            frames=len(source),
            parent_recipe_ids=tuple(
                item.member.recipe_id for item in ensemble_parents
            ),
            executed_model_bundle_hashes=tuple(sorted({
                item.member.executed_bundle_hash for item in ensemble_parents
            })),
        ))
    return tuple(result)


def write_strict_manifest(
    path: Path,
    rows: Sequence[CandidateDeclaration],
) -> None:
    payload = "".join(
        json.dumps({
            "schema": "audio-extract/oracle-routing-basis/v2",
            "work_id": row.work_id,
            "candidate_name": row.candidate_name,
            "recipe_id": row.recipe_id,
            "path": row.path,
            "artifact_pcm_sha256": row.artifact_pcm_sha256,
            "container_sha256": row.container_sha256,
            "frames": row.frames,
            "sample_rate_hz": row.sample_rate_hz,
            "channels": list(row.channels),
            "subtype": row.subtype,
            "parent_recipe_ids": list(row.parent_recipe_ids),
            "executed_model_bundle_hashes": list(
                row.executed_model_bundle_hashes
            ),
        }, sort_keys=True) + "\n"
        for row in rows
    )
    if path.exists() and path.read_text() != payload:
        raise NoVocalBasisError(
            f"refusing to replace differing strict no-vocal manifest: {path}"
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
