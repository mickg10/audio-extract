"""Exact source-lineage verification for certified oracle-routing candidates.

Artifact hashes prove what a candidate contains, but not which source produced
it.  This module independently reopens each candidate's immutable sibling
``recipe.json``, recomputes its recipe identity, extracts the recipe input PCM
and exact grid, and binds that input to the expected truth source for the scene.

Two recipe families are supported:

* the repository's canonical ``audio-extract/recipe/v2`` objects, whose input is
  ``input_pcm`` and whose identity is :func:`audio_extract.identity.recipe_id`;
* the immutable no-vocal ensemble recipe, whose input is
  ``source_pcm_sha256``/``source_grid`` and whose identity is the canonical blob
  SHA used by its writer.

No declaration-only source claim is accepted.  A binding candidate without a
readable, identity-matching recipe is invalid evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import os

import numpy as np
import soundfile as sf

from . import canon, identity, recipe as recipe_mod
from .oracle_routing_basis_v2 import VerifiedCandidate


LINEAGE_SCHEMA = "audio-extract/oracle-routing-source-lineage/v2"
NO_VOCAL_ENSEMBLE_SCHEMA = "audio-extract/oracle-no-vocal-ensemble/v2"
SOURCE_ROLES = frozenset({"voiced_mixture", "no_vocal_accompaniment"})


class SourceLineageError(ValueError):
    """A candidate recipe does not prove derivation from the expected source."""


@dataclass(frozen=True)
class ExpectedSource:
    work_id: str
    role: str
    path: str
    artifact_pcm_sha256: str
    container_sha256: str
    frames: int
    sample_rate_hz: int
    channels: tuple[str, ...]
    subtype: str

    def validate(self) -> None:
        if not self.work_id:
            raise SourceLineageError("expected source work_id is empty")
        if self.role not in SOURCE_ROLES:
            raise SourceLineageError(f"unknown expected source role: {self.role!r}")
        _sha(self.artifact_pcm_sha256, "expected source PCM")
        _sha(self.container_sha256, "expected source container")
        if self.frames <= 0 or self.sample_rate_hz <= 0:
            raise SourceLineageError("expected source grid is not positive")
        if self.channels != ("FL", "FR") or self.subtype != "FLOAT":
            raise SourceLineageError(
                "expected source must be stereo FL/FR FLOAT"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["channels"] = list(self.channels)
        return value


@dataclass(frozen=True)
class CandidateSourceLineage:
    work_id: str
    candidate_name: str
    recipe_id: str
    recipe_path: str
    recipe_container_sha256: str
    recipe_schema: str
    source_role: str
    source_pcm_sha256: str
    source_frames: int
    source_sample_rate_hz: int
    source_channels: tuple[str, ...]
    source_sample_format: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_channels"] = list(self.source_channels)
        return value


def _sha(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if not result.startswith("sha256:") or len(result) != 71:
        raise SourceLineageError(f"{label} must be sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise SourceLineageError(
            f"{label} must be sha256:<64 hex>"
        ) from exc
    return result


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _stable_json(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        if path.is_symlink():
            raise SourceLineageError(f"refusing symlinked recipe: {path}")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except SourceLineageError:
        raise
    except OSError as exc:
        raise SourceLineageError(f"cannot read recipe {path}: {exc}") from exc
    if (
        _signature(before) != _signature(after)
        or _signature(before) != _signature(current)
    ):
        raise SourceLineageError(
            f"recipe changed or path was replaced: {path}"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceLineageError(f"invalid recipe JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SourceLineageError(f"recipe must be a JSON object: {path}")
    return value, "sha256:" + hashlib.sha256(payload).hexdigest()


def _recipe_identity(recipe: Mapping[str, Any]) -> str:
    if recipe.get("schema") == recipe_mod.SCHEMA:
        try:
            return identity.recipe_id(dict(recipe))
        except (TypeError, ValueError) as exc:
            raise SourceLineageError(
                f"canonical recipe is invalid: {exc}"
            ) from exc
    if recipe.get("schema") == NO_VOCAL_ENSEMBLE_SCHEMA:
        try:
            return identity.blob_sha256(canon.canonicalize(dict(recipe)))
        except (TypeError, ValueError) as exc:
            raise SourceLineageError(
                f"no-vocal ensemble recipe is invalid: {exc}"
            ) from exc
    raise SourceLineageError(
        f"unsupported candidate recipe schema: {recipe.get('schema')!r}"
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceLineageError(f"{label} must be a positive integer")
    return value


def _channels(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SourceLineageError(f"{label} must be an array")
    result = tuple(str(item) for item in value)
    if result != ("FL", "FR"):
        raise SourceLineageError(f"{label} must equal ['FL','FR']")
    return result


def _recipe_source(
    recipe: Mapping[str, Any],
) -> tuple[str, int, int, tuple[str, ...], str]:
    schema = recipe.get("schema")
    if schema == recipe_mod.SCHEMA:
        value = recipe.get("input_pcm")
        if not isinstance(value, Mapping):
            raise SourceLineageError("canonical recipe lacks input_pcm")
        return (
            _sha(value.get("sha256"), "recipe input PCM"),
            _positive_int(value.get("frames"), "recipe input frames"),
            _positive_int(
                value.get("sample_rate_hz"), "recipe input sample rate"
            ),
            _channels(
                value.get("channel_layout"),
                "recipe input channel layout",
            ),
            str(value.get("sample_format") or ""),
        )
    if schema == NO_VOCAL_ENSEMBLE_SCHEMA:
        grid = recipe.get("source_grid")
        if not isinstance(grid, Mapping):
            raise SourceLineageError(
                "no-vocal ensemble recipe lacks source_grid"
            )
        if grid.get("subtype") != "FLOAT":
            raise SourceLineageError(
                "no-vocal ensemble source subtype must be FLOAT"
            )
        return (
            _sha(
                recipe.get("source_pcm_sha256"),
                "ensemble source PCM",
            ),
            _positive_int(grid.get("frames"), "ensemble source frames"),
            _positive_int(
                grid.get("sample_rate_hz"),
                "ensemble source sample rate",
            ),
            _channels(grid.get("channels"), "ensemble source channels"),
            "float32-le-interleaved",
        )
    raise SourceLineageError(f"unsupported recipe schema: {schema!r}")


def expected_source_from_audio(
    path: Path,
    *,
    work_id: str,
    role: str,
) -> ExpectedSource:
    """Reopen one exact truth source and freeze its decoded/container identity."""

    if role not in SOURCE_ROLES:
        raise SourceLineageError(f"unknown expected source role: {role!r}")
    if path.is_symlink():
        raise SourceLineageError(f"refusing symlinked truth source: {path}")
    resolved = path.resolve(strict=True)
    info = sf.info(resolved)
    if (
        info.frames <= 0
        or info.samplerate != 44_100
        or info.channels != 2
        or info.subtype != "FLOAT"
    ):
        raise SourceLineageError(
            f"truth source is not exact stereo FLOAT: {resolved}: {info}"
        )
    audio, sample_rate = sf.read(
        resolved, dtype="float32", always_2d=True
    )
    if (
        sample_rate != 44_100
        or audio.shape != (info.frames, 2)
        or not np.all(np.isfinite(audio))
    ):
        raise SourceLineageError(f"truth source decode failed: {resolved}")
    result = ExpectedSource(
        work_id=str(work_id),
        role=role,
        path=str(resolved),
        artifact_pcm_sha256=identity.artifact_pcm_sha256(
            audio, 44_100, ["FL", "FR"], len(audio)
        ),
        container_sha256=_sha_file(resolved),
        frames=len(audio),
        sample_rate_hz=44_100,
        channels=("FL", "FR"),
        subtype="FLOAT",
    )
    result.validate()
    return result


def verify_candidate_source_lineage(
    candidate: VerifiedCandidate,
    expected: ExpectedSource,
) -> CandidateSourceLineage:
    """Prove one candidate recipe names the exact expected source PCM/grid."""

    expected.validate()
    declaration = candidate.declaration
    if declaration.work_id != expected.work_id:
        raise SourceLineageError(
            f"candidate/expected work mismatch: {declaration.work_id} != "
            f"{expected.work_id}"
        )
    if (
        declaration.frames != expected.frames
        or declaration.sample_rate_hz != expected.sample_rate_hz
        or declaration.channels != expected.channels
        or declaration.subtype != expected.subtype
    ):
        raise SourceLineageError(
            f"candidate/source output grid mismatch for "
            f"{declaration.candidate_name}"
        )

    output_path = Path(candidate.resolved_path)
    recipe_path = output_path.parent / "recipe.json"
    if not recipe_path.is_file():
        raise SourceLineageError(
            f"candidate lacks immutable sibling recipe: {recipe_path}"
        )
    recipe, recipe_container = _stable_json(recipe_path)
    computed_recipe_id = _recipe_identity(recipe)
    if computed_recipe_id != declaration.recipe_id:
        raise SourceLineageError(
            f"candidate recipe identity mismatch for "
            f"{declaration.candidate_name}: {computed_recipe_id} != "
            f"{declaration.recipe_id}"
        )
    recipe_work = recipe.get("work_id")
    if recipe_work is not None and str(recipe_work) != declaration.work_id:
        raise SourceLineageError(
            f"recipe work mismatch: {recipe_work!r} != "
            f"{declaration.work_id!r}"
        )
    source_pcm, frames, sample_rate, channels, sample_format = (
        _recipe_source(recipe)
    )
    if source_pcm != expected.artifact_pcm_sha256:
        raise SourceLineageError(
            f"candidate source PCM mismatch for {declaration.candidate_name}: "
            f"{source_pcm} != {expected.artifact_pcm_sha256}"
        )
    if (
        frames != expected.frames
        or sample_rate != expected.sample_rate_hz
        or channels != expected.channels
    ):
        raise SourceLineageError(
            f"candidate recipe source grid mismatch for "
            f"{declaration.candidate_name}"
        )
    if sample_format != "float32-le-interleaved":
        raise SourceLineageError(
            f"candidate source sample format must be float32-le-interleaved, "
            f"got {sample_format!r}"
        )
    return CandidateSourceLineage(
        work_id=declaration.work_id,
        candidate_name=declaration.candidate_name,
        recipe_id=declaration.recipe_id,
        recipe_path=str(recipe_path.resolve(strict=True)),
        recipe_container_sha256=recipe_container,
        recipe_schema=str(recipe.get("schema")),
        source_role=expected.role,
        source_pcm_sha256=source_pcm,
        source_frames=frames,
        source_sample_rate_hz=sample_rate,
        source_channels=channels,
        source_sample_format=sample_format,
    )


def audit_basis_source_lineage(
    candidates: Sequence[VerifiedCandidate],
    expected_sources: Mapping[str, ExpectedSource],
    *,
    role: str,
) -> dict[str, Any]:
    """Verify every pre-dedup declaration against one complete expected source set."""

    if role not in SOURCE_ROLES:
        raise SourceLineageError(f"unknown source role: {role!r}")
    if not candidates:
        raise SourceLineageError("source-lineage candidate set is empty")
    if not expected_sources:
        raise SourceLineageError("expected source set is empty")
    expected = dict(expected_sources)
    if any(source.role != role for source in expected.values()):
        raise SourceLineageError("expected source role differs within audit")
    candidate_works = {row.declaration.work_id for row in candidates}
    if candidate_works != set(expected):
        raise SourceLineageError(
            f"candidate/expected work sets differ: "
            f"{sorted(candidate_works)} != {sorted(expected)}"
        )

    rows = tuple(
        verify_candidate_source_lineage(
            candidate, expected[candidate.declaration.work_id]
        )
        for candidate in candidates
    )
    if len({
        (row.work_id, row.candidate_name, row.recipe_id) for row in rows
    }) != len(rows):
        raise SourceLineageError("source-lineage audit contains duplicate rows")
    return {
        "schema": LINEAGE_SCHEMA,
        "status": "pass",
        "source_role": role,
        "works": sorted(expected),
        "expected_sources": {
            work: expected[work].to_dict() for work in sorted(expected)
        },
        "candidate_count": len(rows),
        "candidates": [row.to_dict() for row in rows],
    }
