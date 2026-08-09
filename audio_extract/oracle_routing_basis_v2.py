"""Verified and deduplicated candidate bases for certified oracle routing.

The certified routing experiment compares immutable accompaniment candidates on
one exact sample grid.  This module reopens every published FLOAT file, verifies
its declared identities, collapses decoded-PCM duplicates, and preserves all
logical aliases (especially the current median champion).

No resampling, truncation, alignment, normalization, or rewriting occurs here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import socket

import numpy as np
import soundfile as sf

from . import identity

BASIS_SCHEMA = "audio-extract/oracle-routing-basis/v2"
DEDUP_SCHEMA = "audio-extract/oracle-routing-basis-dedup/v2"

DEFAULT_CANDIDATE_ORDER = (
    "median_mdx_mel_bs",
    "geomedian_mdx_mel_bs",
    "convex_fusion_uniform",
    "residual_mdx23c",
    "residual_melband",
    "residual_bs_roformer",
    "htdemucs_04573f0d",
    "htdemucs_955717e8",
)


class BasisValidationError(ValueError):
    """A declared basis row is not the exact published artifact it claims."""


@dataclass(frozen=True)
class CandidateDeclaration:
    work_id: str
    candidate_name: str
    recipe_id: str
    path: str
    artifact_pcm_sha256: str
    container_sha256: str
    frames: int
    sample_rate_hz: int = 44_100
    channels: tuple[str, ...] = ("FL", "FR")
    subtype: str = "FLOAT"
    parent_recipe_ids: tuple[str, ...] = ()
    executed_model_bundle_hashes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateDeclaration":
        """Normalize v2 rows and the audited `candidates-v2` row shape."""

        work_id = _nonempty(value.get("work_id"), "work_id")
        candidate_name = _nonempty(
            value.get("candidate_name") or value.get("candidate"),
            "candidate_name",
        )
        recipe_id = _sha_id(value.get("recipe_id"), "recipe_id")
        path = _nonempty(value.get("path"), "path")
        artifact = _sha_id(
            value.get("artifact_pcm_sha256"), "artifact_pcm_sha256"
        )
        container = _sha_id(
            value.get("container_sha256"), "container_sha256"
        )
        frames = int(value.get("frames", 0))
        sample_rate = int(
            value.get("sample_rate_hz", value.get("sr_hz", 0))
        )
        channels = tuple(value.get("channels") or ())
        subtype = _nonempty(value.get("subtype"), "subtype")
        parents = tuple(
            _sha_id(item, "parent_recipe_id")
            for item in (
                value.get("parent_recipe_ids")
                or value.get("parent_candidate_recipe_ids")
                or ()
            )
        )
        bundles = tuple(
            _sha_id(item, "executed_model_bundle_hash")
            for item in (value.get("executed_model_bundle_hashes") or ())
        )
        if frames <= 0 or sample_rate <= 0:
            raise BasisValidationError("frames and sample rate must be positive")
        if channels != ("FL", "FR"):
            raise BasisValidationError(
                f"candidate channels must be ('FL','FR'), got {channels}"
            )
        if subtype != "FLOAT":
            raise BasisValidationError(
                f"candidate subtype must be FLOAT, got {subtype!r}"
            )
        return cls(
            work_id=work_id,
            candidate_name=candidate_name,
            recipe_id=recipe_id,
            path=path,
            artifact_pcm_sha256=artifact,
            container_sha256=container,
            frames=frames,
            sample_rate_hz=sample_rate,
            channels=channels,
            subtype=subtype,
            parent_recipe_ids=parents,
            executed_model_bundle_hashes=bundles,
        )


@dataclass(frozen=True)
class VerifiedCandidate:
    declaration: CandidateDeclaration
    resolved_path: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self.declaration)
        result["channels"] = list(result["channels"])
        result["parent_recipe_ids"] = list(result["parent_recipe_ids"])
        result["executed_model_bundle_hashes"] = list(
            result["executed_model_bundle_hashes"]
        )
        result["resolved_path"] = self.resolved_path
        result["schema"] = BASIS_SCHEMA
        return result


@dataclass(frozen=True)
class DeduplicatedCandidate:
    work_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    canonical_recipe_id: str
    alias_recipe_ids: tuple[str, ...]
    resolved_path: str
    artifact_pcm_sha256: str
    container_sha256: str
    frames: int
    sample_rate_hz: int
    channels: tuple[str, ...]
    subtype: str
    parent_recipe_ids: tuple[str, ...]
    executed_model_bundle_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "aliases", "alias_recipe_ids", "channels", "parent_recipe_ids",
            "executed_model_bundle_hashes",
        ):
            result[name] = list(result[name])
        result["schema"] = DEDUP_SCHEMA
        return result


def _nonempty(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise BasisValidationError(f"{name} must be non-empty")
    return result


def _sha_id(value: Any, name: str) -> str:
    result = _nonempty(value, name).lower()
    if not result.startswith("sha256:") or len(result) != 71:
        raise BasisValidationError(f"{name} must be sha256:<64 hex>")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise BasisValidationError(f"{name} must be sha256:<64 hex>") from exc
    return result


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def local_host_aliases(extra: Iterable[str] = ()) -> frozenset[str]:
    values = {
        socket.gethostname(),
        socket.getfqdn(),
        "localhost",
        "127.0.0.1",
    }
    values.update(str(item).strip() for item in extra if str(item).strip())
    return frozenset(values)


def resolve_declared_path(
    value: str,
    *,
    allowed_host_aliases: Iterable[str] = (),
) -> Path:
    """Resolve a local path or an explicitly local `host:/absolute/path` value."""

    direct = Path(value)
    if direct.is_file():
        return direct.resolve()
    if ":" not in value:
        raise BasisValidationError(f"candidate file does not exist: {value}")
    host, suffix = value.split(":", 1)
    allowed = local_host_aliases(allowed_host_aliases)
    if host not in allowed:
        raise BasisValidationError(
            f"path host {host!r} is not this execution host; allowed={sorted(allowed)}"
        )
    candidate = Path(suffix)
    if not candidate.is_absolute() or not candidate.is_file():
        raise BasisValidationError(f"candidate file does not exist: {value}")
    return candidate.resolve()


def verify_candidate(
    declaration: CandidateDeclaration,
    *,
    allowed_host_aliases: Iterable[str] = (),
) -> VerifiedCandidate:
    """Reopen one declaration and verify its exact grid and identities."""

    path = resolve_declared_path(
        declaration.path, allowed_host_aliases=allowed_host_aliases
    )
    info = sf.info(path)
    actual_grid = (
        int(info.frames), int(info.samplerate), int(info.channels), info.subtype
    )
    expected_grid = (
        declaration.frames,
        declaration.sample_rate_hz,
        len(declaration.channels),
        declaration.subtype,
    )
    if actual_grid != expected_grid:
        raise BasisValidationError(
            f"published grid mismatch for {declaration.candidate_name}: "
            f"{actual_grid} != {expected_grid}"
        )
    container = _sha_file(path)
    if container != declaration.container_sha256:
        raise BasisValidationError(
            f"container identity mismatch for {declaration.candidate_name}: "
            f"{container} != {declaration.container_sha256}"
        )
    audio, sample_rate = sf.read(
        path, dtype="float32", always_2d=True
    )
    if (
        int(sample_rate) != declaration.sample_rate_hz
        or audio.shape != (declaration.frames, 2)
        or not np.all(np.isfinite(audio))
    ):
        raise BasisValidationError(
            f"decoded audio is invalid for {declaration.candidate_name}"
        )
    artifact = identity.artifact_pcm_sha256(
        audio,
        declaration.sample_rate_hz,
        list(declaration.channels),
        declaration.frames,
    )
    if artifact != declaration.artifact_pcm_sha256:
        raise BasisValidationError(
            f"decoded PCM identity mismatch for {declaration.candidate_name}: "
            f"{artifact} != {declaration.artifact_pcm_sha256}"
        )
    return VerifiedCandidate(declaration, str(path))


def load_declarations(paths: Sequence[Path]) -> tuple[CandidateDeclaration, ...]:
    """Load one or more append-only JSONL candidate manifests."""

    result: list[CandidateDeclaration] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                declaration = CandidateDeclaration.from_mapping(value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise BasisValidationError(
                    f"invalid basis row {path}:{line_number}: {exc}"
                ) from exc
            key = (
                declaration.work_id,
                declaration.candidate_name,
                declaration.recipe_id,
            )
            if key in seen:
                raise BasisValidationError(f"duplicate candidate declaration: {key}")
            seen.add(key)
            result.append(declaration)
    if not result:
        raise BasisValidationError("basis manifests contain no candidate rows")
    return tuple(result)


def verify_basis(
    declarations: Sequence[CandidateDeclaration],
    *,
    allowed_host_aliases: Iterable[str] = (),
) -> tuple[VerifiedCandidate, ...]:
    result = tuple(
        verify_candidate(
            declaration, allowed_host_aliases=allowed_host_aliases
        )
        for declaration in declarations
    )
    per_work_grid: dict[str, tuple[int, int, tuple[str, ...], str]] = {}
    for row in result:
        declaration = row.declaration
        grid = (
            declaration.frames,
            declaration.sample_rate_hz,
            declaration.channels,
            declaration.subtype,
        )
        previous = per_work_grid.setdefault(declaration.work_id, grid)
        if previous != grid:
            raise BasisValidationError(
                f"candidate grids differ within work {declaration.work_id}: "
                f"{previous} != {grid}"
            )
    return result


def deduplicate_basis(
    verified: Sequence[VerifiedCandidate],
    *,
    candidate_order: Sequence[str] = DEFAULT_CANDIDATE_ORDER,
    required_aliases: Sequence[str] = (),
) -> tuple[DeduplicatedCandidate, ...]:
    """Collapse decoded-PCM duplicates while preserving every logical alias."""

    if not verified:
        raise BasisValidationError("verified basis is empty")
    priority = {name: index for index, name in enumerate(candidate_order)}
    by_work: dict[str, list[VerifiedCandidate]] = {}
    for row in verified:
        by_work.setdefault(row.declaration.work_id, []).append(row)
    result: list[DeduplicatedCandidate] = []
    for work_id in sorted(by_work):
        rows = by_work[work_id]
        aliases_present = {row.declaration.candidate_name for row in rows}
        missing = set(required_aliases) - aliases_present
        if missing:
            raise BasisValidationError(
                f"work {work_id} lacks required candidate aliases: {sorted(missing)}"
            )
        groups: dict[str, list[VerifiedCandidate]] = {}
        for row in rows:
            groups.setdefault(
                row.declaration.artifact_pcm_sha256, []
            ).append(row)
        for artifact, duplicate_rows in groups.items():
            ordered = sorted(
                duplicate_rows,
                key=lambda row: (
                    priority.get(row.declaration.candidate_name, len(priority)),
                    row.declaration.candidate_name,
                    row.declaration.recipe_id,
                ),
            )
            canonical = ordered[0]
            declarations = [row.declaration for row in ordered]
            result.append(DeduplicatedCandidate(
                work_id=work_id,
                canonical_name=canonical.declaration.candidate_name,
                aliases=tuple(sorted({row.candidate_name for row in declarations})),
                canonical_recipe_id=canonical.declaration.recipe_id,
                alias_recipe_ids=tuple(sorted({row.recipe_id for row in declarations})),
                resolved_path=canonical.resolved_path,
                artifact_pcm_sha256=artifact,
                container_sha256=canonical.declaration.container_sha256,
                frames=canonical.declaration.frames,
                sample_rate_hz=canonical.declaration.sample_rate_hz,
                channels=canonical.declaration.channels,
                subtype=canonical.declaration.subtype,
                parent_recipe_ids=tuple(sorted({
                    parent for row in declarations for parent in row.parent_recipe_ids
                })),
                executed_model_bundle_hashes=tuple(sorted({
                    bundle for row in declarations
                    for bundle in row.executed_model_bundle_hashes
                })),
            ))
    return tuple(sorted(
        result,
        key=lambda row: (
            row.work_id,
            priority.get(row.canonical_name, len(priority)),
            row.canonical_name,
        ),
    ))


def basis_report(
    verified: Sequence[VerifiedCandidate],
    deduplicated: Sequence[DeduplicatedCandidate],
) -> dict[str, Any]:
    works = sorted({row.declaration.work_id for row in verified})
    return {
        "schema": "audio-extract/oracle-routing-basis-audit/v2",
        "status": "pass",
        "works": works,
        "declared_rows": len(verified),
        "unique_decoded_candidates": len(deduplicated),
        "duplicates_collapsed": len(verified) - len(deduplicated),
        "per_work": {
            work: {
                "declared_rows": sum(
                    row.declaration.work_id == work for row in verified
                ),
                "unique_decoded_candidates": sum(
                    row.work_id == work for row in deduplicated
                ),
                "candidates": [
                    row.to_dict() for row in deduplicated if row.work_id == work
                ],
            }
            for work in works
        },
    }


def write_jsonl(path: Path, rows: Sequence[DeduplicatedCandidate]) -> None:
    payload = "".join(
        json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows
    )
    if path.exists() and path.read_text() != payload:
        raise BasisValidationError(f"refusing to replace differing basis: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
