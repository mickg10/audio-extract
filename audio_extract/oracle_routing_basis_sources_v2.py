"""Source adapters for assembling the certified routing basis v2.

The audited `datasets/candidates-v2.jsonl` predates the strict v2 basis schema and
stores some bundle identities as bare 64-hex strings. This adapter performs the
one explicit normalization into `sha256:<hex>` identities before constructing
strict `CandidateDeclaration` objects. The strict basis module itself remains
unambiguous and rejects malformed v2 rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json

from .oracle_routing_basis_v2 import (
    BasisValidationError,
    CandidateDeclaration,
    DeduplicatedCandidate,
    VerifiedCandidate,
    deduplicate_basis,
    verify_basis,
)


def normalize_sha256(value: Any, name: str) -> str:
    result = str(value or "").strip().lower()
    if result.startswith("sha256:"):
        payload = result[7:]
    else:
        payload = result
    if len(payload) != 64:
        raise BasisValidationError(f"{name} must contain 64 hexadecimal digits")
    try:
        int(payload, 16)
    except ValueError as exc:
        raise BasisValidationError(
            f"{name} must contain 64 hexadecimal digits"
        ) from exc
    return "sha256:" + payload


def normalize_audited_candidates_v2_row(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Map one audited baseline row into the strict basis-v2 declaration shape."""

    required = (
        "work_id", "candidate", "recipe_id", "path",
        "artifact_pcm_sha256", "container_sha256", "frames",
        "sr_hz", "channels", "subtype",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise BasisValidationError(
            f"audited candidates-v2 row lacks fields: {missing}"
        )
    return {
        "work_id": value["work_id"],
        "candidate_name": value["candidate"],
        "recipe_id": normalize_sha256(value["recipe_id"], "recipe_id"),
        "path": value["path"],
        "artifact_pcm_sha256": normalize_sha256(
            value["artifact_pcm_sha256"], "artifact_pcm_sha256"
        ),
        "container_sha256": normalize_sha256(
            value["container_sha256"], "container_sha256"
        ),
        "frames": value["frames"],
        "sample_rate_hz": value["sr_hz"],
        "channels": value["channels"],
        "subtype": value["subtype"],
        "parent_recipe_ids": [
            normalize_sha256(item, "parent_candidate_recipe_id")
            for item in value.get("parent_candidate_recipe_ids", ())
        ],
        "executed_model_bundle_hashes": [
            normalize_sha256(item, "executed_model_bundle_hash")
            for item in value.get("executed_model_bundle_hashes", ())
        ],
    }


def load_audited_candidates_v2(path: Path) -> tuple[CandidateDeclaration, ...]:
    result: list[CandidateDeclaration] = []
    seen: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            normalized = normalize_audited_candidates_v2_row(raw)
            declaration = CandidateDeclaration.from_mapping(normalized)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise BasisValidationError(
                f"invalid audited row {path}:{line_number}: {exc}"
            ) from exc
        key = (
            declaration.work_id,
            declaration.candidate_name,
            declaration.recipe_id,
        )
        if key in seen:
            raise BasisValidationError(f"duplicate audited candidate row: {key}")
        seen.add(key)
        result.append(declaration)
    if not result:
        raise BasisValidationError(f"audited manifest is empty: {path}")
    return tuple(result)


def load_strict_basis_v2(path: Path) -> tuple[CandidateDeclaration, ...]:
    result: list[CandidateDeclaration] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            result.append(CandidateDeclaration.from_mapping(value))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise BasisValidationError(
                f"invalid strict basis row {path}:{line_number}: {exc}"
            ) from exc
    if not result:
        raise BasisValidationError(f"strict basis manifest is empty: {path}")
    return tuple(result)


def assemble_basis(
    *,
    audited_candidate_manifests: Sequence[Path] = (),
    strict_basis_manifests: Sequence[Path] = (),
    allowed_host_aliases: Iterable[str] = (),
    required_aliases: Sequence[str] = (),
) -> tuple[tuple[VerifiedCandidate, ...], tuple[DeduplicatedCandidate, ...]]:
    """Load, verify, and decoded-PCM-deduplicate all declared basis sources."""

    declarations: list[CandidateDeclaration] = []
    for path in audited_candidate_manifests:
        declarations.extend(load_audited_candidates_v2(path))
    for path in strict_basis_manifests:
        declarations.extend(load_strict_basis_v2(path))
    if not declarations:
        raise BasisValidationError("no routing basis source manifests were supplied")
    keys: set[tuple[str, str, str]] = set()
    for declaration in declarations:
        key = (
            declaration.work_id,
            declaration.candidate_name,
            declaration.recipe_id,
        )
        if key in keys:
            raise BasisValidationError(f"duplicate declaration across sources: {key}")
        keys.add(key)
    verified = verify_basis(
        tuple(declarations), allowed_host_aliases=allowed_host_aliases
    )
    deduplicated = deduplicate_basis(
        verified, required_aliases=required_aliases
    )
    return verified, deduplicated
