"""Closed-world source-family certificates for grouped D0/R0 datasets.

A string-valued group name is not sufficient to prevent derivative leakage.  This
module makes ``source_family_sha256`` the semantic digest of a content-bearing,
closed-world derivation certificate and verifies exact dataset rows against that
certificate registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import re

from .counterfactual_risk_dataset_contract_v1 import (
    CounterfactualRiskDatasetError,
    DatasetManifest,
)

CERTIFICATE_SCHEMA = "audio-extract/counterfactual-source-family/v1"
REGISTRY_SCHEMA = "audio-extract/counterfactual-source-family-registry/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ROLES = {
    "mixture_root",
    "accompaniment_truth",
    "vocal_truth",
    "candidate",
    "feature_source",
    "query_source",
}


class SourceFamilyError(CounterfactualRiskDatasetError):
    """A source-family certificate, registry, or row binding is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise SourceFamilyError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise SourceFamilyError(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _ordered_shas(values: Sequence[str], name: str, *, allow_empty: bool) -> tuple[str, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise SourceFamilyError(f"{name} must be non-empty")
    for index, value in enumerate(result):
        _sha(value, f"{name}[{index}]")
    if len(set(result)) != len(result):
        raise SourceFamilyError(f"{name} contains duplicates")
    if result != tuple(sorted(result)):
        raise SourceFamilyError(f"{name} must be in canonical lexical order")
    return result


@dataclass(frozen=True)
class SourceArtifact:
    role: str
    recipe_ids: tuple[str, ...]
    artifact_pcm_sha256: str
    parent_pcm_sha256s: tuple[str, ...] = ()

    def validate(self, *, root: bool = False) -> None:
        if self.role not in _ROLES:
            raise SourceFamilyError(f"unknown source-artifact role {self.role!r}")
        _ordered_shas(self.recipe_ids, "recipe_ids", allow_empty=False)
        _sha(self.artifact_pcm_sha256, "artifact_pcm_sha256")
        parents = _ordered_shas(
            self.parent_pcm_sha256s,
            "parent_pcm_sha256s",
            allow_empty=True,
        )
        if root and parents:
            raise SourceFamilyError("root artifacts cannot have parents")
        if not root and self.role in {
            "candidate",
            "feature_source",
            "query_source",
        } and not parents:
            raise SourceFamilyError(
                f"derived artifact role {self.role!r} requires at least one parent"
            )
        if self.artifact_pcm_sha256 in parents:
            raise SourceFamilyError("an artifact cannot directly parent itself")

    def identity_dict(self) -> dict[str, Any]:
        self.validate(root=self.role in {
            "mixture_root",
            "accompaniment_truth",
            "vocal_truth",
        })
        return {
            "role": self.role,
            "recipe_ids": list(self.recipe_ids),
            "artifact_pcm_sha256": self.artifact_pcm_sha256,
            "parent_pcm_sha256s": list(self.parent_pcm_sha256s),
        }

    def contains_candidate(self, recipe_id: str, artifact_pcm_sha256: str) -> bool:
        return (
            self.role == "candidate"
            and artifact_pcm_sha256 == self.artifact_pcm_sha256
            and recipe_id in self.recipe_ids
        )


@dataclass(frozen=True)
class SourceFamilyCertificate:
    mixture_root: SourceArtifact
    accompaniment_truth: SourceArtifact
    vocal_truth: SourceArtifact
    derived_artifacts: tuple[SourceArtifact, ...]
    spectral_grid_sha256s: tuple[str, ...]
    query_condition_sha256s: tuple[str, ...]
    discovery_manifest_sha256: str
    derivation_policy_sha256: str
    verifier_commit: str
    closed_world: bool = True
    status: str = "passed"

    def validate(self) -> None:
        if self.status != "passed":
            raise SourceFamilyError("source-family certificate status must be passed")
        if self.closed_world is not True:
            raise SourceFamilyError(
                "source-family certificate must declare a closed-world inventory"
            )
        roots = (
            (self.mixture_root, "mixture_root"),
            (self.accompaniment_truth, "accompaniment_truth"),
            (self.vocal_truth, "vocal_truth"),
        )
        for artifact, role in roots:
            if artifact.role != role:
                raise SourceFamilyError(
                    f"certificate {role} carries role {artifact.role!r}"
                )
            artifact.validate(root=True)
        _ordered_shas(
            self.spectral_grid_sha256s,
            "spectral_grid_sha256s",
            allow_empty=False,
        )
        _ordered_shas(
            self.query_condition_sha256s,
            "query_condition_sha256s",
            allow_empty=False,
        )
        _sha(self.discovery_manifest_sha256, "discovery_manifest_sha256")
        _sha(self.derivation_policy_sha256, "derivation_policy_sha256")
        _git(self.verifier_commit, "verifier_commit")

        for artifact in self.derived_artifacts:
            if artifact.role in {
                "mixture_root",
                "accompaniment_truth",
                "vocal_truth",
            }:
                raise SourceFamilyError(
                    "root roles cannot appear in derived_artifacts"
                )
            artifact.validate(root=False)
        pairs = tuple(
            (recipe_id, artifact.artifact_pcm_sha256)
            for artifact in self.derived_artifacts
            for recipe_id in artifact.recipe_ids
        )
        if len(set(pairs)) != len(pairs):
            raise SourceFamilyError(
                "derived recipe/PCM identities are duplicated"
            )
        artifact_hashes = tuple(
            artifact.artifact_pcm_sha256 for artifact in self.derived_artifacts
        )
        if len(set(artifact_hashes)) != len(artifact_hashes):
            raise SourceFamilyError(
                "derived artifacts must have unique decoded-PCM identities"
            )

        known = {
            self.mixture_root.artifact_pcm_sha256,
            self.accompaniment_truth.artifact_pcm_sha256,
            self.vocal_truth.artifact_pcm_sha256,
            *artifact_hashes,
        }
        for artifact in self.derived_artifacts:
            unknown = set(artifact.parent_pcm_sha256s) - known
            if unknown:
                raise SourceFamilyError(
                    f"derived artifact has unknown parents: {sorted(unknown)}"
                )

        by_hash = {
            artifact.artifact_pcm_sha256: artifact
            for artifact in self.derived_artifacts
        }

        def reaches_mixture(artifact: SourceArtifact) -> bool:
            pending = list(artifact.parent_pcm_sha256s)
            seen = {artifact.artifact_pcm_sha256}
            while pending:
                current = pending.pop()
                if current == self.mixture_root.artifact_pcm_sha256:
                    return True
                if current in seen:
                    raise SourceFamilyError("source-family derivation graph contains a cycle")
                seen.add(current)
                parent = by_hash.get(current)
                if parent is not None:
                    pending.extend(parent.parent_pcm_sha256s)
            return False

        for artifact in self.derived_artifacts:
            if artifact.role in {"candidate", "feature_source"} and not reaches_mixture(
                artifact
            ):
                raise SourceFamilyError(
                    f"{artifact.role} does not trace to the exact mixture root"
                )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CERTIFICATE_SCHEMA,
            "status": self.status,
            "closed_world": self.closed_world,
            "mixture_root": self.mixture_root.identity_dict(),
            "accompaniment_truth": self.accompaniment_truth.identity_dict(),
            "vocal_truth": self.vocal_truth.identity_dict(),
            "derived_artifacts": [
                artifact.identity_dict() for artifact in self.derived_artifacts
            ],
            "spectral_grid_sha256s": list(self.spectral_grid_sha256s),
            "query_condition_sha256s": list(self.query_condition_sha256s),
            "discovery_manifest_sha256": self.discovery_manifest_sha256,
            "derivation_policy_sha256": self.derivation_policy_sha256,
            "verifier_commit": self.verifier_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())

    def contains_candidate(self, recipe_id: str, artifact_pcm_sha256: str) -> bool:
        self.validate()
        return any(
            artifact.contains_candidate(recipe_id, artifact_pcm_sha256)
            for artifact in self.derived_artifacts
        )


@dataclass(frozen=True)
class SourceFamilyRegistry:
    certificates: tuple[SourceFamilyCertificate, ...]
    source_commit: str

    @classmethod
    def build(
        cls,
        certificates: Sequence[SourceFamilyCertificate],
        *,
        source_commit: str,
    ) -> "SourceFamilyRegistry":
        ordered = tuple(sorted(tuple(certificates), key=lambda value: value.sha256))
        result = cls(ordered, source_commit)
        result.validate()
        return result

    def validate(self) -> None:
        _git(self.source_commit, "registry source_commit")
        if not self.certificates:
            raise SourceFamilyError("source-family registry is empty")
        for certificate in self.certificates:
            certificate.validate()
        identities = tuple(certificate.sha256 for certificate in self.certificates)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(
            identities
        ):
            raise SourceFamilyError(
                "source-family certificates must be unique and canonically ordered"
            )
        mixture_roots = tuple(
            certificate.mixture_root.artifact_pcm_sha256
            for certificate in self.certificates
        )
        if len(set(mixture_roots)) != len(mixture_roots):
            raise SourceFamilyError(
                "one exact mixture root appears in multiple source families"
            )
        candidate_pairs = []
        for certificate in self.certificates:
            for artifact in certificate.derived_artifacts:
                if artifact.role != "candidate":
                    continue
                candidate_pairs.extend(
                    (recipe_id, artifact.artifact_pcm_sha256)
                    for recipe_id in artifact.recipe_ids
                )
        if len(set(candidate_pairs)) != len(candidate_pairs):
            raise SourceFamilyError(
                "one candidate recipe/PCM identity appears in multiple families"
            )

    def by_sha256(self) -> dict[str, SourceFamilyCertificate]:
        self.validate()
        return {certificate.sha256: certificate for certificate in self.certificates}

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": REGISTRY_SCHEMA,
            "source_commit": self.source_commit,
            "certificate_sha256s": [
                certificate.sha256 for certificate in self.certificates
            ],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


def validate_dataset_family_registry(
    dataset: DatasetManifest,
    registry: SourceFamilyRegistry,
) -> None:
    """Verify every exact row against a content-bearing source family."""

    dataset.validate()
    registry.validate()
    certificates = registry.by_sha256()
    for row in dataset.rows:
        family_sha = row.group_family.source_family_sha256
        certificate = certificates.get(family_sha)
        if certificate is None:
            raise SourceFamilyError(
                "dataset row names an unregistered source-family certificate"
            )
        if row.mixture_pcm_sha256 != certificate.mixture_root.artifact_pcm_sha256:
            raise SourceFamilyError("dataset mixture differs from family root")
        if row.accompaniment_truth_pcm_sha256 != (
            certificate.accompaniment_truth.artifact_pcm_sha256
        ):
            raise SourceFamilyError(
                "dataset accompaniment truth differs from family certificate"
            )
        if row.vocal_truth_pcm_sha256 != certificate.vocal_truth.artifact_pcm_sha256:
            raise SourceFamilyError(
                "dataset vocal truth differs from family certificate"
            )
        if row.cell.spectral_grid_sha256 not in (
            certificate.spectral_grid_sha256s
        ):
            raise SourceFamilyError(
                "dataset cell uses an unregistered spectral grid"
            )
        if row.group_family.query_condition_sha256 not in (
            certificate.query_condition_sha256s
        ):
            raise SourceFamilyError(
                "dataset row uses an unregistered query condition"
            )
        for candidate in row.candidates:
            if not certificate.contains_candidate(
                candidate.recipe_id, candidate.artifact_pcm_sha256
            ):
                raise SourceFamilyError(
                    "dataset candidate is absent from the source-family inventory"
                )
