"""Strict closed-world source-family certificates for grouped D0/R0 data.

V2 uses full recipe/decoded-PCM artifact references in the derivation graph.  This
removes the ambiguity of v1's PCM-only parent edges (notably when a no-vocal
mixture and accompaniment truth have identical PCM) and proves that every parent
path of a candidate or feature artifact terminates only at the exact mixture
root.  Multi-parent ensembles and convergent DAGs are supported; cycles, orphan
parents, and clean-truth ancestry are refused.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .counterfactual_risk_dataset_contract_v1 import (
    CounterfactualRiskDatasetError,
    DatasetManifest,
)

CERTIFICATE_SCHEMA = "audio-extract/counterfactual-source-family/v2"
REGISTRY_SCHEMA = "audio-extract/counterfactual-source-family-registry/v2"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ROOT_ROLES = {"mixture_root", "accompaniment_truth", "vocal_truth"}
_DERIVED_ROLES = {"candidate", "feature_source", "query_source"}


class SourceFamilyV2Error(CounterfactualRiskDatasetError):
    """A strict source-family graph, registry, or row binding is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise SourceFamilyV2Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise SourceFamilyV2Error(
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


def _ordered_shas(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise SourceFamilyV2Error(f"{name} must be non-empty")
    for index, value in enumerate(result):
        _sha(value, f"{name}[{index}]")
    if len(set(result)) != len(result):
        raise SourceFamilyV2Error(f"{name} contains duplicates")
    if result != tuple(sorted(result)):
        raise SourceFamilyV2Error(f"{name} must be in canonical lexical order")
    return result


@dataclass(frozen=True, order=True)
class ArtifactRef:
    recipe_id: str
    artifact_pcm_sha256: str

    def validate(self) -> None:
        _sha(self.recipe_id, "artifact recipe_id")
        _sha(self.artifact_pcm_sha256, "artifact artifact_pcm_sha256")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "recipe_id": self.recipe_id,
            "artifact_pcm_sha256": self.artifact_pcm_sha256,
        }


@dataclass(frozen=True)
class ArtifactNode:
    role: str
    identity: ArtifactRef
    parents: tuple[ArtifactRef, ...] = ()

    def validate(self, *, root: bool) -> None:
        if root:
            if self.role not in _ROOT_ROLES:
                raise SourceFamilyV2Error(
                    f"root node carries non-root role {self.role!r}"
                )
        elif self.role not in _DERIVED_ROLES:
            raise SourceFamilyV2Error(
                f"derived node carries unknown role {self.role!r}"
            )
        self.identity.validate()
        for parent in self.parents:
            parent.validate()
        if len(set(self.parents)) != len(self.parents):
            raise SourceFamilyV2Error("artifact parent references are duplicated")
        if self.parents != tuple(sorted(self.parents)):
            raise SourceFamilyV2Error(
                "artifact parent references must be canonically ordered"
            )
        if root and self.parents:
            raise SourceFamilyV2Error("root artifacts cannot have parents")
        if not root and not self.parents:
            raise SourceFamilyV2Error("derived artifacts require at least one parent")
        if self.identity in self.parents:
            raise SourceFamilyV2Error("an artifact cannot directly parent itself")

    def identity_dict(self) -> dict[str, Any]:
        self.validate(root=self.role in _ROOT_ROLES)
        return {
            "role": self.role,
            "identity": self.identity.to_dict(),
            "parents": [parent.to_dict() for parent in self.parents],
        }


@dataclass(frozen=True)
class SourceFamilyCertificateV2:
    mixture_root: ArtifactNode
    accompaniment_truth: ArtifactNode
    vocal_truth: ArtifactNode
    derived_artifacts: tuple[ArtifactNode, ...]
    spectral_grid_sha256s: tuple[str, ...]
    query_condition_sha256s: tuple[str, ...]
    discovery_manifest_sha256: str
    derivation_policy_sha256: str
    verifier_commit: str
    closed_world: bool = True
    status: str = "passed"

    def _roots(self) -> tuple[ArtifactNode, ArtifactNode, ArtifactNode]:
        return self.mixture_root, self.accompaniment_truth, self.vocal_truth

    def _nodes(self) -> dict[ArtifactRef, ArtifactNode]:
        return {
            node.identity: node for node in (*self._roots(), *self.derived_artifacts)
        }

    def validate(self) -> None:
        if self.status != "passed":
            raise SourceFamilyV2Error(
                "source-family certificate status must be passed"
            )
        if self.closed_world is not True:
            raise SourceFamilyV2Error(
                "source-family certificate must declare a closed-world inventory"
            )
        expected_roots = (
            (self.mixture_root, "mixture_root"),
            (self.accompaniment_truth, "accompaniment_truth"),
            (self.vocal_truth, "vocal_truth"),
        )
        for node, role in expected_roots:
            if node.role != role:
                raise SourceFamilyV2Error(
                    f"certificate {role} carries role {node.role!r}"
                )
            node.validate(root=True)
        for node in self.derived_artifacts:
            node.validate(root=False)
        if self.derived_artifacts != tuple(
            sorted(self.derived_artifacts, key=lambda value: value.identity)
        ):
            raise SourceFamilyV2Error(
                "derived artifacts must be canonically ordered by identity"
            )

        all_nodes = (*self._roots(), *self.derived_artifacts)
        refs = tuple(node.identity for node in all_nodes)
        if len(set(refs)) != len(refs):
            raise SourceFamilyV2Error(
                "artifact recipe/PCM identities must be unique in one family"
            )
        nodes = self._nodes()
        for node in self.derived_artifacts:
            unknown = set(node.parents) - set(nodes)
            if unknown:
                raise SourceFamilyV2Error(
                    "derived artifact has unknown parent references: "
                    f"{[item.to_dict() for item in sorted(unknown)]}"
                )

        # Global DFS proves acyclicity independently of whether any one branch
        # reaches the mixture root early.
        state: dict[ArtifactRef, int] = {}

        def visit(ref: ArtifactRef) -> None:
            current = state.get(ref, 0)
            if current == 1:
                raise SourceFamilyV2Error(
                    "source-family derivation graph contains a cycle"
                )
            if current == 2:
                return
            state[ref] = 1
            for parent in nodes[ref].parents:
                if parent in nodes and nodes[parent].role in _DERIVED_ROLES:
                    visit(parent)
            state[ref] = 2

        for node in self.derived_artifacts:
            visit(node.identity)

        root_role = {
            self.mixture_root.identity: "mixture_root",
            self.accompaniment_truth.identity: "accompaniment_truth",
            self.vocal_truth.identity: "vocal_truth",
        }
        memo: dict[ArtifactRef, frozenset[str]] = {}

        def terminal_roots(ref: ArtifactRef) -> frozenset[str]:
            if ref in root_role:
                return frozenset((root_role[ref],))
            cached = memo.get(ref)
            if cached is not None:
                return cached
            result: set[str] = set()
            for parent in nodes[ref].parents:
                result.update(terminal_roots(parent))
            frozen = frozenset(result)
            memo[ref] = frozen
            return frozen

        for node in self.derived_artifacts:
            roots = terminal_roots(node.identity)
            if node.role == "query_source" and not roots:
                raise SourceFamilyV2Error(
                    "query_source ancestry has no certified terminal root"
                )
            # Candidate, feature-source, and inference-time query artifacts must
            # all be mixture-only: a clean accompaniment/vocal truth reachable in
            # their ancestry would leak the exact target at inference.
            if node.role in {
                "candidate",
                "feature_source",
                "query_source",
            } and roots != frozenset(("mixture_root",)):
                raise SourceFamilyV2Error(
                    f"{node.role} ancestry must terminate only at mixture_root; "
                    f"found {sorted(roots)}"
                )

        _ordered_shas(self.spectral_grid_sha256s, "spectral_grid_sha256s")
        _ordered_shas(self.query_condition_sha256s, "query_condition_sha256s")
        _sha(self.discovery_manifest_sha256, "discovery_manifest_sha256")
        _sha(self.derivation_policy_sha256, "derivation_policy_sha256")
        _git(self.verifier_commit, "verifier_commit")

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
                node.identity_dict() for node in self.derived_artifacts
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

    def contains_candidate(self, candidate: ArtifactRef) -> bool:
        self.validate()
        return any(
            node.role == "candidate" and node.identity == candidate
            for node in self.derived_artifacts
        )


@dataclass(frozen=True)
class SourceFamilyRegistryV2:
    certificates: tuple[SourceFamilyCertificateV2, ...]
    source_commit: str

    @classmethod
    def build(
        cls,
        certificates: Sequence[SourceFamilyCertificateV2],
        *,
        source_commit: str,
    ) -> SourceFamilyRegistryV2:
        ordered = tuple(sorted(certificates, key=lambda value: value.sha256))
        result = cls(ordered, source_commit)
        result.validate()
        return result

    def validate(self) -> None:
        _git(self.source_commit, "registry source_commit")
        if not self.certificates:
            raise SourceFamilyV2Error("source-family registry is empty")
        for certificate in self.certificates:
            certificate.validate()
        identities = tuple(certificate.sha256 for certificate in self.certificates)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(
            identities
        ):
            raise SourceFamilyV2Error(
                "source-family certificates must be unique and canonically ordered"
            )
        mixture_refs = tuple(
            certificate.mixture_root.identity for certificate in self.certificates
        )
        if len(set(mixture_refs)) != len(mixture_refs):
            raise SourceFamilyV2Error(
                "one exact mixture artifact appears in multiple source families"
            )
        mixture_pcms = tuple(
            certificate.mixture_root.identity.artifact_pcm_sha256
            for certificate in self.certificates
        )
        if len(set(mixture_pcms)) != len(mixture_pcms):
            raise SourceFamilyV2Error(
                "decoded mixture PCM appears in multiple source families"
            )
        closed_world_manifests = tuple(
            certificate.discovery_manifest_sha256
            for certificate in self.certificates
            if certificate.closed_world
        )
        if len(set(closed_world_manifests)) != len(closed_world_manifests):
            raise SourceFamilyV2Error(
                "closed-world discovery manifest appears in multiple source families"
            )
        candidate_refs = tuple(
            node.identity
            for certificate in self.certificates
            for node in certificate.derived_artifacts
            if node.role == "candidate"
        )
        if len(set(candidate_refs)) != len(candidate_refs):
            raise SourceFamilyV2Error(
                "one candidate artifact identity appears in multiple families"
            )

    def by_sha256(self) -> dict[str, SourceFamilyCertificateV2]:
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


def validate_dataset_family_registry_v2(
    dataset: DatasetManifest,
    registry: SourceFamilyRegistryV2,
) -> None:
    """Bind every exact row to one strict source-family certificate."""

    dataset.validate()
    registry.validate()
    certificates = registry.by_sha256()
    for row in dataset.rows:
        family_sha = row.group_family.source_family_sha256
        certificate = certificates.get(family_sha)
        if certificate is None:
            raise SourceFamilyV2Error(
                "dataset row names an unregistered source-family certificate"
            )
        if row.mixture_pcm_sha256 != (
            certificate.mixture_root.identity.artifact_pcm_sha256
        ):
            raise SourceFamilyV2Error("dataset mixture differs from family root")
        if row.accompaniment_truth_pcm_sha256 != (
            certificate.accompaniment_truth.identity.artifact_pcm_sha256
        ):
            raise SourceFamilyV2Error(
                "dataset accompaniment truth differs from family certificate"
            )
        if row.vocal_truth_pcm_sha256 != (
            certificate.vocal_truth.identity.artifact_pcm_sha256
        ):
            raise SourceFamilyV2Error(
                "dataset vocal truth differs from family certificate"
            )
        if row.cell.spectral_grid_sha256 not in certificate.spectral_grid_sha256s:
            raise SourceFamilyV2Error(
                "dataset cell uses an unregistered spectral grid"
            )
        if row.group_family.query_condition_sha256 not in (
            certificate.query_condition_sha256s
        ):
            raise SourceFamilyV2Error(
                "dataset row uses an unregistered query condition"
            )
        for candidate in row.candidates:
            ref = ArtifactRef(
                recipe_id=candidate.recipe_id,
                artifact_pcm_sha256=candidate.artifact_pcm_sha256,
            )
            if not certificate.contains_candidate(ref):
                raise SourceFamilyV2Error(
                    "dataset candidate is absent from the source-family inventory"
                )
