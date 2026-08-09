"""Independent recipe-to-stable-slot certificates for D0/R0 candidates.

A row cannot prove that an exact source-specific recipe implements a declared
stable candidate slot merely by repeating the slot hash.  This module carries a
content-bearing verifier certificate and requires exact agreement among the
stable slot, source-family graph, recipe semantics, decoded artifact, and row
binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import re

from .counterfactual_risk_dataset_contract_v2 import (
    CandidateArtifactBinding,
    CandidateSlot,
    DatasetManifestV2,
)
from .counterfactual_risk_dataset_source_binding_v2 import (
    validate_dataset_v2_family_registry,
)
from .counterfactual_risk_source_family_v2 import (
    ArtifactRef,
    SourceFamilyRegistryV2,
    SourceFamilyV2Error,
)

CERTIFICATE_SCHEMA = "audio-extract/candidate-slot-projection/v1"
REGISTRY_SCHEMA = "audio-extract/candidate-slot-projection-registry/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class CandidateProjectionError(SourceFamilyV2Error):
    """A recipe-to-slot projection certificate or dataset binding is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise CandidateProjectionError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise CandidateProjectionError(
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


@dataclass(frozen=True)
class CandidateSlotProjectionCertificate:
    source_family_sha256: str
    source_root: ArtifactRef
    candidate: ArtifactRef
    slot_sha256: str
    slot_id: str
    model_bundle_sha256: str
    adapter_bundle_sha256: str
    construction_contract_sha256: str
    query_contract_sha256: str
    output_role: str
    recipe_semantic_sha256: str
    recipe_container_sha256: str
    effective_defaults_sha256: str
    source_lineage_sha256: str
    verification_manifest_sha256: str
    verifier_commit: str
    status: str = "passed"

    @property
    def certificate_key(self) -> tuple[str, str, ArtifactRef]:
        return self.source_family_sha256, self.slot_sha256, self.candidate

    def validate(self, *, expected_slot: CandidateSlot | None = None) -> None:
        if self.status != "passed":
            raise CandidateProjectionError(
                "candidate slot projection status must be passed"
            )
        _sha(self.source_family_sha256, "source_family_sha256")
        self.source_root.validate()
        self.candidate.validate()
        for name, value in (
            ("slot_sha256", self.slot_sha256),
            ("model_bundle_sha256", self.model_bundle_sha256),
            ("adapter_bundle_sha256", self.adapter_bundle_sha256),
            (
                "construction_contract_sha256",
                self.construction_contract_sha256,
            ),
            ("query_contract_sha256", self.query_contract_sha256),
            ("recipe_semantic_sha256", self.recipe_semantic_sha256),
            ("recipe_container_sha256", self.recipe_container_sha256),
            ("effective_defaults_sha256", self.effective_defaults_sha256),
            ("source_lineage_sha256", self.source_lineage_sha256),
            ("verification_manifest_sha256", self.verification_manifest_sha256),
        ):
            _sha(value, name)
        if not isinstance(self.slot_id, str) or not self.slot_id:
            raise CandidateProjectionError("slot_id must be a non-empty string")
        if self.output_role != "instrumental":
            raise CandidateProjectionError(
                "candidate projection must target instrumental output"
            )
        _git(self.verifier_commit, "verifier_commit")
        if expected_slot is not None:
            expected_slot.validate()
            comparisons = {
                "slot_sha256": (self.slot_sha256, expected_slot.sha256),
                "slot_id": (self.slot_id, expected_slot.slot_id),
                "model_bundle_sha256": (
                    self.model_bundle_sha256,
                    expected_slot.model_bundle_sha256,
                ),
                "adapter_bundle_sha256": (
                    self.adapter_bundle_sha256,
                    expected_slot.adapter_bundle_sha256,
                ),
                "construction_contract_sha256": (
                    self.construction_contract_sha256,
                    expected_slot.construction_contract_sha256,
                ),
                "query_contract_sha256": (
                    self.query_contract_sha256,
                    expected_slot.query_contract_sha256,
                ),
                "output_role": (self.output_role, expected_slot.output_role),
            }
            different = sorted(
                name for name, (actual, expected) in comparisons.items()
                if actual != expected
            )
            if different:
                raise CandidateProjectionError(
                    "candidate projection differs from stable slot fields: "
                    f"{different}"
                )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CERTIFICATE_SCHEMA,
            "status": self.status,
            "source_family_sha256": self.source_family_sha256,
            "source_root": self.source_root.to_dict(),
            "candidate": self.candidate.to_dict(),
            "slot_sha256": self.slot_sha256,
            "slot_id": self.slot_id,
            "model_bundle_sha256": self.model_bundle_sha256,
            "adapter_bundle_sha256": self.adapter_bundle_sha256,
            "construction_contract_sha256": (
                self.construction_contract_sha256
            ),
            "query_contract_sha256": self.query_contract_sha256,
            "output_role": self.output_role,
            "recipe_semantic_sha256": self.recipe_semantic_sha256,
            "recipe_container_sha256": self.recipe_container_sha256,
            "effective_defaults_sha256": self.effective_defaults_sha256,
            "source_lineage_sha256": self.source_lineage_sha256,
            "verification_manifest_sha256": self.verification_manifest_sha256,
            "verifier_commit": self.verifier_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())

    def validate_row_binding(
        self,
        binding: CandidateArtifactBinding,
        slot: CandidateSlot,
    ) -> None:
        self.validate(expected_slot=slot)
        binding.validate(
            expected_slot=slot,
            expected_source_family_sha256=self.source_family_sha256,
        )
        if ArtifactRef(binding.recipe_id, binding.artifact_pcm_sha256) != self.candidate:
            raise CandidateProjectionError(
                "row artifact differs from projection candidate identity"
            )
        if binding.recipe_semantic_sha256 != self.recipe_semantic_sha256:
            raise CandidateProjectionError(
                "row recipe semantic identity differs from projection certificate"
            )
        if binding.verifier_commit != self.verifier_commit:
            raise CandidateProjectionError(
                "row artifact verifier commit differs from projection certificate"
            )


@dataclass(frozen=True)
class CandidateProjectionRegistry:
    certificates: tuple[CandidateSlotProjectionCertificate, ...]
    source_commit: str

    @classmethod
    def build(
        cls,
        certificates: Sequence[CandidateSlotProjectionCertificate],
        *,
        source_commit: str,
    ) -> "CandidateProjectionRegistry":
        ordered = tuple(
            sorted(tuple(certificates), key=lambda value: value.certificate_key)
        )
        result = cls(ordered, source_commit)
        result.validate()
        return result

    def validate(self) -> None:
        _git(self.source_commit, "projection registry source_commit")
        if not self.certificates:
            raise CandidateProjectionError("candidate projection registry is empty")
        for certificate in self.certificates:
            certificate.validate()
        keys = tuple(certificate.certificate_key for certificate in self.certificates)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise CandidateProjectionError(
                "candidate projection keys must be unique and canonically ordered"
            )
        candidates = tuple(
            certificate.candidate for certificate in self.certificates
        )
        if len(set(candidates)) != len(candidates):
            raise CandidateProjectionError(
                "one exact candidate artifact has multiple slot projections"
            )

    def by_key(
        self,
    ) -> dict[
        tuple[str, str, ArtifactRef], CandidateSlotProjectionCertificate
    ]:
        self.validate()
        return {
            certificate.certificate_key: certificate
            for certificate in self.certificates
        }

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


def validate_dataset_candidate_projections(
    dataset: DatasetManifestV2,
    source_registry: SourceFamilyRegistryV2,
    projection_registry: CandidateProjectionRegistry,
) -> None:
    """Require one independently verified slot projection per dataset artifact."""

    validate_dataset_v2_family_registry(dataset, source_registry)
    projection_registry.validate()
    source_certificates = source_registry.by_sha256()
    projections = projection_registry.by_key()
    observed: set[tuple[str, str, ArtifactRef]] = set()
    for row in dataset.rows:
        source_family = row.group_family.source_family_sha256
        family = source_certificates[source_family]
        for slot, binding in zip(
            row.candidate_panel.slots, row.candidate_artifacts
        ):
            candidate = ArtifactRef(
                binding.recipe_id, binding.artifact_pcm_sha256
            )
            key = (source_family, slot.sha256, candidate)
            certificate = projections.get(key)
            if certificate is None:
                raise CandidateProjectionError(
                    "dataset artifact lacks an independent slot projection certificate"
                )
            if certificate.source_root != family.mixture_root.identity:
                raise CandidateProjectionError(
                    "projection source root differs from the strict family mixture root"
                )
            if not family.contains_candidate(candidate):
                raise CandidateProjectionError(
                    "projection candidate is absent from the strict family inventory"
                )
            certificate.validate_row_binding(binding, slot)
            observed.add(key)

    unused = set(projections) - observed
    if unused:
        raise CandidateProjectionError(
            "projection registry contains certificates unused by the dataset"
        )
