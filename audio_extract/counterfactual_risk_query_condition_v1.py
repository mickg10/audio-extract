"""Content-bearing mixture-derived target-singer query conditions for D0/R0.

A bare query-condition hash is not provenance.  This module binds the target
identity, exact mixture-derived query artifact, encoder/feature contracts,
embedding bytes, extraction geometry, quality policy/report, and verifier code.
The query artifact must already be a strict ``query_source`` node whose complete
ancestry terminates only at the mixture root.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import re

from .counterfactual_risk_dataset_contract_v2 import DatasetManifestV2
from .counterfactual_risk_dataset_source_binding_v2 import (
    validate_dataset_v2_family_registry,
)
from .counterfactual_risk_source_family_v2 import (
    ArtifactRef,
    SourceFamilyRegistryV2,
    SourceFamilyV2Error,
)

CERTIFICATE_SCHEMA = "audio-extract/counterfactual-query-condition/v1"
REGISTRY_SCHEMA = "audio-extract/counterfactual-query-condition-registry/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ALLOWED_QUALITY_CLASSES = {
    "mixture_derived_verified",
    "mixture_derived_low_confidence",
}


class QueryConditionError(SourceFamilyV2Error):
    """A target-singer query certificate or dataset binding is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise QueryConditionError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise QueryConditionError(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise QueryConditionError(
            f"{name} must be a non-empty, trim-stable string"
        )
    return value


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
class QueryConditionCertificate:
    source_family_sha256: str
    target_singer_id: str
    target_identity_sha256: str
    query_source: ArtifactRef
    query_encoder_bundle_sha256: str
    query_feature_contract_sha256: str
    query_embedding_sha256: str
    query_segment_manifest_sha256: str
    query_projection_sha256: str
    quality_class: str
    quality_policy_sha256: str
    quality_report_sha256: str
    verifier_commit: str
    inference_safe: bool = True
    status: str = "passed"

    def validate(self) -> None:
        if self.status != "passed":
            raise QueryConditionError(
                "query condition certificate status must be passed"
            )
        if self.inference_safe is not True:
            raise QueryConditionError(
                "query condition must be certified inference-safe"
            )
        _text(self.target_singer_id, "target_singer_id")
        self.query_source.validate()
        for name, value in (
            ("source_family_sha256", self.source_family_sha256),
            ("target_identity_sha256", self.target_identity_sha256),
            (
                "query_encoder_bundle_sha256",
                self.query_encoder_bundle_sha256,
            ),
            (
                "query_feature_contract_sha256",
                self.query_feature_contract_sha256,
            ),
            ("query_embedding_sha256", self.query_embedding_sha256),
            (
                "query_segment_manifest_sha256",
                self.query_segment_manifest_sha256,
            ),
            ("query_projection_sha256", self.query_projection_sha256),
            ("quality_policy_sha256", self.quality_policy_sha256),
            ("quality_report_sha256", self.quality_report_sha256),
        ):
            _sha(value, name)
        if self.quality_class not in _ALLOWED_QUALITY_CLASSES:
            raise QueryConditionError(
                f"unknown query quality class {self.quality_class!r}"
            )
        if self.quality_class != "mixture_derived_verified":
            raise QueryConditionError(
                "only a verified mixture-derived query may enter D0/R0 inference"
            )
        _git(self.verifier_commit, "verifier_commit")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CERTIFICATE_SCHEMA,
            "status": self.status,
            "inference_safe": self.inference_safe,
            "source_family_sha256": self.source_family_sha256,
            "target_singer_id": self.target_singer_id,
            "target_identity_sha256": self.target_identity_sha256,
            "query_source": self.query_source.to_dict(),
            "query_encoder_bundle_sha256": (
                self.query_encoder_bundle_sha256
            ),
            "query_feature_contract_sha256": (
                self.query_feature_contract_sha256
            ),
            "query_embedding_sha256": self.query_embedding_sha256,
            "query_segment_manifest_sha256": (
                self.query_segment_manifest_sha256
            ),
            "query_projection_sha256": self.query_projection_sha256,
            "quality_class": self.quality_class,
            "quality_policy_sha256": self.quality_policy_sha256,
            "quality_report_sha256": self.quality_report_sha256,
            "verifier_commit": self.verifier_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class QueryConditionRegistry:
    certificates: tuple[QueryConditionCertificate, ...]
    source_commit: str

    @classmethod
    def build(
        cls,
        certificates: Sequence[QueryConditionCertificate],
        *,
        source_commit: str,
    ) -> "QueryConditionRegistry":
        ordered = tuple(sorted(tuple(certificates), key=lambda value: value.sha256))
        result = cls(ordered, source_commit)
        result.validate()
        return result

    def validate(self) -> None:
        _git(self.source_commit, "query registry source_commit")
        if not self.certificates:
            raise QueryConditionError("query condition registry is empty")
        for certificate in self.certificates:
            certificate.validate()
        identities = tuple(certificate.sha256 for certificate in self.certificates)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(
            identities
        ):
            raise QueryConditionError(
                "query certificates must be unique and canonically ordered"
            )
        embeddings = tuple(
            (
                certificate.source_family_sha256,
                certificate.query_embedding_sha256,
            )
            for certificate in self.certificates
        )
        if len(set(embeddings)) != len(embeddings):
            raise QueryConditionError(
                "one source-family query embedding has multiple certificates"
            )

    def by_sha256(self) -> dict[str, QueryConditionCertificate]:
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


def validate_dataset_query_conditions(
    dataset: DatasetManifestV2,
    source_registry: SourceFamilyRegistryV2,
    query_registry: QueryConditionRegistry,
) -> None:
    """Require one inference-safe content-bearing query per dataset group."""

    validate_dataset_v2_family_registry(dataset, source_registry)
    query_registry.validate()
    source_certificates = source_registry.by_sha256()
    queries = query_registry.by_sha256()
    observed: set[str] = set()
    for row in dataset.rows:
        query_sha = row.group_family.query_condition_sha256
        certificate = queries.get(query_sha)
        if certificate is None:
            raise QueryConditionError(
                "dataset row names an unregistered query-condition certificate"
            )
        certificate.validate()
        if certificate.source_family_sha256 != (
            row.group_family.source_family_sha256
        ):
            raise QueryConditionError(
                "query certificate names a different source family"
            )
        if certificate.target_singer_id != row.group_family.target_singer_id:
            raise QueryConditionError(
                "query certificate names a different target singer"
            )
        if certificate.query_feature_contract_sha256 != (
            row.feature_contract_sha256
        ):
            raise QueryConditionError(
                "query certificate names a different inference feature contract"
            )
        family = source_certificates[certificate.source_family_sha256]
        matching_nodes = tuple(
            node
            for node in family.derived_artifacts
            if node.identity == certificate.query_source
        )
        if len(matching_nodes) != 1 or matching_nodes[0].role != "query_source":
            raise QueryConditionError(
                "query source is absent from the strict family query inventory"
            )
        # family.validate() has already proved every query-source ancestry path
        # terminates only at the exact mixture root.
        if query_sha not in family.query_condition_sha256s:
            raise QueryConditionError(
                "query certificate is absent from the strict family condition inventory"
            )
        observed.add(query_sha)

    unused = set(queries) - observed
    if unused:
        raise QueryConditionError(
            "query condition registry contains certificates unused by the dataset"
        )
