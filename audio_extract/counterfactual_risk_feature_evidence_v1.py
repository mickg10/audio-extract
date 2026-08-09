"""Identity-bearing inference-feature availability for grouped D0/R0 rows.

Finite zero is not evidence.  Exact dataset rows keep a canonical finite feature
vector, while this separate certificate carries per-feature availability and the
frozen required-feature policy.  Missing values must be stored as zero, remain
explicitly masked at inference, and force abstention whenever a required feature
is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import re

import numpy as np

from .counterfactual_risk_dataset_contract_v2 import (
    DatasetManifestV2,
)

CERTIFICATE_SCHEMA = "audio-extract/counterfactual-feature-evidence/v1"
REGISTRY_SCHEMA = "audio-extract/counterfactual-feature-evidence-registry/v1"
INFERENCE_SCHEMA = "audio-extract/counterfactual-risk-inference-row/v3"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class FeatureEvidenceError(ValueError):
    """A feature-evidence certificate, registry, or inference projection is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise FeatureEvidenceError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise FeatureEvidenceError(
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


def _bool_tuple(value: Sequence[Any], name: str) -> tuple[bool, ...]:
    result = tuple(value)
    if not result:
        raise FeatureEvidenceError(f"{name} must be non-empty")
    if any(not isinstance(item, (bool, np.bool_)) for item in result):
        raise FeatureEvidenceError(f"{name} must contain only booleans")
    return tuple(bool(item) for item in result)


def _mask_sha(value: Sequence[bool], *, component: str) -> str:
    array = np.ascontiguousarray(tuple(value), dtype=np.uint8)
    header = {
        "schema": CERTIFICATE_SCHEMA,
        "component": component,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class FeatureEvidenceCertificate:
    row_id: str
    source_family_sha256: str
    feature_contract_sha256: str
    feature_sha256: str
    available: tuple[bool, ...]
    required: tuple[bool, ...]
    missing_value_policy_sha256: str
    extraction_report_sha256: str
    extractor_bundle_sha256: str
    verifier_commit: str
    status: str = "passed"

    def validate(self, *, feature_count: int | None = None) -> None:
        if self.status != "passed":
            raise FeatureEvidenceError(
                "feature evidence certificate status must be passed"
            )
        for name, value in (
            ("row_id", self.row_id),
            ("source_family_sha256", self.source_family_sha256),
            ("feature_contract_sha256", self.feature_contract_sha256),
            ("feature_sha256", self.feature_sha256),
            (
                "missing_value_policy_sha256",
                self.missing_value_policy_sha256,
            ),
            ("extraction_report_sha256", self.extraction_report_sha256),
            ("extractor_bundle_sha256", self.extractor_bundle_sha256),
        ):
            _sha(value, name)
        available = _bool_tuple(self.available, "feature availability")
        required = _bool_tuple(self.required, "required feature mask")
        if len(available) != len(required):
            raise FeatureEvidenceError(
                "feature availability and required masks differ in length"
            )
        if feature_count is not None and len(available) != feature_count:
            raise FeatureEvidenceError(
                "feature-evidence mask length differs from the feature vector"
            )
        _git(self.verifier_commit, "verifier_commit")

    @property
    def prediction_allowed(self) -> bool:
        self.validate()
        return all(
            available or not required
            for available, required in zip(self.available, self.required)
        )

    @property
    def availability_sha256(self) -> str:
        self.validate()
        return _mask_sha(self.available, component="availability")

    @property
    def required_mask_sha256(self) -> str:
        self.validate()
        return _mask_sha(self.required, component="required")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CERTIFICATE_SCHEMA,
            "status": self.status,
            "row_id": self.row_id,
            "source_family_sha256": self.source_family_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "feature_sha256": self.feature_sha256,
            "availability_sha256": self.availability_sha256,
            "required_mask_sha256": self.required_mask_sha256,
            "available": list(self.available),
            "required": list(self.required),
            "prediction_allowed": self.prediction_allowed,
            "missing_value_policy_sha256": self.missing_value_policy_sha256,
            "extraction_report_sha256": self.extraction_report_sha256,
            "extractor_bundle_sha256": self.extractor_bundle_sha256,
            "verifier_commit": self.verifier_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class FeatureEvidenceRegistry:
    certificates: tuple[FeatureEvidenceCertificate, ...]
    source_commit: str

    @classmethod
    def build(
        cls,
        certificates: Sequence[FeatureEvidenceCertificate],
        *,
        source_commit: str,
    ) -> "FeatureEvidenceRegistry":
        ordered = tuple(sorted(tuple(certificates), key=lambda value: value.row_id))
        result = cls(ordered, source_commit)
        result.validate()
        return result

    def validate(self) -> None:
        _git(self.source_commit, "feature registry source_commit")
        if not self.certificates:
            raise FeatureEvidenceError("feature evidence registry is empty")
        for certificate in self.certificates:
            certificate.validate()
        row_ids = tuple(certificate.row_id for certificate in self.certificates)
        if row_ids != tuple(sorted(row_ids)) or len(set(row_ids)) != len(row_ids):
            raise FeatureEvidenceError(
                "feature evidence row IDs must be unique and canonically ordered"
            )
        required_masks = {
            (
                certificate.feature_contract_sha256,
                certificate.required_mask_sha256,
            )
            for certificate in self.certificates
        }
        if len(required_masks) != 1:
            raise FeatureEvidenceError(
                "one feature contract cannot use multiple required-feature masks"
            )

    def by_row_id(self) -> dict[str, FeatureEvidenceCertificate]:
        self.validate()
        return {certificate.row_id: certificate for certificate in self.certificates}

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


def validate_dataset_feature_evidence(
    dataset: DatasetManifestV2,
    registry: FeatureEvidenceRegistry,
) -> None:
    """Require exact feature/mask evidence for every row and no unused certs."""

    dataset.validate()
    registry.validate()
    certificates = registry.by_row_id()
    observed: set[str] = set()
    for row in dataset.rows:
        certificate = certificates.get(row.row_id)
        if certificate is None:
            raise FeatureEvidenceError(
                "dataset row lacks a feature-evidence certificate"
            )
        features = np.asarray(row.features, dtype=np.float64)
        certificate.validate(feature_count=len(features))
        if certificate.source_family_sha256 != (
            row.group_family.source_family_sha256
        ):
            raise FeatureEvidenceError(
                "feature certificate names a different source family"
            )
        if certificate.feature_contract_sha256 != row.feature_contract_sha256:
            raise FeatureEvidenceError(
                "feature certificate names a different feature contract"
            )
        if certificate.feature_sha256 != row.feature_sha256:
            raise FeatureEvidenceError(
                "feature certificate names different feature bytes"
            )
        mask = np.asarray(certificate.available, dtype=bool)
        if np.any(features[~mask] != 0.0):
            raise FeatureEvidenceError(
                "unavailable feature entries must be stored as canonical zero"
            )
        observed.add(row.row_id)
    unused = set(certificates) - observed
    if unused:
        raise FeatureEvidenceError(
            "feature evidence registry contains certificates unused by the dataset"
        )


def inference_records_with_feature_evidence(
    dataset: DatasetManifestV2,
    registry: FeatureEvidenceRegistry,
) -> tuple[dict[str, Any], ...]:
    """Return feature-only records with explicit availability and abstention."""

    validate_dataset_feature_evidence(dataset, registry)
    certificates = registry.by_row_id()
    result = []
    for row in dataset.rows:
        certificate = certificates[row.row_id]
        record = row.to_inference_record()
        record["schema"] = INFERENCE_SCHEMA
        record["feature_availability_sha256"] = (
            certificate.availability_sha256
        )
        record["required_feature_mask_sha256"] = (
            certificate.required_mask_sha256
        )
        record["feature_available"] = list(certificate.available)
        record["feature_required"] = list(certificate.required)
        record["prediction_allowed"] = certificate.prediction_allowed
        record["feature_evidence_sha256"] = certificate.sha256
        result.append(record)
    return tuple(result)
