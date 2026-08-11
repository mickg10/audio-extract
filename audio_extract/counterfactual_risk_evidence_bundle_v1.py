"""One fail-closed semantic bundle for the dormant CPU-only D0/R0 study.

The individual dataset, source-family, query, recipe-slot projection, cell
partition, feature-evidence, and split contracts are easy to validate
selectively by mistake.  This module is the single semantic gate: a bundle is
valid only when every component verifies against the same exact dataset and
source commit.  Validation does not authorize export, fitting, calibration,
routing, rendering, or production promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .counterfactual_risk_candidate_projection_v1 import (
    CandidateProjectionRegistry,
    validate_dataset_candidate_projections,
)
from .counterfactual_risk_cell_partition_v1 import (
    CellPartitionRegistry,
    validate_dataset_cell_partitions,
)
from .counterfactual_risk_dataset_contract_v2 import (
    DatasetManifestV2,
    SplitManifestV2,
)
from .counterfactual_risk_dataset_source_binding_v2 import (
    validate_dataset_v2_family_registry,
)
from .counterfactual_risk_feature_evidence_v1 import (
    FeatureEvidenceRegistry,
    validate_dataset_feature_evidence,
)
from .counterfactual_risk_source_family_v2 import SourceFamilyRegistryV2

BUNDLE_SCHEMA = "audio-extract/counterfactual-risk-evidence-bundle/v1"
PURPOSE = "D0_R0_EQUAL_COMPUTE_CPU_ONLY"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class EvidenceBundleError(ValueError):
    """The combined exact evidence bundle is incomplete or inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise EvidenceBundleError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise EvidenceBundleError(
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
class EvidenceBundleV1:
    dataset: DatasetManifestV2
    split: SplitManifestV2
    source_registry: SourceFamilyRegistryV2
    projection_registry: CandidateProjectionRegistry
    partition_registry: CellPartitionRegistry
    feature_registry: FeatureEvidenceRegistry
    dataset_container_sha256: str
    dataset_schema_sha256: str
    source_family_schema_sha256: str
    projection_schema_sha256: str
    partition_schema_sha256: str
    feature_evidence_schema_sha256: str
    split_policy_sha256: str
    export_policy_sha256: str
    source_commit: str
    verifier_commit: str
    purpose: str = PURPOSE
    status: str = "passed"

    def validate(self) -> None:
        if self.status != "passed":
            raise EvidenceBundleError("evidence bundle status must be passed")
        if self.purpose != PURPOSE:
            raise EvidenceBundleError("unknown evidence bundle purpose")
        _git(self.source_commit, "bundle source_commit")
        _git(self.verifier_commit, "bundle verifier_commit")
        for name in (
            "dataset_container_sha256",
            "dataset_schema_sha256",
            "source_family_schema_sha256",
            "projection_schema_sha256",
            "partition_schema_sha256",
            "feature_evidence_schema_sha256",
            "split_policy_sha256",
            "export_policy_sha256",
        ):
            _sha(getattr(self, name), name)

        # Validate each component before comparing its source identity so a
        # malformed object cannot hide behind a syntactically valid commit.
        self.dataset.validate()
        self.source_registry.validate()
        self.projection_registry.validate()
        self.partition_registry.validate()
        self.feature_registry.validate()
        self.split.validate(self.dataset)

        commits = {
            "dataset": self.dataset.source_commit,
            "split": self.split.source_commit,
            "source_registry": self.source_registry.source_commit,
            "projection_registry": self.projection_registry.source_commit,
            "partition_registry": self.partition_registry.source_commit,
            "feature_registry": self.feature_registry.source_commit,
        }
        different = sorted(
            name for name, value in commits.items() if value != self.source_commit
        )
        if different:
            raise EvidenceBundleError(
                "evidence components were produced from different source commits: "
                f"{different}"
            )

        # Every semantic boundary is required.  Callers cannot opt out of one
        # verifier and still obtain a valid bundle identity.
        validate_dataset_v2_family_registry(
            self.dataset, self.source_registry
        )
        validate_dataset_candidate_projections(
            self.dataset, self.source_registry, self.projection_registry
        )
        validate_dataset_cell_partitions(
            self.dataset, self.partition_registry
        )
        validate_dataset_feature_evidence(
            self.dataset, self.feature_registry
        )

        if len(self.dataset.group_family_sha256s) < 3:
            raise EvidenceBundleError(
                "D0/R0 evidence requires at least train/calibration/test group families"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": BUNDLE_SCHEMA,
            "status": self.status,
            "purpose": self.purpose,
            "source_commit": self.source_commit,
            "verifier_commit": self.verifier_commit,
            "dataset_sha256": self.dataset.sha256,
            "dataset_container_sha256": self.dataset_container_sha256,
            "split_sha256": self.split.sha256(self.dataset),
            "source_registry_sha256": self.source_registry.sha256,
            "projection_registry_sha256": self.projection_registry.sha256,
            "partition_registry_sha256": self.partition_registry.sha256,
            "feature_registry_sha256": self.feature_registry.sha256,
            "dataset_schema_sha256": self.dataset_schema_sha256,
            "source_family_schema_sha256": self.source_family_schema_sha256,
            "projection_schema_sha256": self.projection_schema_sha256,
            "partition_schema_sha256": self.partition_schema_sha256,
            "feature_evidence_schema_sha256": (
                self.feature_evidence_schema_sha256
            ),
            "split_policy_sha256": self.split_policy_sha256,
            "export_policy_sha256": self.export_policy_sha256,
            "row_count": len(self.dataset.rows),
            "group_family_count": len(self.dataset.group_family_sha256s),
            "candidate_slot_count": len(
                self.dataset.rows[0].candidate_panel.slots
            ),
            "metric_count": len(self.dataset.rows[0].metric_names),
            "feature_count": len(self.dataset.rows[0].features),
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())
