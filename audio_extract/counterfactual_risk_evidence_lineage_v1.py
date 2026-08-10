"""Bind one held-out comparison unit to the frozen exact-evidence bundle.

The report-derived source-lineage documents are useful immutable projections,
but a projection must not be allowed to establish its own authority.  This
module resolves the repository's existing ``EvidenceBundleV1`` and derives the
source parents from the strict ``SourceFamilyCertificateV2`` already registered
inside that bundle.

The binding proves, for one held-out unit and selected cell partition, that:

* the preregistration and unit name the exact evidence-bundle identity;
* dataset, split, candidate panel, feature, metric, and routing contracts agree;
* the unit is in the frozen test partition;
* its source-family certificate is an actual member of the bundle registry;
* the selected cell-partition certificate is an actual member of the bundle;
* mixture/accompaniment/vocal roots are derived from that certificate rather
  than accepted as caller-supplied common-parent declarations.

This is dormant CPU-only research infrastructure.  It performs no fitting,
calibration execution, audio rendering, model selection, or promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json
import re

from .counterfactual_risk_cell_partition_v1 import CellPartitionCertificate
from .counterfactual_risk_evidence_bundle_v1 import EvidenceBundleV1
from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
)
from .counterfactual_risk_source_family_v2 import SourceFamilyCertificateV2

BINDING_SCHEMA = "audio-extract/d0-r0-evidence-bundle-unit-binding/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class EvidenceBundleLineageError(ValueError):
    """A held-out unit is not transitively bound to the frozen evidence bundle."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise EvidenceBundleLineageError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise EvidenceBundleLineageError(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceBundleLineageError(
            f"evidence-lineage value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class EvidenceBundleUnitBindingV1:
    """Resolve one preregistered held-out unit through ``EvidenceBundleV1``."""

    evidence_bundle: EvidenceBundleV1
    source_family_certificate_sha256: str
    partition_certificate_sha256: str
    verifier_commit: str
    status: str = "verified"

    def _resolved_source_family(
        self,
        unit: HeldOutUnitV1,
    ) -> SourceFamilyCertificateV2:
        certificates = self.evidence_bundle.source_registry.by_sha256()
        source = certificates.get(unit.source_family_sha256)
        if source is None:
            raise EvidenceBundleLineageError(
                "held-out unit names a source family absent from the evidence bundle"
            )
        if self.source_family_certificate_sha256 != source.sha256:
            raise EvidenceBundleLineageError(
                "binding names another source-family certificate"
            )
        return source

    def _matching_rows(self, unit: HeldOutUnitV1):
        rows = tuple(
            row
            for row in self.evidence_bundle.dataset.rows
            if row.group_family.sha256 == unit.group_family_sha256
        )
        if not rows:
            raise EvidenceBundleLineageError(
                "held-out group family is absent from the exact evidence dataset"
            )
        if any(row.group_family != unit.group_family for row in rows):
            raise EvidenceBundleLineageError(
                "dataset group-family content differs from the held-out unit"
            )
        return rows

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> None:
        prereg.validate()
        unit.validate()
        certificate.validate()
        self.evidence_bundle.validate()
        if self.status != "verified":
            raise EvidenceBundleLineageError(
                "evidence-bundle unit binding status must be verified"
            )
        _sha(
            self.source_family_certificate_sha256,
            "source_family_certificate_sha256",
        )
        _sha(
            self.partition_certificate_sha256,
            "partition_certificate_sha256",
        )
        _git(self.verifier_commit, "binding verifier_commit")

        bundle = self.evidence_bundle
        expected_context = {
            "preregistration exact_evidence_manifest_sha256": (
                prereg.exact_evidence_manifest_sha256,
                bundle.sha256,
            ),
            "unit exact_evidence_sha256": (
                unit.exact_evidence_sha256,
                bundle.sha256,
            ),
            "dataset_sha256": (
                prereg.dataset_sha256,
                bundle.dataset.sha256,
            ),
            "split_sha256": (
                prereg.split_sha256,
                bundle.split.sha256(bundle.dataset),
            ),
            "source_commit": (
                prereg.source_commit,
                bundle.source_commit,
            ),
        }
        drift = [
            name
            for name, (actual, expected) in expected_context.items()
            if actual != expected
        ]
        if drift:
            raise EvidenceBundleLineageError(
                f"comparison context differs from the evidence bundle: {drift}"
            )

        if unit.group_family_sha256 not in (
            bundle.split.test_group_family_sha256s
        ):
            raise EvidenceBundleLineageError(
                "held-out unit is not in the frozen evidence-bundle test split"
            )

        rows = self._matching_rows(unit)
        row_contexts = {
            "candidate_panel_sha256": {
                row.candidate_panel.sha256 for row in rows
            },
            "feature_contract_sha256": {
                row.feature_contract_sha256 for row in rows
            },
            "metric_contract_sha256": {
                row.metric_contract_sha256 for row in rows
            },
            "routing_policy_sha256": {
                row.route_policy_sha256 for row in rows
            },
        }
        expected_rows = {
            "candidate_panel_sha256": prereg.candidate_panel_sha256,
            "feature_contract_sha256": prereg.feature_contract_sha256,
            "metric_contract_sha256": prereg.metric_contract_sha256,
            "routing_policy_sha256": prereg.routing_policy_sha256,
        }
        row_drift = [
            name
            for name, values in row_contexts.items()
            if values != {expected_rows[name]}
        ]
        if row_drift:
            raise EvidenceBundleLineageError(
                f"held-out evidence rows differ from preregistration: {row_drift}"
            )

        source = self._resolved_source_family(unit)
        if source.sha256 != unit.source_family_sha256:
            raise EvidenceBundleLineageError(
                "held-out unit does not name the resolved source-family certificate"
            )
        if certificate.group_family_sha256 != unit.group_family_sha256:
            raise EvidenceBundleLineageError(
                "selected partition belongs to another held-out group family"
            )
        resolved_partition = bundle.partition_registry.by_key().get(
            certificate.partition_key
        )
        if resolved_partition is None:
            raise EvidenceBundleLineageError(
                "selected partition is absent from the evidence bundle"
            )
        if (
            resolved_partition.sha256 != certificate.sha256
            or resolved_partition != certificate
        ):
            raise EvidenceBundleLineageError(
                "selected partition content differs from the bundled certificate"
            )
        if self.partition_certificate_sha256 != certificate.sha256:
            raise EvidenceBundleLineageError(
                "binding names another partition certificate"
            )
        if certificate.spectral_grid_sha256 not in (
            source.spectral_grid_sha256s
        ):
            raise EvidenceBundleLineageError(
                "selected partition grid is absent from the source-family certificate"
            )

        mixture = source.mixture_root.identity.artifact_pcm_sha256
        accompaniment = source.accompaniment_truth.identity.artifact_pcm_sha256
        vocal = source.vocal_truth.identity.artifact_pcm_sha256
        if any(row.mixture_pcm_sha256 != mixture for row in rows):
            raise EvidenceBundleLineageError(
                "exact evidence rows use another mixture root"
            )
        if any(
            row.accompaniment_truth_pcm_sha256 != accompaniment
            for row in rows
        ):
            raise EvidenceBundleLineageError(
                "exact evidence rows use another accompaniment truth"
            )
        if any(row.vocal_truth_pcm_sha256 != vocal for row in rows):
            raise EvidenceBundleLineageError(
                "exact evidence rows use another vocal truth"
            )

    def derived_parent_facts(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> dict[str, Any]:
        self.validate(prereg, unit, certificate)
        source = self._resolved_source_family(unit)
        return {
            "source_family_certificate_sha256": source.sha256,
            "discovery_manifest_sha256": source.discovery_manifest_sha256,
            "mixture_recipe_id": source.mixture_root.identity.recipe_id,
            "mixture_pcm_sha256": (
                source.mixture_root.identity.artifact_pcm_sha256
            ),
            "accompaniment_recipe_id": (
                source.accompaniment_truth.identity.recipe_id
            ),
            "accompaniment_pcm_sha256": (
                source.accompaniment_truth.identity.artifact_pcm_sha256
            ),
            "vocal_recipe_id": source.vocal_truth.identity.recipe_id,
            "vocal_pcm_sha256": (
                source.vocal_truth.identity.artifact_pcm_sha256
            ),
        }

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> dict[str, Any]:
        self.validate(prereg, unit, certificate)
        bundle = self.evidence_bundle
        return {
            "schema": BINDING_SCHEMA,
            "status": self.status,
            "evidence_bundle_sha256": bundle.sha256,
            "dataset_sha256": bundle.dataset.sha256,
            "split_sha256": bundle.split.sha256(bundle.dataset),
            "source_registry_sha256": bundle.source_registry.sha256,
            "partition_registry_sha256": bundle.partition_registry.sha256,
            "unit_sha256": unit.sha256,
            "group_family_sha256": unit.group_family_sha256,
            "source_family_certificate_sha256": (
                self.source_family_certificate_sha256
            ),
            "partition_certificate_sha256": (
                self.partition_certificate_sha256
            ),
            "derived_parent_facts": self.derived_parent_facts(
                prereg,
                unit,
                certificate,
            ),
            "verifier_commit": self.verifier_commit,
        }

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg, unit, certificate))
