"""Content-derived exact-source lineage and pre-execution manifest v3.

V2 carried truth and decoded-PCM parent hashes as sibling declarations.  A
caller could therefore claim that unrelated risk-evidence and partition-source
reports shared the same parents.  V3 embeds each exact verification report as
canonical immutable UTF-8 bytes, requires the byte hash to equal the already
frozen unit/certificate identity, and derives the common parents by parsing those
bytes.

The normalized report is itself an independently verified artifact.  It names
the upstream source artifact and the exact truth/M/A/V parents.  Changing any
parent changes the report bytes and therefore cannot preserve the frozen unit or
partition-report SHA.

This module wraps the v2 pre-execution manifest so existing route attestation
logic can consume the stronger object through the same property/method surface.
It does not execute a model, render audio, or make a promotion decision.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .counterfactual_risk_cell_partition_v1 import CellPartitionCertificate
from .counterfactual_risk_evidence_bundle_v1 import EvidenceBundleV1
from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
)
from .counterfactual_risk_grouped_comparison_v3 import FrozenArmInferenceRunV3
from .counterfactual_risk_inference_partition_manifest_v2 import (
    InferenceInputPartitionManifestV2,
)

REPORT_SCHEMA = "audio-extract/exact-source-parent-report/v3"
DOCUMENT_SCHEMA = "audio-extract/exact-source-report-document/v3"
LINEAGE_SCHEMA = "audio-extract/d0-r0-exact-source-lineage/v3"
MANIFEST_SCHEMA = "audio-extract/d0-r0-inference-partition-inputs/v3"
REPORT_ROLES = {"risk_evidence", "partition_source"}
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class ExactSourceLineageV3Error(ValueError):
    """Exact report bytes or their derived common parents are inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise ExactSourceLineageV3Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise ExactSourceLineageV3Error(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ExactSourceLineageV3Error(
            f"{name} must be a non-empty trim-stable string"
        )
    return value


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
        raise ExactSourceLineageV3Error(
            f"exact source report is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class ExactSourceReportDocumentV3:
    """One immutable normalized report whose bytes are the referenced artifact."""

    payload_utf8: bytes

    @classmethod
    def build(
        cls,
        *,
        role: str,
        group_family_sha256: str,
        source_artifact_sha256: str,
        source_artifact_schema: str,
        truth_manifest_sha256: str,
        mixture_pcm_sha256: str,
        accompaniment_pcm_sha256: str,
        vocal_pcm_sha256: str,
        verifier_commit: str,
    ) -> ExactSourceReportDocumentV3:
        value = {
            "schema": REPORT_SCHEMA,
            "status": "verified",
            "role": role,
            "group_family_sha256": group_family_sha256,
            "source_artifact_sha256": source_artifact_sha256,
            "source_artifact_schema": source_artifact_schema,
            "truth_manifest_sha256": truth_manifest_sha256,
            "mixture_pcm_sha256": mixture_pcm_sha256,
            "accompaniment_pcm_sha256": accompaniment_pcm_sha256,
            "vocal_pcm_sha256": vocal_pcm_sha256,
            "verifier_commit": verifier_commit,
        }
        result = cls(_canonical(value))
        result.validate()
        return result

    def report(self) -> Mapping[str, Any]:
        if type(self.payload_utf8) is not bytes or not self.payload_utf8:
            raise ExactSourceLineageV3Error(
                "exact source report payload must be immutable non-empty bytes"
            )
        try:
            value = json.loads(self.payload_utf8.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExactSourceLineageV3Error(
                f"exact source report bytes are not UTF-8 JSON: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ExactSourceLineageV3Error(
                "exact source report payload is not a JSON object"
            )
        if _canonical(value) != self.payload_utf8:
            raise ExactSourceLineageV3Error(
                "exact source report bytes are not canonical"
            )
        return value

    def validate(self) -> None:
        value = self.report()
        expected = {
            "schema",
            "status",
            "role",
            "group_family_sha256",
            "source_artifact_sha256",
            "source_artifact_schema",
            "truth_manifest_sha256",
            "mixture_pcm_sha256",
            "accompaniment_pcm_sha256",
            "vocal_pcm_sha256",
            "verifier_commit",
        }
        if set(value) != expected:
            raise ExactSourceLineageV3Error(
                "exact source report has unknown or missing fields"
            )
        if value["schema"] != REPORT_SCHEMA or value["status"] != "verified":
            raise ExactSourceLineageV3Error(
                "exact source report has wrong schema/status"
            )
        if value["role"] not in REPORT_ROLES:
            raise ExactSourceLineageV3Error("unknown exact source report role")
        for name in (
            "group_family_sha256",
            "source_artifact_sha256",
            "truth_manifest_sha256",
            "mixture_pcm_sha256",
            "accompaniment_pcm_sha256",
            "vocal_pcm_sha256",
        ):
            _sha(value[name], f"report.{name}")
        _text(value["source_artifact_schema"], "source_artifact_schema")
        _git(value["verifier_commit"], "report verifier_commit")

    @property
    def sha256(self) -> str:
        self.validate()
        return "sha256:" + hashlib.sha256(self.payload_utf8).hexdigest()

    @property
    def role(self) -> str:
        return str(self.report()["role"])

    @property
    def group_family_sha256(self) -> str:
        return str(self.report()["group_family_sha256"])

    @property
    def verifier_commit(self) -> str:
        return str(self.report()["verifier_commit"])

    @property
    def parent_tuple(self) -> tuple[str, str, str, str]:
        value = self.report()
        return (
            str(value["truth_manifest_sha256"]),
            str(value["mixture_pcm_sha256"]),
            str(value["accompaniment_pcm_sha256"]),
            str(value["vocal_pcm_sha256"]),
        )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": DOCUMENT_SCHEMA,
            "payload_sha256": self.sha256,
            "report": dict(self.report()),
        }


@dataclass(frozen=True)
class ExactSourceLineageV3:
    risk_evidence_report: ExactSourceReportDocumentV3
    partition_source_report: ExactSourceReportDocumentV3
    lineage_verifier_commit: str
    status: str = "verified"

    def validate(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> None:
        unit.validate()
        certificate.validate()
        if self.status != "verified":
            raise ExactSourceLineageV3Error(
                "exact source lineage status must be verified"
            )
        _git(self.lineage_verifier_commit, "lineage_verifier_commit")
        self.risk_evidence_report.validate()
        self.partition_source_report.validate()
        if self.risk_evidence_report.role != "risk_evidence":
            raise ExactSourceLineageV3Error(
                "risk evidence document has the wrong role"
            )
        if self.partition_source_report.role != "partition_source":
            raise ExactSourceLineageV3Error(
                "partition source document has the wrong role"
            )
        if self.risk_evidence_report.sha256 != unit.exact_evidence_sha256:
            raise ExactSourceLineageV3Error(
                "risk evidence report bytes differ from the frozen unit"
            )
        if (
            self.partition_source_report.sha256
            != certificate.exact_source_report_sha256
        ):
            raise ExactSourceLineageV3Error(
                "partition source report bytes differ from the certificate"
            )
        expected_group = unit.group_family_sha256
        if certificate.group_family_sha256 != expected_group:
            raise ExactSourceLineageV3Error(
                "partition belongs to another source family"
            )
        if (
            self.risk_evidence_report.group_family_sha256 != expected_group
            or self.partition_source_report.group_family_sha256
            != expected_group
        ):
            raise ExactSourceLineageV3Error(
                "exact source reports name another source family"
            )
        if (
            self.risk_evidence_report.parent_tuple
            != self.partition_source_report.parent_tuple
        ):
            raise ExactSourceLineageV3Error(
                "risk and partition reports derive different truth/PCM parents"
            )
        if (
            self.partition_source_report.verifier_commit
            != certificate.verifier_commit
        ):
            raise ExactSourceLineageV3Error(
                "partition report verifier differs from its certificate"
            )

    def validate_bundle_membership(
        self,
        bundle: EvidenceBundleV1,
        exact_evidence_bundle_sha256: str,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> None:
        """Anchor content-derived lineage to a frozen exact-evidence bundle.

        Parsing report bytes proves the two reports are internally consistent,
        but NOT that they belong to the frozen study: a caller could synthesize
        a brand-new, internally-valid report with arbitrary parents.  This
        resolver requires BOTH properties at once — content-derivation (above)
        AND frozen-bundle membership:

        * the referenced ``exact_evidence_bundle_sha256`` bundle must already
          exist and its recomputed hash must equal the declared identity;
        * the held-out unit and its source family must be members of the bundle;
        * the selected partition certificate must be a member of the bundle's
          partition registry (its exact partition-source report is thereby a
          member, since the certificate names that report byte-hash);
        * the derived mixture/accompaniment/vocal parents must be exactly the
          frozen source-family roots the bundle certifies (M/A/V membership).

        A synthesized-but-internally-valid report that is not a member fails
        closed, as do copied/reused hashes, a substituted partition
        certificate, and a mismatched bundle.
        """

        # (1) Content-derivation must hold first: report bytes equal the frozen
        # unit/certificate identities and both reports derive the same parents.
        self.validate(unit, certificate)

        # (2) The referenced bundle must exist and its hash must verify.
        bundle.validate()
        declared = _sha(
            exact_evidence_bundle_sha256, "exact_evidence_bundle_sha256"
        )
        if bundle.sha256 != declared:
            raise ExactSourceLineageV3Error(
                "evidence bundle hash does not match the declared "
                "exact_evidence_bundle_sha256"
            )

        # (3) The held-out unit's group family must be a member of the bundle.
        if unit.group_family_sha256 not in bundle.dataset.group_family_sha256s:
            raise ExactSourceLineageV3Error(
                "held-out unit is not a member of the evidence bundle dataset"
            )

        # (4) The selected partition certificate must be a member of the bundle
        # partition registry.  self.validate already bound the partition-source
        # report byte-hash to certificate.exact_source_report_sha256, so proving
        # the certificate is a member proves the exact report is a member too.
        member_partition_shas = {
            member.sha256
            for member in bundle.partition_registry.certificates
        }
        if certificate.sha256 not in member_partition_shas:
            raise ExactSourceLineageV3Error(
                "partition certificate is not a member of the bundle "
                "partition registry"
            )

        # (5) M/A/V parent membership: the parents derived from the report bytes
        # must be exactly the frozen source-family roots the bundle certifies.
        source_certificate = bundle.source_registry.by_sha256().get(
            unit.source_family_sha256
        )
        if source_certificate is None:
            raise ExactSourceLineageV3Error(
                "unit source family is not a member of the bundle source "
                "registry"
            )
        _, mixture, accompaniment, vocal = self.parent_tuple
        frozen_roots = (
            source_certificate.mixture_root.identity.artifact_pcm_sha256,
            source_certificate.accompaniment_truth.identity.artifact_pcm_sha256,
            source_certificate.vocal_truth.identity.artifact_pcm_sha256,
        )
        if (mixture, accompaniment, vocal) != frozen_roots:
            raise ExactSourceLineageV3Error(
                "derived mixture/accompaniment/vocal parents are not the frozen "
                "source-family roots certified by the bundle"
            )

    @property
    def parent_tuple(self) -> tuple[str, str, str, str]:
        return self.risk_evidence_report.parent_tuple

    def identity_dict(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> dict[str, Any]:
        self.validate(unit, certificate)
        return {
            "schema": LINEAGE_SCHEMA,
            "status": self.status,
            "risk_evidence_report": (
                self.risk_evidence_report.identity_dict()
            ),
            "partition_source_report": (
                self.partition_source_report.identity_dict()
            ),
            "common_truth_manifest_sha256": self.parent_tuple[0],
            "common_mixture_pcm_sha256": self.parent_tuple[1],
            "common_accompaniment_pcm_sha256": self.parent_tuple[2],
            "common_vocal_pcm_sha256": self.parent_tuple[3],
            "lineage_verifier_commit": self.lineage_verifier_commit,
        }

    def sha256(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> str:
        return _mapping_sha(self.identity_dict(unit, certificate))


@dataclass(frozen=True)
class InferenceInputPartitionManifestV3:
    """V2 manifest plus source parents derived from frozen report bytes."""

    base: InferenceInputPartitionManifestV2
    source_lineage: ExactSourceLineageV3

    @property
    def unit_sha256(self) -> str:
        return self.base.unit_sha256

    @property
    def group_family_sha256(self) -> str:
        return self.base.group_family_sha256

    @property
    def arm_id(self) -> str:
        return self.base.arm_id

    @property
    def arm_run_sha256(self) -> str:
        return self.base.arm_run_sha256

    @property
    def dataset_sha256(self) -> str:
        return self.base.dataset_sha256

    @property
    def split_sha256(self) -> str:
        return self.base.split_sha256

    @property
    def test_subset_sha256(self) -> str:
        return self.base.test_subset_sha256

    @property
    def candidate_panel_sha256(self) -> str:
        return self.base.candidate_panel_sha256

    @property
    def feature_contract_sha256(self) -> str:
        return self.base.feature_contract_sha256

    @property
    def metric_contract_sha256(self) -> str:
        return self.base.metric_contract_sha256

    @property
    def partition_certificate_sha256(self) -> str:
        return self.base.partition_certificate_sha256

    @property
    def spectral_grid_sha256(self) -> str:
        return self.base.spectral_grid_sha256

    @property
    def resolution_ms(self) -> int:
        return self.base.resolution_ms

    @property
    def time_ranges_sha256(self) -> str:
        return self.base.time_ranges_sha256

    @property
    def frequency_ranges_sha256(self) -> str:
        return self.base.frequency_ranges_sha256

    @property
    def measure_contract_sha256(self) -> str:
        return self.base.measure_contract_sha256

    @property
    def partition_key(self) -> tuple[str, str, int]:
        return self.base.partition_key

    def _validate_without_run(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> None:
        self.base._validate_without_run(prereg, unit, certificate)
        self.source_lineage.validate(unit, certificate)
        truth, mixture, accompaniment, vocal = self.source_lineage.parent_tuple
        declared = self.base.source_lineage
        expected = {
            "exact_evidence_sha256": unit.exact_evidence_sha256,
            "exact_source_report_sha256": (
                certificate.exact_source_report_sha256
            ),
            "truth_manifest_sha256": truth,
            "mixture_pcm_sha256": mixture,
            "accompaniment_pcm_sha256": accompaniment,
            "vocal_pcm_sha256": vocal,
            "partition_source_verifier_commit": certificate.verifier_commit,
        }
        drift = [
            name
            for name, value in expected.items()
            if getattr(declared, name) != value
        ]
        if drift:
            raise ExactSourceLineageV3Error(
                f"legacy manifest lineage differs from report-derived facts: {drift}"
            )

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> dict[str, Any]:
        self._validate_without_run(prereg, unit, certificate)
        return {
            "schema": MANIFEST_SCHEMA,
            "base_manifest": self.base.identity_dict(
                prereg, unit, certificate
            ),
            "base_manifest_sha256": self.base.sha256(
                prereg, unit, certificate
            ),
            "source_lineage": self.source_lineage.identity_dict(
                unit, certificate
            ),
            "source_lineage_sha256": self.source_lineage.sha256(
                unit, certificate
            ),
        }

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg, unit, certificate))

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        run: FrozenArmInferenceRunV3,
        certificate: CellPartitionCertificate,
    ) -> None:
        self._validate_without_run(prereg, unit, certificate)
        run.validate(prereg, unit)
        expected = {
            "arm_id": self.arm_id,
            "arm_run_sha256": self.arm_run_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "test_subset_sha256": self.test_subset_sha256,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
        }
        drift = [
            name
            for name, value in expected.items()
            if getattr(run, name) != value
        ]
        if drift:
            raise ExactSourceLineageV3Error(
                f"inference run differs from v3 input manifest: {drift}"
            )
        if run.inference_input_manifest_sha256 != self.sha256(
            prereg, unit, certificate
        ):
            raise ExactSourceLineageV3Error(
                "inference run does not name the report-derived input manifest"
            )
