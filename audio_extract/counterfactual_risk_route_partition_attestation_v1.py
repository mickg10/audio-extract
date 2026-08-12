"""Bind each D0/R0 route artifact to the exact certified cell partition.

A route label grid and a partition certificate may have compatible dimensions
while referring to different spectral ranges or resolutions.  This module adds
an independent attestation layer between the grouped comparison report and the
partition-based diagnostics:

    route artifact + submission
        -> exact CellPartitionCertificate
        -> verified per-cell route/partition attestation
        -> complete attestation registry
        -> non-promoting diagnostics envelope

The attestation does not train a model, render audio, or choose a winner.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .counterfactual_risk_cell_partition_v1 import (
    CellPartitionCertificate,
)
from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
)
from .counterfactual_risk_grouped_comparison_v3 import (
    GroupedComparisonReportV3,
    WorkArmEvaluationV3,
)
from .counterfactual_risk_grouped_diagnostics_v3 import (
    GeometryManifestV3,
    GroupedComparisonDiagnosticsV3,
)

ATTESTATION_SCHEMA = "audio-extract/d0-r0-route-partition-attestation/v1"
REGISTRY_SCHEMA = "audio-extract/d0-r0-route-partition-attestation-registry/v1"
ENVELOPE_SCHEMA = "audio-extract/d0-r0-certified-diagnostics-envelope/v1"
ENVELOPE_STATUS = "COMPLETE_NO_PROMOTION_DECISION"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class RoutePartitionAttestationError(ValueError):
    """Route and partition evidence do not describe one verified grid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise RoutePartitionAttestationError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise RoutePartitionAttestationError(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RoutePartitionAttestationError(
            f"attestation value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class RoutePartitionAttestationV1:
    """One independently verified unit/arm route-to-grid binding."""

    preregistration_sha256: str
    unit_sha256: str
    arm_id: str
    route_artifact_sha256: str
    route_output_sha256: str
    submission_sha256: str
    inference_input_manifest_sha256: str
    partition_certificate_sha256: str
    spectral_grid_sha256: str
    resolution_ms: int
    time_cell_count: int
    band_count: int
    labels_shape: tuple[int, int] | None
    labels_sha256: str | None
    source_commit: str
    verifier_commit: str
    status: str = "verified"

    @property
    def key(self) -> tuple[str, str]:
        return self.unit_sha256, self.arm_id

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        row: WorkArmEvaluationV3,
        certificate: CellPartitionCertificate,
    ) -> None:
        prereg.validate()
        row.validate(prereg)
        certificate.validate()
        if self.status != "verified":
            raise RoutePartitionAttestationError(
                "route-partition attestation status must be verified"
            )
        for name in (
            "preregistration_sha256",
            "unit_sha256",
            "route_artifact_sha256",
            "route_output_sha256",
            "submission_sha256",
            "inference_input_manifest_sha256",
            "partition_certificate_sha256",
            "spectral_grid_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.source_commit, "attestation source_commit")
        _git(self.verifier_commit, "attestation verifier_commit")
        if self.preregistration_sha256 != prereg.sha256:
            raise RoutePartitionAttestationError(
                "attestation names another preregistration"
            )
        if self.unit_sha256 != row.unit.sha256 or self.arm_id != row.arm_id:
            raise RoutePartitionAttestationError(
                "attestation names another unit/arm cell"
            )
        if self.source_commit != prereg.source_commit:
            raise RoutePartitionAttestationError(
                "attestation source commit differs"
            )
        artifact = row.route_provenance.route_artifact
        output = artifact.route_output
        submission = row.route_provenance.submission
        run = artifact.inference_run
        expected = {
            "route_artifact_sha256": artifact.sha256(prereg, row.unit),
            "route_output_sha256": output.sha256(row.unit, row.arm_id),
            "submission_sha256": submission.sha256,
            "inference_input_manifest_sha256": (
                run.inference_input_manifest_sha256
            ),
            "partition_certificate_sha256": certificate.sha256,
            "spectral_grid_sha256": certificate.spectral_grid_sha256,
        }
        drift = [
            name for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise RoutePartitionAttestationError(
                f"attestation differs from route/partition evidence: {drift}"
            )
        if certificate.group_family_sha256 != row.unit.group_family_sha256:
            raise RoutePartitionAttestationError(
                "attested partition belongs to another group family"
            )
        if certificate.exact_source_report_sha256 != (
            row.unit.exact_evidence_sha256
        ):
            raise RoutePartitionAttestationError(
                "attested partition belongs to another exact source report"
            )
        for name, expected_value in (
            ("resolution_ms", certificate.resolution_ms),
            ("time_cell_count", certificate.time_cell_count),
            ("band_count", certificate.band_count),
        ):
            value = getattr(self, name)
            if type(value) is not int or value != expected_value:
                raise RoutePartitionAttestationError(
                    f"attested {name} differs from partition certificate"
                )
        if submission.status == "ROUTE":
            expected_shape = (
                certificate.time_cell_count,
                certificate.band_count,
            )
            if self.labels_shape != expected_shape:
                raise RoutePartitionAttestationError(
                    "route label shape differs from certified partition"
                )
            if output.labels_shape != expected_shape:
                raise RoutePartitionAttestationError(
                    "route output shape differs from certified partition"
                )
            if self.labels_sha256 != submission.labels_sha256 or (
                self.labels_sha256 != output.labels_sha256
            ):
                raise RoutePartitionAttestationError(
                    "attested label identity differs from route output/submission"
                )
            _sha(self.labels_sha256, "attested labels_sha256")
        elif submission.status == "ABSTAIN":
            if self.labels_shape is not None or self.labels_sha256 is not None:
                raise RoutePartitionAttestationError(
                    "abstention attestation must not expose labels"
                )
            if output.labels_shape is not None or output.labels_sha256 is not None:
                raise RoutePartitionAttestationError(
                    "abstaining route output unexpectedly exposes labels"
                )
        else:  # submission.validate() should already prevent this
            raise RoutePartitionAttestationError(
                "unknown route submission status"
            )

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        row: WorkArmEvaluationV3,
        certificate: CellPartitionCertificate,
    ) -> dict[str, Any]:
        self.validate(prereg, row, certificate)
        return {
            "schema": ATTESTATION_SCHEMA,
            "status": self.status,
            "preregistration_sha256": self.preregistration_sha256,
            "unit_sha256": self.unit_sha256,
            "arm_id": self.arm_id,
            "route_artifact_sha256": self.route_artifact_sha256,
            "route_output_sha256": self.route_output_sha256,
            "submission_sha256": self.submission_sha256,
            "inference_input_manifest_sha256": (
                self.inference_input_manifest_sha256
            ),
            "partition_certificate_sha256": (
                self.partition_certificate_sha256
            ),
            "spectral_grid_sha256": self.spectral_grid_sha256,
            "resolution_ms": self.resolution_ms,
            "time_cell_count": self.time_cell_count,
            "band_count": self.band_count,
            "labels_shape": (
                None if self.labels_shape is None else list(self.labels_shape)
            ),
            "labels_sha256": self.labels_sha256,
            "source_commit": self.source_commit,
            "verifier_commit": self.verifier_commit,
        }

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        row: WorkArmEvaluationV3,
        certificate: CellPartitionCertificate,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg, row, certificate))


@dataclass(frozen=True)
class RoutePartitionAttestationRegistryV1:
    preregistration_sha256: str
    report_sha256: str
    geometry_manifest_sha256: str
    attestations: tuple[RoutePartitionAttestationV1, ...]
    source_commit: str
    verifier_commit: str
    status: str = "frozen"

    def validate(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
    ) -> None:
        report.validate(prereg)
        geometry.validate(prereg)
        if self.status != "frozen":
            raise RoutePartitionAttestationError(
                "attestation registry must be frozen"
            )
        for name in (
            "preregistration_sha256",
            "report_sha256",
            "geometry_manifest_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.source_commit, "registry source_commit")
        _git(self.verifier_commit, "registry verifier_commit")
        if self.preregistration_sha256 != prereg.sha256:
            raise RoutePartitionAttestationError(
                "attestation registry names another preregistration"
            )
        if self.report_sha256 != report.sha256(prereg):
            raise RoutePartitionAttestationError(
                "attestation registry names another grouped report"
            )
        if self.geometry_manifest_sha256 != geometry.sha256(prereg):
            raise RoutePartitionAttestationError(
                "attestation registry names another geometry manifest"
            )
        if self.source_commit != prereg.source_commit:
            raise RoutePartitionAttestationError(
                "attestation registry source commit differs"
            )
        rows = {
            (row.unit.sha256, row.arm_id): row
            for row in report.work_evaluations
        }
        partitions = geometry.resolved(prereg)
        ordered = tuple(sorted(self.attestations, key=lambda row: row.key))
        if self.attestations != ordered:
            raise RoutePartitionAttestationError(
                "attestations must be in canonical unit/arm order"
            )
        if tuple(row.key for row in ordered) != tuple(sorted(rows)):
            raise RoutePartitionAttestationError(
                "attestation registry does not exactly cover report cells"
            )
        identities = []
        for attestation in ordered:
            row = rows[attestation.key]
            certificate = partitions[attestation.unit_sha256]
            attestation.validate(prereg, row, certificate)
            identities.append(
                attestation.sha256(prereg, row, certificate)
            )
        if len(identities) != len(set(identities)):
            raise RoutePartitionAttestationError(
                "attestation registry repeats one attestation identity"
            )

    def identity_dict(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
    ) -> dict[str, Any]:
        self.validate(report, prereg, geometry)
        rows = {
            (row.unit.sha256, row.arm_id): row
            for row in report.work_evaluations
        }
        partitions = geometry.resolved(prereg)
        return {
            "schema": REGISTRY_SCHEMA,
            "status": self.status,
            "preregistration_sha256": self.preregistration_sha256,
            "report_sha256": self.report_sha256,
            "geometry_manifest_sha256": self.geometry_manifest_sha256,
            "source_commit": self.source_commit,
            "verifier_commit": self.verifier_commit,
            "attestations": [
                row.identity_dict(
                    prereg,
                    rows[row.key],
                    partitions[row.unit_sha256],
                )
                for row in self.attestations
            ],
        }

    def sha256(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
    ) -> str:
        return _mapping_sha(self.identity_dict(report, prereg, geometry))


@dataclass(frozen=True)
class CertifiedGroupedDiagnosticsEnvelopeV1:
    report_sha256: str
    geometry_manifest_sha256: str
    diagnostics_sha256: str
    attestation_registry_sha256: str
    verifier_commit: str
    status: str = ENVELOPE_STATUS

    def validate(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
        diagnostics: GroupedComparisonDiagnosticsV3,
        attestations: RoutePartitionAttestationRegistryV1,
    ) -> None:
        report.validate(prereg)
        geometry.validate(prereg)
        diagnostics.validate(report, prereg, geometry)
        attestations.validate(report, prereg, geometry)
        if self.status != ENVELOPE_STATUS:
            raise RoutePartitionAttestationError(
                "certified diagnostics envelope must not promote an arm"
            )
        for name in (
            "report_sha256",
            "geometry_manifest_sha256",
            "diagnostics_sha256",
            "attestation_registry_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.verifier_commit, "envelope verifier_commit")
        expected = {
            "report_sha256": report.sha256(prereg),
            "geometry_manifest_sha256": geometry.sha256(prereg),
            "diagnostics_sha256": diagnostics.sha256(
                report, prereg, geometry
            ),
            "attestation_registry_sha256": attestations.sha256(
                report, prereg, geometry
            ),
        }
        drift = [
            name for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise RoutePartitionAttestationError(
                f"certified diagnostics envelope differs: {drift}"
            )

    def identity_dict(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
        diagnostics: GroupedComparisonDiagnosticsV3,
        attestations: RoutePartitionAttestationRegistryV1,
    ) -> dict[str, Any]:
        self.validate(report, prereg, geometry, diagnostics, attestations)
        return {
            "schema": ENVELOPE_SCHEMA,
            "status": self.status,
            "promotion_decision": None,
            "report_sha256": self.report_sha256,
            "geometry_manifest_sha256": self.geometry_manifest_sha256,
            "diagnostics_sha256": self.diagnostics_sha256,
            "attestation_registry_sha256": (
                self.attestation_registry_sha256
            ),
            "verifier_commit": self.verifier_commit,
        }

    def sha256(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
        diagnostics: GroupedComparisonDiagnosticsV3,
        attestations: RoutePartitionAttestationRegistryV1,
    ) -> str:
        return _mapping_sha(
            self.identity_dict(
                report, prereg, geometry, diagnostics, attestations
            )
        )
