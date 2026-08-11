"""Route/partition attestation with report-derived inference inputs v2.

V1 copied an opaque ``inference_input_manifest_sha256`` from the arm run and
copied partition facts from a separately selected certificate. A same-shaped
certificate at another resolution could therefore be substituted while all
local hashes were rebuilt.

The authoritative v2 attestation now requires
``InferenceInputPartitionManifestV3``. That manifest embeds the pre-execution
partition contract and two content-bearing exact-source reports, so the route
cannot be certified through the older caller-supplied ``ExactSourceLineageV2``
path.

A frozen partition registry may contain several legitimate resolutions for one
source family. The registry below requires exactly one attestation per held-out
unit/arm cell, requires D0 and R0 to select the same certificate for a unit, and
allows additional certified resolutions to remain unused.

This is descriptive CPU-only research infrastructure. It cannot select a
winner or authorize model training.
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
    CellPartitionRegistry,
)
from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
)
from .counterfactual_risk_grouped_comparison_v3 import (
    GroupedComparisonReportV3,
    WorkArmEvaluationV3,
)
from .counterfactual_risk_source_lineage_v3 import (
    InferenceInputPartitionManifestV3,
)

ATTESTATION_SCHEMA = "audio-extract/d0-r0-route-partition-attestation/v2"
REGISTRY_SCHEMA = "audio-extract/d0-r0-route-partition-attestation-registry/v2"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class RoutePartitionAttestationV2Error(ValueError):
    """Route, inference-input, and partition evidence are inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise RoutePartitionAttestationV2Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise RoutePartitionAttestationV2Error(
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
        raise RoutePartitionAttestationV2Error(
            f"attestation value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class RoutePartitionAttestationV2:
    """One unit/arm route bound to the exact pre-execution v3 manifest."""

    input_manifest: InferenceInputPartitionManifestV3
    route_artifact_sha256: str
    route_output_sha256: str
    submission_sha256: str
    partition_certificate_sha256: str
    labels_shape: tuple[int, int] | None
    labels_sha256: str | None
    verifier_commit: str
    status: str = "verified"

    @property
    def unit_sha256(self) -> str:
        return self.input_manifest.unit_sha256

    @property
    def arm_id(self) -> str:
        return self.input_manifest.arm_id

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
            raise RoutePartitionAttestationV2Error(
                "route-partition attestation status must be verified"
            )
        if not isinstance(
            self.input_manifest,
            InferenceInputPartitionManifestV3,
        ):
            raise RoutePartitionAttestationV2Error(
                "route attestation requires report-derived source lineage v3"
            )
        for name in (
            "route_artifact_sha256",
            "route_output_sha256",
            "submission_sha256",
            "partition_certificate_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.verifier_commit, "attestation verifier_commit")
        if self.key != (row.unit.sha256, row.arm_id):
            raise RoutePartitionAttestationV2Error(
                "attestation names another unit/arm cell"
            )
        artifact = row.route_provenance.route_artifact
        output = artifact.route_output
        submission = row.route_provenance.submission
        run = artifact.inference_run
        self.input_manifest.validate(
            prereg,
            row.unit,
            run,
            certificate,
        )
        expected = {
            "route_artifact_sha256": artifact.sha256(prereg, row.unit),
            "route_output_sha256": output.sha256(row.unit, row.arm_id),
            "submission_sha256": submission.sha256,
            "partition_certificate_sha256": certificate.sha256,
        }
        drift = [
            name
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise RoutePartitionAttestationV2Error(
                f"attestation differs from route/partition evidence: {drift}"
            )
        if submission.status == "ROUTE":
            expected_shape = (
                certificate.time_cell_count,
                certificate.band_count,
            )
            if self.labels_shape != expected_shape:
                raise RoutePartitionAttestationV2Error(
                    "route label shape differs from certified partition"
                )
            if output.labels_shape != expected_shape:
                raise RoutePartitionAttestationV2Error(
                    "route output shape differs from certified partition"
                )
            if self.labels_sha256 != submission.labels_sha256 or (
                self.labels_sha256 != output.labels_sha256
            ):
                raise RoutePartitionAttestationV2Error(
                    "label identity differs across output/submission/attestation"
                )
            _sha(self.labels_sha256, "attested labels_sha256")
        elif submission.status == "ABSTAIN":
            if self.labels_shape is not None or self.labels_sha256 is not None:
                raise RoutePartitionAttestationV2Error(
                    "abstention attestation must not expose labels"
                )
            if output.labels_shape is not None or output.labels_sha256 is not None:
                raise RoutePartitionAttestationV2Error(
                    "abstaining route output unexpectedly exposes labels"
                )
        else:
            raise RoutePartitionAttestationV2Error(
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
            "unit_sha256": self.unit_sha256,
            "arm_id": self.arm_id,
            "input_manifest": self.input_manifest.identity_dict(
                prereg,
                row.unit,
                certificate,
            ),
            "input_manifest_sha256": self.input_manifest.sha256(
                prereg,
                row.unit,
                certificate,
            ),
            "route_artifact_sha256": self.route_artifact_sha256,
            "route_output_sha256": self.route_output_sha256,
            "submission_sha256": self.submission_sha256,
            "partition_certificate_sha256": (
                self.partition_certificate_sha256
            ),
            "labels_shape": (
                None if self.labels_shape is None else list(self.labels_shape)
            ),
            "labels_sha256": self.labels_sha256,
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
class RoutePartitionAttestationRegistryV2:
    preregistration_sha256: str
    report_sha256: str
    partition_registry_sha256: str
    attestations: tuple[RoutePartitionAttestationV2, ...]
    source_commit: str
    verifier_commit: str
    status: str = "frozen"

    def validate(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
    ) -> None:
        report.validate(prereg)
        partitions.validate()
        if self.status != "frozen":
            raise RoutePartitionAttestationV2Error(
                "attestation registry must be frozen"
            )
        for name in (
            "preregistration_sha256",
            "report_sha256",
            "partition_registry_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.source_commit, "registry source_commit")
        _git(self.verifier_commit, "registry verifier_commit")
        expected_header = {
            "preregistration_sha256": prereg.sha256,
            "report_sha256": report.sha256(prereg),
            "partition_registry_sha256": partitions.sha256,
            "source_commit": prereg.source_commit,
        }
        drift = [
            name
            for name, value in expected_header.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise RoutePartitionAttestationV2Error(
                f"attestation registry header differs: {drift}"
            )
        if partitions.source_commit != prereg.source_commit:
            raise RoutePartitionAttestationV2Error(
                "partition registry source commit differs"
            )
        rows = {
            (row.unit.sha256, row.arm_id): row
            for row in report.work_evaluations
        }
        certificates = partitions.by_key()
        ordered = tuple(sorted(self.attestations, key=lambda row: row.key))
        if self.attestations != ordered:
            raise RoutePartitionAttestationV2Error(
                "attestations must be in canonical unit/arm order"
            )
        expected_keys = tuple(sorted(rows))
        actual_keys = tuple(row.key for row in ordered)
        if actual_keys != expected_keys or len(set(actual_keys)) != len(actual_keys):
            raise RoutePartitionAttestationV2Error(
                "attestation registry does not exactly cover report cells"
            )
        identities: dict[str, list[str]] = {
            "attestation": [],
            "input-manifest": [],
            "route-artifact": [],
            "route-output": [],
            "submission": [],
        }
        selected_by_unit: dict[str, list[RoutePartitionAttestationV2]] = {}
        for attestation in ordered:
            if not isinstance(
                attestation.input_manifest,
                InferenceInputPartitionManifestV3,
            ):
                raise RoutePartitionAttestationV2Error(
                    "registry contains an input manifest without source lineage v3"
                )
            row = rows[attestation.key]
            certificate = certificates.get(
                attestation.input_manifest.partition_key
            )
            if certificate is None:
                raise RoutePartitionAttestationV2Error(
                    "attested input manifest selects an uncertified partition"
                )
            attestation.validate(prereg, row, certificate)
            identities["attestation"].append(
                attestation.sha256(prereg, row, certificate)
            )
            identities["input-manifest"].append(
                attestation.input_manifest.sha256(
                    prereg,
                    row.unit,
                    certificate,
                )
            )
            identities["route-artifact"].append(
                attestation.route_artifact_sha256
            )
            identities["route-output"].append(
                attestation.route_output_sha256
            )
            identities["submission"].append(
                attestation.submission_sha256
            )
            selected_by_unit.setdefault(attestation.unit_sha256, []).append(
                attestation
            )
        for name, values in identities.items():
            if len(values) != len(set(values)):
                raise RoutePartitionAttestationV2Error(
                    f"attestation registry reuses one {name} across cells"
                )
        for unit_sha, values in selected_by_unit.items():
            if {row.arm_id for row in values} != {"D0", "R0"}:
                raise RoutePartitionAttestationV2Error(
                    f"unit {unit_sha} lacks both comparison arms"
                )
            selections = {
                (
                    row.partition_certificate_sha256,
                    row.input_manifest.spectral_grid_sha256,
                    row.input_manifest.resolution_ms,
                    row.input_manifest.time_ranges_sha256,
                    row.input_manifest.frequency_ranges_sha256,
                )
                for row in values
            }
            if len(selections) != 1:
                raise RoutePartitionAttestationV2Error(
                    "D0/R0 selected different certified partitions for one unit"
                )

    def resolved_certificates(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
    ) -> dict[tuple[str, str], CellPartitionCertificate]:
        self.validate(report, prereg, partitions)
        certificates = partitions.by_key()
        return {
            row.key: certificates[row.input_manifest.partition_key]
            for row in self.attestations
        }

    def identity_dict(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
    ) -> dict[str, Any]:
        self.validate(report, prereg, partitions)
        rows = {
            (row.unit.sha256, row.arm_id): row
            for row in report.work_evaluations
        }
        certificates = partitions.by_key()
        return {
            "schema": REGISTRY_SCHEMA,
            "status": self.status,
            "preregistration_sha256": self.preregistration_sha256,
            "report_sha256": self.report_sha256,
            "partition_registry_sha256": self.partition_registry_sha256,
            "source_commit": self.source_commit,
            "verifier_commit": self.verifier_commit,
            "attestations": [
                row.identity_dict(
                    prereg,
                    rows[row.key],
                    certificates[row.input_manifest.partition_key],
                )
                for row in self.attestations
            ],
        }

    def sha256(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
    ) -> str:
        return _mapping_sha(self.identity_dict(report, prereg, partitions))
