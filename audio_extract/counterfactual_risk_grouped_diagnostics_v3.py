"""Cell-partition-bound diagnostics for grouped D0/R0 comparison v3.

The authoritative geometry is the existing `CellPartitionRegistry`: it contains
the complete row-major cell list, exact rational measures, time/frequency-range
hashes, spectral grid, resolution, source-report identity, and verifier commit.
This module binds each held-out unit to exactly one certificate from that
registry before normalizing route switch counts.

The artifact is descriptive and non-promoting.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .counterfactual_risk_cell_partition_v1 import (
    CellPartitionCertificate,
    CellPartitionRegistry,
)
from .counterfactual_risk_grouped_comparison_v1 import (
    ROUTE_STATUSES,
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
)
from .counterfactual_risk_grouped_comparison_v3 import (
    GroupedComparisonReportV3,
)
from .counterfactual_risk_grouped_diagnostics_v2 import (
    OUTCOME_ORDER,
    NormalizedSwitchSummaryV2,
    StatusDominanceSummaryV2,
)

UNIT_BINDING_SCHEMA = "audio-extract/d0-r0-unit-partition-binding/v3"
GEOMETRY_MANIFEST_SCHEMA = "audio-extract/d0-r0-geometry-manifest/v3"
DIAGNOSTICS_SCHEMA = "audio-extract/d0-r0-grouped-diagnostics/v3"
DIAGNOSTICS_STATUS = "COMPLETE_NO_PROMOTION_DECISION"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupedDiagnosticsV3Error(ValueError):
    """Certified partition evidence or grouped diagnostics are inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedDiagnosticsV3Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedDiagnosticsV3Error(
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
        raise GroupedDiagnosticsV3Error(
            f"diagnostic value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class UnitPartitionBindingV3:
    """Bind one held-out unit to one certificate in the frozen registry."""

    unit_sha256: str
    group_family_sha256: str
    spectral_grid_sha256: str
    resolution_ms: int
    partition_certificate_sha256: str
    exact_evidence_sha256: str
    exact_panel_sha256: str
    exact_preflight_sha256: str
    exact_oracle_decision_sha256: str
    status: str = "verified"

    @property
    def partition_key(self) -> tuple[str, str, int]:
        return (
            self.group_family_sha256,
            self.spectral_grid_sha256,
            self.resolution_ms,
        )

    def validate(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> None:
        unit.validate()
        certificate.validate()
        if self.status != "verified":
            raise GroupedDiagnosticsV3Error(
                "unit partition binding status must be verified"
            )
        for name in (
            "unit_sha256",
            "group_family_sha256",
            "spectral_grid_sha256",
            "partition_certificate_sha256",
            "exact_evidence_sha256",
            "exact_panel_sha256",
            "exact_preflight_sha256",
            "exact_oracle_decision_sha256",
        ):
            _sha(getattr(self, name), name)
        if type(self.resolution_ms) is not int or self.resolution_ms < 1:
            raise GroupedDiagnosticsV3Error(
                "resolution_ms must be a positive integer"
            )
        expected_unit = {
            "unit_sha256": unit.sha256,
            "group_family_sha256": unit.group_family_sha256,
            "exact_evidence_sha256": unit.exact_evidence_sha256,
            "exact_panel_sha256": unit.exact_panel_sha256,
            "exact_preflight_sha256": unit.exact_preflight_sha256,
            "exact_oracle_decision_sha256": (
                unit.exact_oracle_decision_sha256
            ),
        }
        drift = [
            name for name, expected in expected_unit.items()
            if getattr(self, name) != expected
        ]
        if drift:
            raise GroupedDiagnosticsV3Error(
                f"partition binding names another exact unit: {drift}"
            )
        if self.partition_key != certificate.partition_key:
            raise GroupedDiagnosticsV3Error(
                "partition binding key differs from certificate"
            )
        if self.partition_certificate_sha256 != certificate.sha256:
            raise GroupedDiagnosticsV3Error(
                "partition binding names another certificate"
            )
        if certificate.exact_source_report_sha256 != (
            unit.exact_evidence_sha256
        ):
            raise GroupedDiagnosticsV3Error(
                "cell partition was verified against another exact source report"
            )

    def identity_dict(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> dict[str, Any]:
        self.validate(unit, certificate)
        return {"schema": UNIT_BINDING_SCHEMA, **self.__dict__}

    def sha256(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> str:
        return _mapping_sha(self.identity_dict(unit, certificate))


@dataclass(frozen=True)
class GeometryManifestV3:
    preregistration_sha256: str
    partition_registry: CellPartitionRegistry
    bindings: tuple[UnitPartitionBindingV3, ...]
    verifier_commit: str
    status: str = "frozen"

    def validate(self, prereg: GroupedComparisonPreregistrationV1) -> None:
        prereg.validate()
        if self.status != "frozen":
            raise GroupedDiagnosticsV3Error(
                "geometry manifest must be frozen"
            )
        _sha(self.preregistration_sha256, "preregistration_sha256")
        _git(self.verifier_commit, "geometry verifier_commit")
        if self.preregistration_sha256 != prereg.sha256:
            raise GroupedDiagnosticsV3Error(
                "geometry manifest names another preregistration"
            )
        self.partition_registry.validate()
        if self.partition_registry.source_commit != prereg.source_commit:
            raise GroupedDiagnosticsV3Error(
                "partition registry source commit differs from preregistration"
            )
        units = {unit.sha256: unit for unit in prereg.expected_units}
        certificates = self.partition_registry.by_key()
        ordered = tuple(sorted(self.bindings, key=lambda row: row.unit_sha256))
        if self.bindings != ordered:
            raise GroupedDiagnosticsV3Error(
                "unit partition bindings must be in canonical unit order"
            )
        if tuple(row.unit_sha256 for row in ordered) != tuple(sorted(units)):
            raise GroupedDiagnosticsV3Error(
                "geometry manifest does not exactly cover held-out units"
            )
        if len({row.unit_sha256 for row in ordered}) != len(ordered):
            raise GroupedDiagnosticsV3Error(
                "geometry manifest repeats a held-out unit"
            )
        used_certificates = []
        for binding in ordered:
            certificate = certificates.get(binding.partition_key)
            if certificate is None:
                raise GroupedDiagnosticsV3Error(
                    "unit binding has no certificate in the frozen registry"
                )
            binding.validate(units[binding.unit_sha256], certificate)
            used_certificates.append(certificate.sha256)
        if len(used_certificates) != len(set(used_certificates)):
            raise GroupedDiagnosticsV3Error(
                "geometry manifest reuses one partition across held-out units"
            )
        if set(used_certificates) != {
            certificate.sha256
            for certificate in self.partition_registry.certificates
        }:
            raise GroupedDiagnosticsV3Error(
                "partition registry contains unused or unbound certificates"
            )

    def resolved(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> dict[str, CellPartitionCertificate]:
        self.validate(prereg)
        certificates = self.partition_registry.by_key()
        return {
            binding.unit_sha256: certificates[binding.partition_key]
            for binding in self.bindings
        }

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> dict[str, Any]:
        self.validate(prereg)
        units = {unit.sha256: unit for unit in prereg.expected_units}
        certificates = self.partition_registry.by_key()
        return {
            "schema": GEOMETRY_MANIFEST_SCHEMA,
            "status": self.status,
            "preregistration_sha256": self.preregistration_sha256,
            "verifier_commit": self.verifier_commit,
            "partition_registry": self.partition_registry.identity_dict(),
            "partition_registry_sha256": self.partition_registry.sha256,
            "bindings": [
                binding.identity_dict(
                    units[binding.unit_sha256],
                    certificates[binding.partition_key],
                )
                for binding in self.bindings
            ],
        }

    def sha256(self, prereg: GroupedComparisonPreregistrationV1) -> str:
        return _mapping_sha(self.identity_dict(prereg))


@dataclass(frozen=True)
class GroupedComparisonDiagnosticsV3:
    report_sha256: str
    geometry_manifest_sha256: str
    verifier_commit: str
    d0_switching: NormalizedSwitchSummaryV2
    r0_switching: NormalizedSwitchSummaryV2
    status_dominance: StatusDominanceSummaryV2
    status: str = DIAGNOSTICS_STATUS

    def validate(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
    ) -> None:
        report.validate(prereg)
        geometry.validate(prereg)
        if self.status != DIAGNOSTICS_STATUS:
            raise GroupedDiagnosticsV3Error(
                "diagnostics must not contain a promotion decision"
            )
        _sha(self.report_sha256, "report_sha256")
        _sha(self.geometry_manifest_sha256, "geometry_manifest_sha256")
        _git(self.verifier_commit, "diagnostics verifier_commit")
        if self.report_sha256 != report.sha256(prereg):
            raise GroupedDiagnosticsV3Error(
                "diagnostics names another grouped report"
            )
        if self.geometry_manifest_sha256 != geometry.sha256(prereg):
            raise GroupedDiagnosticsV3Error(
                "diagnostics names another geometry manifest"
            )
        self.d0_switching.validate()
        self.r0_switching.validate()
        self.status_dominance.validate()
        expected = build_grouped_diagnostics_v3(
            report,
            prereg,
            geometry,
            verifier_commit=self.verifier_commit,
            _skip_validation=True,
        )
        if self != expected:
            raise GroupedDiagnosticsV3Error(
                "stored diagnostics differ from report/partition evidence"
            )

    def identity_dict(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
    ) -> dict[str, Any]:
        self.validate(report, prereg, geometry)
        return {
            "schema": DIAGNOSTICS_SCHEMA,
            "status": self.status,
            "promotion_decision": None,
            "report_sha256": self.report_sha256,
            "geometry_manifest_sha256": self.geometry_manifest_sha256,
            "verifier_commit": self.verifier_commit,
            "d0_switching": self.d0_switching.identity_dict(),
            "r0_switching": self.r0_switching.identity_dict(),
            "status_dominance": self.status_dominance.identity_dict(),
        }

    def sha256(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV3,
    ) -> str:
        return _mapping_sha(self.identity_dict(report, prereg, geometry))


def _validate_every_submitted_route(
    report: GroupedComparisonReportV3,
    partitions: Mapping[str, CellPartitionCertificate],
) -> None:
    for row in report.work_evaluations:
        if row.evaluation.status not in ROUTE_STATUSES:
            continue
        certificate = partitions[row.unit.sha256]
        temporal = row.evaluation.temporal_switches
        frequency = row.evaluation.frequency_switches
        if type(temporal) is not int or type(frequency) is not int:
            raise GroupedDiagnosticsV3Error(
                "submitted route lacks integer switch counts"
            )
        temporal_boundaries = max(0, certificate.time_cell_count - 1) * (
            certificate.band_count
        )
        frequency_boundaries = certificate.time_cell_count * max(
            0, certificate.band_count - 1
        )
        if temporal > temporal_boundaries:
            raise GroupedDiagnosticsV3Error(
                f"{row.arm_id} temporal switches exceed certified boundaries"
            )
        if frequency > frequency_boundaries:
            raise GroupedDiagnosticsV3Error(
                f"{row.arm_id} frequency switches exceed certified boundaries"
            )


def _switch_summary(
    arm_id: str,
    report: GroupedComparisonReportV3,
    partitions: Mapping[str, CellPartitionCertificate],
) -> NormalizedSwitchSummaryV2:
    records = tuple(
        row for row in report.work_evaluations
        if row.arm_id == arm_id and row.evaluation.status == "ROUTE_SAFE"
    )
    temporal_total = 0
    temporal_boundaries = 0
    frequency_total = 0
    frequency_boundaries = 0
    temporal_rates = []
    frequency_rates = []
    for row in records:
        certificate = partitions[row.unit.sha256]
        t_boundaries = max(0, certificate.time_cell_count - 1) * (
            certificate.band_count
        )
        f_boundaries = certificate.time_cell_count * max(
            0, certificate.band_count - 1
        )
        temporal = int(row.evaluation.temporal_switches)
        frequency = int(row.evaluation.frequency_switches)
        temporal_total += temporal
        temporal_boundaries += t_boundaries
        frequency_total += frequency
        frequency_boundaries += f_boundaries
        if t_boundaries:
            temporal_rates.append(temporal / t_boundaries)
        if f_boundaries:
            frequency_rates.append(frequency / f_boundaries)
    result = NormalizedSwitchSummaryV2(
        arm_id=arm_id,
        safe_unit_count=len(records),
        temporal_switch_total=temporal_total,
        temporal_boundary_total=temporal_boundaries,
        temporal_switch_rate=_rate(temporal_total, temporal_boundaries),
        mean_unit_temporal_switch_rate=(
            fmean(temporal_rates) if temporal_rates else None
        ),
        frequency_switch_total=frequency_total,
        frequency_boundary_total=frequency_boundaries,
        frequency_switch_rate=_rate(frequency_total, frequency_boundaries),
        mean_unit_frequency_switch_rate=(
            fmean(frequency_rates) if frequency_rates else None
        ),
    )
    result.validate()
    return result


def _status_dominance(
    report: GroupedComparisonReportV3,
    prereg: GroupedComparisonPreregistrationV1,
) -> StatusDominanceSummaryV2:
    by_key = {
        (row.unit.sha256, row.arm_id): row.evaluation.status
        for row in report.work_evaluations
    }
    d0_better = 0
    r0_better = 0
    equal = 0
    unavailable = 0
    for unit in prereg.expected_units:
        d0 = by_key[(unit.sha256, "D0")]
        r0 = by_key[(unit.sha256, "R0")]
        if d0 == r0 == "ORACLE_UNAVAILABLE":
            unavailable += 1
            continue
        if d0 == "ORACLE_UNAVAILABLE" or r0 == "ORACLE_UNAVAILABLE":
            raise GroupedDiagnosticsV3Error(
                "oracle availability differs inside grouped report"
            )
        if d0 not in OUTCOME_ORDER or r0 not in OUTCOME_ORDER:
            raise GroupedDiagnosticsV3Error(
                "unknown outcome in status-dominance summary"
            )
        if OUTCOME_ORDER[d0] > OUTCOME_ORDER[r0]:
            d0_better += 1
        elif OUTCOME_ORDER[r0] > OUTCOME_ORDER[d0]:
            r0_better += 1
        else:
            equal += 1
    result = StatusDominanceSummaryV2(
        unit_count=len(prereg.expected_units),
        oracle_available_count=len(prereg.expected_units) - unavailable,
        d0_status_better_count=d0_better,
        r0_status_better_count=r0_better,
        equal_status_count=equal,
        oracle_unavailable_count=unavailable,
    )
    result.validate()
    return result


def build_grouped_diagnostics_v3(
    report: GroupedComparisonReportV3,
    prereg: GroupedComparisonPreregistrationV1,
    geometry: GeometryManifestV3,
    *,
    verifier_commit: str,
    _skip_validation: bool = False,
) -> GroupedComparisonDiagnosticsV3:
    if not _skip_validation:
        report.validate(prereg)
        geometry.validate(prereg)
    _git(verifier_commit, "diagnostics verifier_commit")
    partitions = geometry.resolved(prereg)
    _validate_every_submitted_route(report, partitions)
    result = GroupedComparisonDiagnosticsV3(
        report_sha256=report.sha256(prereg),
        geometry_manifest_sha256=geometry.sha256(prereg),
        verifier_commit=verifier_commit,
        d0_switching=_switch_summary("D0", report, partitions),
        r0_switching=_switch_summary("R0", report, partitions),
        status_dominance=_status_dominance(report, prereg),
    )
    if not _skip_validation:
        result.validate(report, prereg, geometry)
    return result
