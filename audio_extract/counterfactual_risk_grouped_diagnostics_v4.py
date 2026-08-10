"""Attestation-resolved grouped D0/R0 diagnostics v4.

V3 attempted to bind one partition certificate directly to every held-out unit.
That was too strict for registries containing several legitimate resolutions and
it conflated the partition source report with the complete-route risk-evidence
artifact.  V4 derives each route's grid through the content-bearing v2
inference-input/partition attestation.

D0 and R0 must select the same certified partition for a unit.  Extra certified
resolutions may remain in the registry.  Rational cell measure belongs to risk
aggregation; switch normalization counts graph boundaries and is therefore based
only on the selected rectangular grid dimensions.

The artifact is descriptive and cannot promote an arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping
import hashlib
import json
import re

from .counterfactual_risk_cell_partition_v1 import CellPartitionRegistry
from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
    ROUTE_STATUSES,
)
from .counterfactual_risk_grouped_comparison_v3 import GroupedComparisonReportV3
from .counterfactual_risk_grouped_diagnostics_v2 import (
    NormalizedSwitchSummaryV2,
    OUTCOME_ORDER,
    OUTCOME_ORDER_ID,
    StatusDominanceSummaryV2,
)
from .counterfactual_risk_route_partition_attestation_v2 import (
    RoutePartitionAttestationRegistryV2,
)

DIAGNOSTICS_SCHEMA = "audio-extract/d0-r0-grouped-diagnostics/v4"
ENVELOPE_SCHEMA = "audio-extract/d0-r0-certified-diagnostics-envelope/v2"
STATUS = "COMPLETE_NO_PROMOTION_DECISION"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupedDiagnosticsV4Error(ValueError):
    """Resolved route/partition diagnostics are incomplete or inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedDiagnosticsV4Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedDiagnosticsV4Error(
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
        raise GroupedDiagnosticsV4Error(
            f"diagnostic value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _boundary_counts(certificate) -> tuple[int, int]:
    temporal = max(0, certificate.time_cell_count - 1) * certificate.band_count
    frequency = certificate.time_cell_count * max(0, certificate.band_count - 1)
    return temporal, frequency


def _validate_every_submitted_route(
    report: GroupedComparisonReportV3,
    resolved,
) -> None:
    for row in report.work_evaluations:
        if row.evaluation.status not in ROUTE_STATUSES:
            continue
        certificate = resolved[(row.unit.sha256, row.arm_id)]
        temporal = row.evaluation.temporal_switches
        frequency = row.evaluation.frequency_switches
        if type(temporal) is not int or type(frequency) is not int:
            raise GroupedDiagnosticsV4Error(
                "submitted route lacks exact integer switch counts"
            )
        t_boundaries, f_boundaries = _boundary_counts(certificate)
        if temporal > t_boundaries:
            raise GroupedDiagnosticsV4Error(
                f"{row.arm_id} temporal switches exceed selected partition"
            )
        if frequency > f_boundaries:
            raise GroupedDiagnosticsV4Error(
                f"{row.arm_id} frequency switches exceed selected partition"
            )
        output = row.route_provenance.route_artifact.route_output
        expected_shape = (
            certificate.time_cell_count,
            certificate.band_count,
        )
        if output.labels_shape != expected_shape:
            raise GroupedDiagnosticsV4Error(
                "submitted route shape differs from selected partition"
            )


def _switch_summary(arm_id: str, report: GroupedComparisonReportV3, resolved):
    records = tuple(
        row
        for row in report.work_evaluations
        if row.arm_id == arm_id and row.evaluation.status == "ROUTE_SAFE"
    )
    temporal_total = 0
    temporal_boundaries = 0
    frequency_total = 0
    frequency_boundaries = 0
    temporal_rates = []
    frequency_rates = []
    for row in records:
        certificate = resolved[(row.unit.sha256, row.arm_id)]
        t_boundaries, f_boundaries = _boundary_counts(certificate)
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
            raise GroupedDiagnosticsV4Error(
                "oracle availability differs between D0 and R0"
            )
        if d0 not in OUTCOME_ORDER or r0 not in OUTCOME_ORDER:
            raise GroupedDiagnosticsV4Error(
                "unknown complete-route status in dominance summary"
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
        outcome_order_id=OUTCOME_ORDER_ID,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class GroupedComparisonDiagnosticsV4:
    report_sha256: str
    attestation_registry_sha256: str
    verifier_commit: str
    d0_switching: NormalizedSwitchSummaryV2
    r0_switching: NormalizedSwitchSummaryV2
    status_dominance: StatusDominanceSummaryV2
    status: str = STATUS

    def validate(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
        attestations: RoutePartitionAttestationRegistryV2,
    ) -> None:
        report.validate(prereg)
        attestations.validate(report, prereg, partitions)
        if self.status != STATUS:
            raise GroupedDiagnosticsV4Error(
                "diagnostics must not contain a promotion decision"
            )
        _sha(self.report_sha256, "report_sha256")
        _sha(self.attestation_registry_sha256, "attestation_registry_sha256")
        _git(self.verifier_commit, "diagnostics verifier_commit")
        if self.report_sha256 != report.sha256(prereg):
            raise GroupedDiagnosticsV4Error(
                "diagnostics names another grouped report"
            )
        expected_attestations = attestations.sha256(report, prereg, partitions)
        if self.attestation_registry_sha256 != expected_attestations:
            raise GroupedDiagnosticsV4Error(
                "diagnostics names another attestation registry"
            )
        self.d0_switching.validate()
        self.r0_switching.validate()
        self.status_dominance.validate()
        expected = build_grouped_diagnostics_v4(
            report,
            prereg,
            partitions,
            attestations,
            verifier_commit=self.verifier_commit,
            _skip_validation=True,
        )
        if self != expected:
            raise GroupedDiagnosticsV4Error(
                "stored diagnostics differ from route/partition evidence"
            )

    def identity_dict(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
        attestations: RoutePartitionAttestationRegistryV2,
    ) -> dict[str, Any]:
        self.validate(report, prereg, partitions, attestations)
        return {
            "schema": DIAGNOSTICS_SCHEMA,
            "status": self.status,
            "promotion_decision": None,
            "report_sha256": self.report_sha256,
            "attestation_registry_sha256": (
                self.attestation_registry_sha256
            ),
            "verifier_commit": self.verifier_commit,
            "d0_switching": self.d0_switching.identity_dict(),
            "r0_switching": self.r0_switching.identity_dict(),
            "status_dominance": self.status_dominance.identity_dict(),
        }

    def sha256(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
        attestations: RoutePartitionAttestationRegistryV2,
    ) -> str:
        return _mapping_sha(
            self.identity_dict(report, prereg, partitions, attestations)
        )


@dataclass(frozen=True)
class CertifiedGroupedDiagnosticsEnvelopeV2:
    report_sha256: str
    attestation_registry_sha256: str
    diagnostics_sha256: str
    verifier_commit: str
    status: str = STATUS

    def validate(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
        attestations: RoutePartitionAttestationRegistryV2,
        diagnostics: GroupedComparisonDiagnosticsV4,
    ) -> None:
        diagnostics.validate(report, prereg, partitions, attestations)
        if self.status != STATUS:
            raise GroupedDiagnosticsV4Error(
                "certified diagnostics envelope must remain non-promoting"
            )
        for name in (
            "report_sha256",
            "attestation_registry_sha256",
            "diagnostics_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.verifier_commit, "envelope verifier_commit")
        expected = {
            "report_sha256": report.sha256(prereg),
            "attestation_registry_sha256": attestations.sha256(
                report, prereg, partitions
            ),
            "diagnostics_sha256": diagnostics.sha256(
                report, prereg, partitions, attestations
            ),
        }
        drift = [
            name
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise GroupedDiagnosticsV4Error(
                f"certified diagnostics envelope differs: {drift}"
            )

    def identity_dict(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        partitions: CellPartitionRegistry,
        attestations: RoutePartitionAttestationRegistryV2,
        diagnostics: GroupedComparisonDiagnosticsV4,
    ) -> dict[str, Any]:
        self.validate(report, prereg, partitions, attestations, diagnostics)
        return {
            "schema": ENVELOPE_SCHEMA,
            "status": self.status,
            "promotion_decision": None,
            "report_sha256": self.report_sha256,
            "attestation_registry_sha256": (
                self.attestation_registry_sha256
            ),
            "diagnostics_sha256": self.diagnostics_sha256,
            "verifier_commit": self.verifier_commit,
        }


def build_grouped_diagnostics_v4(
    report: GroupedComparisonReportV3,
    prereg: GroupedComparisonPreregistrationV1,
    partitions: CellPartitionRegistry,
    attestations: RoutePartitionAttestationRegistryV2,
    *,
    verifier_commit: str,
    _skip_validation: bool = False,
) -> GroupedComparisonDiagnosticsV4:
    report.validate(prereg)
    attestations.validate(report, prereg, partitions)
    resolved = attestations.resolved_certificates(report, prereg, partitions)
    _validate_every_submitted_route(report, resolved)
    result = GroupedComparisonDiagnosticsV4(
        report_sha256=report.sha256(prereg),
        attestation_registry_sha256=attestations.sha256(
            report, prereg, partitions
        ),
        verifier_commit=verifier_commit,
        d0_switching=_switch_summary("D0", report, resolved),
        r0_switching=_switch_summary("R0", report, resolved),
        status_dominance=_status_dominance(report, prereg),
    )
    if not _skip_validation:
        result.validate(report, prereg, partitions, attestations)
    return result
