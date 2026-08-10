"""Certified normalized diagnostics for grouped D0/R0 comparison v3.

V1 normalized switching only for safe routes and accepted free-standing geometry
rows.  V2 embeds a content-bearing exact partition certificate per held-out unit,
validates switch feasibility for every submitted route status, and validates all
stored summaries before comparing them with recomputed evidence.

The artifact remains descriptive and non-promoting.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
    ROUTE_STATUSES,
)
from .counterfactual_risk_grouped_comparison_v3 import (
    GroupedComparisonReportV3,
)

PARTITION_CERTIFICATE_SCHEMA = (
    "audio-extract/d0-r0-exact-partition-certificate/v2"
)
GEOMETRY_MANIFEST_SCHEMA = "audio-extract/d0-r0-geometry-manifest/v2"
SWITCH_SUMMARY_SCHEMA = "audio-extract/d0-r0-normalized-switch-summary/v2"
OUTCOME_SCHEMA = "audio-extract/d0-r0-status-dominance-summary/v2"
DIAGNOSTICS_SCHEMA = "audio-extract/d0-r0-grouped-diagnostics/v2"
DIAGNOSTICS_STATUS = "COMPLETE_NO_PROMOTION_DECISION"
OUTCOME_ORDER_ID = "fail_closed_exact_status_order/v1"
OUTCOME_ORDER = {
    "ROUTE_CATASTROPHIC_FALSE_SAFE": 0,
    "ROUTE_INCOMPLETE_REQUIRED_EVIDENCE": 1,
    "ABSTAIN": 2,
    "ROUTE_SAFE": 3,
}
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupedDiagnosticsV2Error(ValueError):
    """The diagnostic artifact is incomplete, unbound, or inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedDiagnosticsV2Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedDiagnosticsV2Error(
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
        raise GroupedDiagnosticsV2Error(
            f"diagnostic value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _count(value: Any, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise GroupedDiagnosticsV2Error(
            f"{name} must be a {qualifier} integer"
        )
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _validate_rate(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
        raise GroupedDiagnosticsV2Error(
            f"{name} must be a finite float in [0,1] or null"
        )
    return value


def _same_number(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=8.0 * math.ulp(max(1.0, abs(expected))),
    )


@dataclass(frozen=True)
class ExactPartitionCertificateV2:
    """Verified exact grid and partition lineage for one held-out unit."""

    preregistration_sha256: str
    unit_sha256: str
    group_family_sha256: str
    exact_panel_sha256: str
    exact_preflight_sha256: str
    exact_oracle_decision_sha256: str
    spectral_grid_sha256: str
    partition_contract_sha256: str
    partition_artifact_sha256: str
    time_cell_count: int
    band_count: int
    source_commit: str
    verifier_commit: str
    status: str = "verified"

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> None:
        prereg.validate()
        unit.validate()
        if self.status != "verified":
            raise GroupedDiagnosticsV2Error(
                "partition certificate status must be verified"
            )
        for name in (
            "preregistration_sha256",
            "unit_sha256",
            "group_family_sha256",
            "exact_panel_sha256",
            "exact_preflight_sha256",
            "exact_oracle_decision_sha256",
            "spectral_grid_sha256",
            "partition_contract_sha256",
            "partition_artifact_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.source_commit, "partition source_commit")
        _git(self.verifier_commit, "partition verifier_commit")
        _count(self.time_cell_count, "time_cell_count", positive=True)
        _count(self.band_count, "band_count", positive=True)
        expected = {
            "preregistration_sha256": prereg.sha256,
            "unit_sha256": unit.sha256,
            "group_family_sha256": unit.group_family_sha256,
            "exact_panel_sha256": unit.exact_panel_sha256,
            "exact_preflight_sha256": unit.exact_preflight_sha256,
            "exact_oracle_decision_sha256": (
                unit.exact_oracle_decision_sha256
            ),
        }
        drift = [
            name for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise GroupedDiagnosticsV2Error(
                f"partition certificate names another exact unit: {drift}"
            )
        if self.source_commit != prereg.source_commit:
            raise GroupedDiagnosticsV2Error(
                "partition certificate source commit differs"
            )

    @property
    def temporal_boundary_count(self) -> int:
        _count(self.time_cell_count, "time_cell_count", positive=True)
        _count(self.band_count, "band_count", positive=True)
        return max(0, self.time_cell_count - 1) * self.band_count

    @property
    def frequency_boundary_count(self) -> int:
        _count(self.time_cell_count, "time_cell_count", positive=True)
        _count(self.band_count, "band_count", positive=True)
        return self.time_cell_count * max(0, self.band_count - 1)

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> dict[str, Any]:
        self.validate(prereg, unit)
        return {
            "schema": PARTITION_CERTIFICATE_SCHEMA,
            **self.__dict__,
            "temporal_boundary_count": self.temporal_boundary_count,
            "frequency_boundary_count": self.frequency_boundary_count,
        }

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg, unit))


@dataclass(frozen=True)
class GeometryManifestV2:
    preregistration_sha256: str
    certificates: tuple[ExactPartitionCertificateV2, ...]
    status: str = "frozen"

    def validate(self, prereg: GroupedComparisonPreregistrationV1) -> None:
        prereg.validate()
        if self.status != "frozen":
            raise GroupedDiagnosticsV2Error(
                "geometry manifest must be frozen"
            )
        _sha(self.preregistration_sha256, "preregistration_sha256")
        if self.preregistration_sha256 != prereg.sha256:
            raise GroupedDiagnosticsV2Error(
                "geometry manifest names another preregistration"
            )
        by_unit = {unit.sha256: unit for unit in prereg.expected_units}
        ordered = tuple(
            sorted(self.certificates, key=lambda row: row.unit_sha256)
        )
        if self.certificates != ordered:
            raise GroupedDiagnosticsV2Error(
                "partition certificates must be in canonical unit order"
            )
        if len({row.unit_sha256 for row in ordered}) != len(ordered):
            raise GroupedDiagnosticsV2Error(
                "geometry manifest repeats a unit"
            )
        if tuple(row.unit_sha256 for row in ordered) != tuple(sorted(by_unit)):
            raise GroupedDiagnosticsV2Error(
                "geometry manifest does not exactly cover held-out units"
            )
        certificate_hashes = []
        for row in ordered:
            row.validate(prereg, by_unit[row.unit_sha256])
            certificate_hashes.append(
                row.sha256(prereg, by_unit[row.unit_sha256])
            )
        if len(certificate_hashes) != len(set(certificate_hashes)):
            raise GroupedDiagnosticsV2Error(
                "geometry manifest reuses one partition certificate"
            )

    def by_unit(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> dict[str, ExactPartitionCertificateV2]:
        self.validate(prereg)
        return {row.unit_sha256: row for row in self.certificates}

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> dict[str, Any]:
        self.validate(prereg)
        by_unit = {unit.sha256: unit for unit in prereg.expected_units}
        return {
            "schema": GEOMETRY_MANIFEST_SCHEMA,
            "status": self.status,
            "preregistration_sha256": self.preregistration_sha256,
            "certificates": [
                row.identity_dict(prereg, by_unit[row.unit_sha256])
                for row in self.certificates
            ],
        }

    def sha256(self, prereg: GroupedComparisonPreregistrationV1) -> str:
        return _mapping_sha(self.identity_dict(prereg))


@dataclass(frozen=True)
class NormalizedSwitchSummaryV2:
    arm_id: str
    safe_unit_count: int
    temporal_switch_total: int
    temporal_boundary_total: int
    temporal_switch_rate: float | None
    mean_unit_temporal_switch_rate: float | None
    frequency_switch_total: int
    frequency_boundary_total: int
    frequency_switch_rate: float | None
    mean_unit_frequency_switch_rate: float | None

    def validate(self) -> None:
        if self.arm_id not in {"D0", "R0"}:
            raise GroupedDiagnosticsV2Error("unknown switch-summary arm")
        for name in (
            "safe_unit_count",
            "temporal_switch_total",
            "temporal_boundary_total",
            "frequency_switch_total",
            "frequency_boundary_total",
        ):
            _count(getattr(self, name), name)
        for name in (
            "temporal_switch_rate",
            "mean_unit_temporal_switch_rate",
            "frequency_switch_rate",
            "mean_unit_frequency_switch_rate",
        ):
            _validate_rate(getattr(self, name), name)
        if self.temporal_switch_total > self.temporal_boundary_total:
            raise GroupedDiagnosticsV2Error(
                "temporal switch total exceeds available boundaries"
            )
        if self.frequency_switch_total > self.frequency_boundary_total:
            raise GroupedDiagnosticsV2Error(
                "frequency switch total exceeds available boundaries"
            )
        expected_temporal = _rate(
            self.temporal_switch_total, self.temporal_boundary_total
        )
        expected_frequency = _rate(
            self.frequency_switch_total, self.frequency_boundary_total
        )
        if not _same_number(self.temporal_switch_rate, expected_temporal):
            raise GroupedDiagnosticsV2Error(
                "temporal switch rate differs from totals"
            )
        if not _same_number(self.frequency_switch_rate, expected_frequency):
            raise GroupedDiagnosticsV2Error(
                "frequency switch rate differs from totals"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": SWITCH_SUMMARY_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class StatusDominanceSummaryV2:
    unit_count: int
    oracle_available_count: int
    d0_status_better_count: int
    r0_status_better_count: int
    equal_status_count: int
    oracle_unavailable_count: int
    outcome_order_id: str = OUTCOME_ORDER_ID

    def validate(self) -> None:
        if self.outcome_order_id != OUTCOME_ORDER_ID:
            raise GroupedDiagnosticsV2Error(
                "unknown outcome-order policy"
            )
        for name in (
            "unit_count",
            "oracle_available_count",
            "d0_status_better_count",
            "r0_status_better_count",
            "equal_status_count",
            "oracle_unavailable_count",
        ):
            _count(getattr(self, name), name)
        if self.oracle_available_count + self.oracle_unavailable_count != (
            self.unit_count
        ):
            raise GroupedDiagnosticsV2Error(
                "status-dominance oracle counts do not sum"
            )
        if (
            self.d0_status_better_count
            + self.r0_status_better_count
            + self.equal_status_count
            != self.oracle_available_count
        ):
            raise GroupedDiagnosticsV2Error(
                "status-dominance outcomes do not cover available units"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": OUTCOME_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class GroupedComparisonDiagnosticsV2:
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
        geometry: GeometryManifestV2,
    ) -> None:
        report.validate(prereg)
        geometry.validate(prereg)
        if self.status != DIAGNOSTICS_STATUS:
            raise GroupedDiagnosticsV2Error(
                "diagnostics must not contain a promotion decision"
            )
        _sha(self.report_sha256, "report_sha256")
        _sha(self.geometry_manifest_sha256, "geometry_manifest_sha256")
        _git(self.verifier_commit, "verifier_commit")
        if self.report_sha256 != report.sha256(prereg):
            raise GroupedDiagnosticsV2Error(
                "diagnostics names another grouped report"
            )
        if self.geometry_manifest_sha256 != geometry.sha256(prereg):
            raise GroupedDiagnosticsV2Error(
                "diagnostics names another geometry manifest"
            )
        self.d0_switching.validate()
        self.r0_switching.validate()
        self.status_dominance.validate()
        expected = build_grouped_diagnostics_v2(
            report,
            prereg,
            geometry,
            verifier_commit=self.verifier_commit,
            _skip_validation=True,
        )
        if self != expected:
            raise GroupedDiagnosticsV2Error(
                "stored diagnostics differ from report/partition evidence"
            )

    def identity_dict(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV2,
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
        geometry: GeometryManifestV2,
    ) -> str:
        return _mapping_sha(self.identity_dict(report, prereg, geometry))


def _validate_every_route_switch_count(
    report: GroupedComparisonReportV3,
    geometry_by_unit: Mapping[str, ExactPartitionCertificateV2],
) -> None:
    for row in report.work_evaluations:
        if row.evaluation.status not in ROUTE_STATUSES:
            continue
        grid = geometry_by_unit[row.unit.sha256]
        temporal = row.evaluation.temporal_switches
        frequency = row.evaluation.frequency_switches
        if type(temporal) is not int or type(frequency) is not int:
            raise GroupedDiagnosticsV2Error(
                "submitted route lacks integer switch counts"
            )
        if temporal > grid.temporal_boundary_count:
            raise GroupedDiagnosticsV2Error(
                f"{row.arm_id} temporal switches exceed unit boundaries"
            )
        if frequency > grid.frequency_boundary_count:
            raise GroupedDiagnosticsV2Error(
                f"{row.arm_id} frequency switches exceed unit boundaries"
            )


def _switch_summary(
    arm_id: str,
    report: GroupedComparisonReportV3,
    geometry_by_unit: Mapping[str, ExactPartitionCertificateV2],
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
        grid = geometry_by_unit[row.unit.sha256]
        temporal = int(row.evaluation.temporal_switches)
        frequency = int(row.evaluation.frequency_switches)
        temporal_total += temporal
        temporal_boundaries += grid.temporal_boundary_count
        frequency_total += frequency
        frequency_boundaries += grid.frequency_boundary_count
        if grid.temporal_boundary_count:
            temporal_rates.append(temporal / grid.temporal_boundary_count)
        if grid.frequency_boundary_count:
            frequency_rates.append(frequency / grid.frequency_boundary_count)
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
            raise GroupedDiagnosticsV2Error(
                "oracle availability differs inside grouped report"
            )
        if d0 not in OUTCOME_ORDER or r0 not in OUTCOME_ORDER:
            raise GroupedDiagnosticsV2Error(
                "unknown outcome in status dominance summary"
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


def build_grouped_diagnostics_v2(
    report: GroupedComparisonReportV3,
    prereg: GroupedComparisonPreregistrationV1,
    geometry: GeometryManifestV2,
    *,
    verifier_commit: str,
    _skip_validation: bool = False,
) -> GroupedComparisonDiagnosticsV2:
    if not _skip_validation:
        report.validate(prereg)
        geometry.validate(prereg)
    _git(verifier_commit, "verifier_commit")
    by_unit = geometry.by_unit(prereg)
    _validate_every_route_switch_count(report, by_unit)
    result = GroupedComparisonDiagnosticsV2(
        report_sha256=report.sha256(prereg),
        geometry_manifest_sha256=geometry.sha256(prereg),
        verifier_commit=verifier_commit,
        d0_switching=_switch_summary("D0", report, by_unit),
        r0_switching=_switch_summary("R0", report, by_unit),
        status_dominance=_status_dominance(report, prereg),
    )
    if not _skip_validation:
        result.validate(report, prereg, geometry)
    return result
