"""Normalized diagnostics for a completed grouped D0/R0 comparison.

The base grouped report intentionally keeps raw temporal/frequency switch counts
because those are the facts emitted by complete-route evaluation.  Raw counts
are not comparable across works with different routing grids.  This companion
artifact binds exact per-unit grid geometry, reports normalized switching, and
adds a fail-closed status-dominance cross-check.  It still makes no promotion
decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
    GroupedComparisonReportV1,
)

GEOMETRY_SCHEMA = "audio-extract/d0-r0-unit-routing-geometry/v1"
GEOMETRY_MANIFEST_SCHEMA = "audio-extract/d0-r0-geometry-manifest/v1"
SWITCH_SUMMARY_SCHEMA = "audio-extract/d0-r0-normalized-switch-summary/v1"
OUTCOME_SCHEMA = "audio-extract/d0-r0-status-dominance-summary/v1"
DIAGNOSTICS_SCHEMA = "audio-extract/d0-r0-grouped-diagnostics/v1"
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


class GroupedDiagnosticsError(ValueError):
    """The report/grid diagnostics are incomplete or internally inconsistent."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedDiagnosticsError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedDiagnosticsError(
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


def _count(value: Any, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise GroupedDiagnosticsError(f"{name} must be a {qualifier} integer")
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True)
class UnitRoutingGeometryV1:
    unit_sha256: str
    spectral_grid_sha256: str
    partition_certificate_sha256: str
    time_cell_count: int
    band_count: int

    def validate(self) -> None:
        for name in (
            "unit_sha256",
            "spectral_grid_sha256",
            "partition_certificate_sha256",
        ):
            _sha(getattr(self, name), name)
        _count(self.time_cell_count, "time_cell_count", positive=True)
        _count(self.band_count, "band_count", positive=True)

    @property
    def temporal_boundary_count(self) -> int:
        self.validate()
        return max(0, self.time_cell_count - 1) * self.band_count

    @property
    def frequency_boundary_count(self) -> int:
        self.validate()
        return self.time_cell_count * max(0, self.band_count - 1)

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": GEOMETRY_SCHEMA,
            "unit_sha256": self.unit_sha256,
            "spectral_grid_sha256": self.spectral_grid_sha256,
            "partition_certificate_sha256": self.partition_certificate_sha256,
            "time_cell_count": self.time_cell_count,
            "band_count": self.band_count,
            "temporal_boundary_count": self.temporal_boundary_count,
            "frequency_boundary_count": self.frequency_boundary_count,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class GeometryManifestV1:
    preregistration_sha256: str
    units: tuple[UnitRoutingGeometryV1, ...]
    status: str = "frozen"

    def validate(self, prereg: GroupedComparisonPreregistrationV1) -> None:
        prereg.validate()
        if self.status != "frozen":
            raise GroupedDiagnosticsError("geometry manifest must be frozen")
        _sha(self.preregistration_sha256, "preregistration_sha256")
        if self.preregistration_sha256 != prereg.sha256:
            raise GroupedDiagnosticsError(
                "geometry manifest names another preregistration"
            )
        for row in self.units:
            row.validate()
        ordered = tuple(sorted(self.units, key=lambda row: row.unit_sha256))
        if self.units != ordered:
            raise GroupedDiagnosticsError(
                "geometry rows must be in canonical unit order"
            )
        actual = tuple(row.unit_sha256 for row in self.units)
        expected = tuple(sorted(unit.sha256 for unit in prereg.expected_units))
        if len(set(actual)) != len(actual):
            raise GroupedDiagnosticsError("geometry manifest contains duplicates")
        if actual != expected:
            raise GroupedDiagnosticsError(
                "geometry manifest does not exactly cover held-out units"
            )

    def identity_dict(
        self, prereg: GroupedComparisonPreregistrationV1
    ) -> dict[str, Any]:
        self.validate(prereg)
        return {
            "schema": GEOMETRY_MANIFEST_SCHEMA,
            "status": self.status,
            "preregistration_sha256": self.preregistration_sha256,
            "units": [row.identity_dict() for row in self.units],
        }

    def sha256(self, prereg: GroupedComparisonPreregistrationV1) -> str:
        return _mapping_sha(self.identity_dict(prereg))


@dataclass(frozen=True)
class NormalizedSwitchSummaryV1:
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
            raise GroupedDiagnosticsError("unknown switch-summary arm")
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
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            ):
                raise GroupedDiagnosticsError(f"{name} must lie in [0,1]")
        if self.temporal_switch_total > self.temporal_boundary_total:
            raise GroupedDiagnosticsError(
                "temporal switch total exceeds available boundaries"
            )
        if self.frequency_switch_total > self.frequency_boundary_total:
            raise GroupedDiagnosticsError(
                "frequency switch total exceeds available boundaries"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": SWITCH_SUMMARY_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class StatusDominanceSummaryV1:
    unit_count: int
    oracle_available_count: int
    d0_status_better_count: int
    r0_status_better_count: int
    equal_status_count: int
    oracle_unavailable_count: int
    outcome_order_id: str = OUTCOME_ORDER_ID

    def validate(self) -> None:
        if self.outcome_order_id != OUTCOME_ORDER_ID:
            raise GroupedDiagnosticsError("unknown outcome-order policy")
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
            raise GroupedDiagnosticsError(
                "status-dominance oracle counts do not sum"
            )
        if (
            self.d0_status_better_count
            + self.r0_status_better_count
            + self.equal_status_count
            != self.oracle_available_count
        ):
            raise GroupedDiagnosticsError(
                "status-dominance outcomes do not cover available units"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema": OUTCOME_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class GroupedComparisonDiagnosticsV1:
    report_sha256: str
    geometry_manifest_sha256: str
    verifier_commit: str
    d0_switching: NormalizedSwitchSummaryV1
    r0_switching: NormalizedSwitchSummaryV1
    status_dominance: StatusDominanceSummaryV1
    status: str = DIAGNOSTICS_STATUS

    def validate(
        self,
        report: GroupedComparisonReportV1,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV1,
    ) -> None:
        report.validate(prereg)
        geometry.validate(prereg)
        if self.status != DIAGNOSTICS_STATUS:
            raise GroupedDiagnosticsError(
                "diagnostics must not contain a promotion decision"
            )
        _sha(self.report_sha256, "report_sha256")
        _sha(self.geometry_manifest_sha256, "geometry_manifest_sha256")
        _git(self.verifier_commit, "verifier_commit")
        if self.report_sha256 != report.sha256(prereg):
            raise GroupedDiagnosticsError("diagnostics names another report")
        if self.geometry_manifest_sha256 != geometry.sha256(prereg):
            raise GroupedDiagnosticsError(
                "diagnostics names another geometry manifest"
            )
        expected = build_grouped_diagnostics_v1(
            report,
            prereg,
            geometry,
            verifier_commit=self.verifier_commit,
            _skip_validation=True,
        )
        if self != expected:
            raise GroupedDiagnosticsError(
                "stored diagnostics differ from report/grid evidence"
            )

    def identity_dict(
        self,
        report: GroupedComparisonReportV1,
        prereg: GroupedComparisonPreregistrationV1,
        geometry: GeometryManifestV1,
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


def _switch_summary(
    arm_id: str,
    report: GroupedComparisonReportV1,
    geometry_by_unit: Mapping[str, UnitRoutingGeometryV1],
) -> NormalizedSwitchSummaryV1:
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
        grid = geometry_by_unit[row.unit.sha256]
        temporal = int(row.evaluation.temporal_switches)
        frequency = int(row.evaluation.frequency_switches)
        if temporal > grid.temporal_boundary_count:
            raise GroupedDiagnosticsError(
                f"{arm_id} temporal switches exceed unit boundaries"
            )
        if frequency > grid.frequency_boundary_count:
            raise GroupedDiagnosticsError(
                f"{arm_id} frequency switches exceed unit boundaries"
            )
        temporal_total += temporal
        temporal_boundaries += grid.temporal_boundary_count
        frequency_total += frequency
        frequency_boundaries += grid.frequency_boundary_count
        if grid.temporal_boundary_count:
            temporal_rates.append(temporal / grid.temporal_boundary_count)
        if grid.frequency_boundary_count:
            frequency_rates.append(frequency / grid.frequency_boundary_count)
    result = NormalizedSwitchSummaryV1(
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
    report: GroupedComparisonReportV1,
    prereg: GroupedComparisonPreregistrationV1,
) -> StatusDominanceSummaryV1:
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
            raise GroupedDiagnosticsError(
                "oracle availability differs inside completed report"
            )
        if d0 not in OUTCOME_ORDER or r0 not in OUTCOME_ORDER:
            raise GroupedDiagnosticsError("unknown outcome in dominance summary")
        if OUTCOME_ORDER[d0] > OUTCOME_ORDER[r0]:
            d0_better += 1
        elif OUTCOME_ORDER[r0] > OUTCOME_ORDER[d0]:
            r0_better += 1
        else:
            equal += 1
    result = StatusDominanceSummaryV1(
        unit_count=len(prereg.expected_units),
        oracle_available_count=len(prereg.expected_units) - unavailable,
        d0_status_better_count=d0_better,
        r0_status_better_count=r0_better,
        equal_status_count=equal,
        oracle_unavailable_count=unavailable,
    )
    result.validate()
    return result


def build_grouped_diagnostics_v1(
    report: GroupedComparisonReportV1,
    prereg: GroupedComparisonPreregistrationV1,
    geometry: GeometryManifestV1,
    *,
    verifier_commit: str,
    _skip_validation: bool = False,
) -> GroupedComparisonDiagnosticsV1:
    report.validate(prereg)
    geometry.validate(prereg)
    _git(verifier_commit, "verifier_commit")
    by_unit = {row.unit_sha256: row for row in geometry.units}
    result = GroupedComparisonDiagnosticsV1(
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
