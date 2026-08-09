"""Complete exact-cell partitions and rational physical measures for D0/R0.

A grouped dataset must not be a cherry-picked bag of cells.  This module binds
one or more complete rectangular exact grids per source family, proves dense
row-major time/band coverage, carries canonical positive rational cell measure,
and verifies that the dataset contains exactly the certified cells—no omissions,
extras, or silent uniform reweighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

from .counterfactual_risk_dataset_contract_v2 import (
    CounterfactualRiskDatasetV2Error,
    DatasetManifestV2,
)

CERTIFICATE_SCHEMA = "audio-extract/counterfactual-cell-partition/v1"
REGISTRY_SCHEMA = "audio-extract/counterfactual-cell-partition-registry/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class CellPartitionError(CounterfactualRiskDatasetV2Error):
    """A cell partition, physical measure, or dataset binding is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise CellPartitionError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise CellPartitionError(
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


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CellPartitionError(f"{name} must be an integer, not boolean")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise CellPartitionError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True, order=True)
class RationalMeasure:
    numerator: int
    denominator: int = 1

    def validate(self) -> None:
        numerator = _positive_int(self.numerator, "measure numerator")
        denominator = _positive_int(self.denominator, "measure denominator")
        reduced = Fraction(numerator, denominator)
        if (
            reduced.numerator != numerator
            or reduced.denominator != denominator
        ):
            raise CellPartitionError(
                "cell measure must use a positive reduced rational representation"
            )

    @property
    def fraction(self) -> Fraction:
        self.validate()
        return Fraction(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
        }

    @classmethod
    def from_fraction(cls, value: Fraction) -> "RationalMeasure":
        if value <= 0:
            raise CellPartitionError("cell measure fraction must be positive")
        return cls(value.numerator, value.denominator)


@dataclass(frozen=True)
class CellPartitionEntry:
    time_index: int
    band_index: int
    cell_sha256: str
    measure: RationalMeasure

    def validate(self) -> None:
        _positive_int(self.time_index, "time_index", allow_zero=True)
        _positive_int(self.band_index, "band_index", allow_zero=True)
        _sha(self.cell_sha256, "cell_sha256")
        self.measure.validate()

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "time_index": self.time_index,
            "band_index": self.band_index,
            "cell_sha256": self.cell_sha256,
            "measure": self.measure.to_dict(),
        }


@dataclass(frozen=True)
class CellPartitionCertificate:
    group_family_sha256: str
    spectral_grid_sha256: str
    resolution_ms: int
    time_cell_count: int
    band_count: int
    entries: tuple[CellPartitionEntry, ...]
    time_ranges_sha256: str
    frequency_ranges_sha256: str
    measure_contract_sha256: str
    exact_source_report_sha256: str
    verifier_commit: str
    status: str = "passed"

    @property
    def partition_key(self) -> tuple[str, str, int]:
        return (
            self.group_family_sha256,
            self.spectral_grid_sha256,
            self.resolution_ms,
        )

    def validate(self) -> None:
        if self.status != "passed":
            raise CellPartitionError(
                "cell partition certificate status must be passed"
            )
        _sha(self.group_family_sha256, "group_family_sha256")
        _sha(self.spectral_grid_sha256, "spectral_grid_sha256")
        _positive_int(self.resolution_ms, "resolution_ms")
        time_count = _positive_int(self.time_cell_count, "time_cell_count")
        band_count = _positive_int(self.band_count, "band_count")
        expected_count = time_count * band_count
        if len(self.entries) != expected_count:
            raise CellPartitionError(
                "cell partition entry count differs from the rectangular grid"
            )
        for entry in self.entries:
            entry.validate()
        expected_indices = tuple(
            (time_index, band_index)
            for time_index in range(time_count)
            for band_index in range(band_count)
        )
        observed_indices = tuple(
            (entry.time_index, entry.band_index) for entry in self.entries
        )
        if observed_indices != expected_indices:
            raise CellPartitionError(
                "cell entries must provide dense canonical row-major coverage"
            )
        cell_ids = tuple(entry.cell_sha256 for entry in self.entries)
        if len(set(cell_ids)) != len(cell_ids):
            raise CellPartitionError("cell partition contains duplicate cell IDs")
        if self.total_measure <= 0:
            raise CellPartitionError("cell partition total measure is not positive")
        for name, value in (
            ("time_ranges_sha256", self.time_ranges_sha256),
            ("frequency_ranges_sha256", self.frequency_ranges_sha256),
            ("measure_contract_sha256", self.measure_contract_sha256),
            ("exact_source_report_sha256", self.exact_source_report_sha256),
        ):
            _sha(value, name)
        _git(self.verifier_commit, "verifier_commit")

    @property
    def total_measure(self) -> Fraction:
        return sum(
            (entry.measure.fraction for entry in self.entries),
            start=Fraction(0, 1),
        )

    def normalized_measure(self, cell_sha256: str) -> Fraction:
        self.validate()
        for entry in self.entries:
            if entry.cell_sha256 == cell_sha256:
                return entry.measure.fraction / self.total_measure
        raise CellPartitionError(
            "requested cell is absent from the certified partition"
        )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        total = self.total_measure
        return {
            "schema": CERTIFICATE_SCHEMA,
            "status": self.status,
            "group_family_sha256": self.group_family_sha256,
            "spectral_grid_sha256": self.spectral_grid_sha256,
            "resolution_ms": self.resolution_ms,
            "time_cell_count": self.time_cell_count,
            "band_count": self.band_count,
            "entries": [entry.identity_dict() for entry in self.entries],
            "total_measure": {
                "numerator": total.numerator,
                "denominator": total.denominator,
            },
            "time_ranges_sha256": self.time_ranges_sha256,
            "frequency_ranges_sha256": self.frequency_ranges_sha256,
            "measure_contract_sha256": self.measure_contract_sha256,
            "exact_source_report_sha256": self.exact_source_report_sha256,
            "verifier_commit": self.verifier_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class CellPartitionRegistry:
    certificates: tuple[CellPartitionCertificate, ...]
    source_commit: str

    @classmethod
    def build(
        cls,
        certificates: Sequence[CellPartitionCertificate],
        *,
        source_commit: str,
    ) -> "CellPartitionRegistry":
        ordered = tuple(
            sorted(tuple(certificates), key=lambda value: value.partition_key)
        )
        result = cls(ordered, source_commit)
        result.validate()
        return result

    def validate(self) -> None:
        _git(self.source_commit, "partition registry source_commit")
        if not self.certificates:
            raise CellPartitionError("cell partition registry is empty")
        for certificate in self.certificates:
            certificate.validate()
        keys = tuple(certificate.partition_key for certificate in self.certificates)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise CellPartitionError(
                "cell partition keys must be unique and canonically ordered"
            )
        cell_owners: dict[str, tuple[str, str, int]] = {}
        for certificate in self.certificates:
            for entry in certificate.entries:
                previous = cell_owners.setdefault(
                    entry.cell_sha256, certificate.partition_key
                )
                if previous != certificate.partition_key:
                    raise CellPartitionError(
                        "one cell identity appears in multiple certified partitions"
                    )

    def by_key(self) -> dict[tuple[str, str, int], CellPartitionCertificate]:
        self.validate()
        return {
            certificate.partition_key: certificate
            for certificate in self.certificates
        }

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


def validate_dataset_cell_partitions(
    dataset: DatasetManifestV2,
    registry: CellPartitionRegistry,
) -> None:
    """Require exact equality between dataset cells and certified partitions."""

    dataset.validate()
    registry.validate()
    certificates = registry.by_key()
    observed: dict[tuple[str, str, int], list[str]] = {
        key: [] for key in certificates
    }
    dataset_groups = set(dataset.group_family_sha256s)
    for row in dataset.rows:
        key = (
            row.group_family.sha256,
            row.cell.spectral_grid_sha256,
            row.cell.resolution_ms,
        )
        certificate = certificates.get(key)
        if certificate is None:
            raise CellPartitionError(
                "dataset row has no matching certified cell partition"
            )
        cell_id = row.cell.sha256
        if cell_id not in {entry.cell_sha256 for entry in certificate.entries}:
            raise CellPartitionError(
                "dataset row cell is absent from its certified partition"
            )
        observed[key].append(cell_id)

    for key, certificate in certificates.items():
        if certificate.group_family_sha256 not in dataset_groups:
            raise CellPartitionError(
                "partition registry contains a group absent from the dataset"
            )
        expected = tuple(entry.cell_sha256 for entry in certificate.entries)
        actual = tuple(observed[key])
        if len(set(actual)) != len(actual):
            raise CellPartitionError(
                "dataset repeats a cell from one certified partition"
            )
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise CellPartitionError(
                "dataset does not exactly cover a certified partition; "
                f"missing={missing}, extra={extra}"
            )


def normalized_row_measures(
    dataset: DatasetManifestV2,
    registry: CellPartitionRegistry,
) -> dict[str, Fraction]:
    """Return exact normalized physical measure for every dataset row."""

    validate_dataset_cell_partitions(dataset, registry)
    certificates = registry.by_key()
    result: dict[str, Fraction] = {}
    for row in dataset.rows:
        key = (
            row.group_family.sha256,
            row.cell.spectral_grid_sha256,
            row.cell.resolution_ms,
        )
        result[row.row_id] = certificates[key].normalized_measure(row.cell.sha256)
    return result
