"""Finite abstraction of the binding oracle-routing decision policy.

This module deliberately excludes audio and floating-point metrics. It models
only the final evidence-completeness and promotion state machine after every
concrete artifact, optimizer certificate, metric, and threshold has been
validated.

The abstraction closes two dangerous loopholes:

* fragments from different methods cannot be combined into one promotion;
* sensitivity-only success cannot be relabelled as a primary-resolution pass.

Every required method/resolution cell is present in one closed assignment. A
binding decision is available only when every required cell is valid. Exactly
one method and one primary resolution are preregistered for promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterable, Mapping


METHODS = ("O2", "O3")
RESOLUTIONS = ("primary", "sensitivity_a", "sensitivity_b")
PRIMARY_RESOLUTION = "primary"
SENSITIVITY_RESOLUTIONS = ("sensitivity_a", "sensitivity_b")
CellKey = tuple[str, str]


class AbstractDecision(str, Enum):
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    ACTIONABLE = "ACTIONABLE"
    RESOLUTION_SENSITIVE = "RESOLUTION_SENSITIVE"
    NO_ACTIONABLE_GAP = "NO_ACTIONABLE_GAP"


@dataclass(frozen=True)
class AbstractCell:
    """One method/resolution validation result.

    ``passes`` is meaningful only when ``valid`` is true. Keeping it explicit
    nevertheless lets the exhaustive model include adversarial assignments in
    which stale or malformed evidence carries a truthy gate bit.
    """

    valid: bool
    passes: bool

    def __post_init__(self) -> None:
        if type(self.valid) is not bool or type(self.passes) is not bool:
            raise TypeError("abstract cell fields must be bool")


@dataclass(frozen=True)
class AbstractReport:
    """Closed finite assignment for the architecture-decision state machine."""

    selected_method: str
    cells: Mapping[CellKey, AbstractCell]

    def validate(self) -> None:
        if self.selected_method not in METHODS:
            raise ValueError(f"unknown selected method: {self.selected_method!r}")
        expected = {
            (method, resolution)
            for method in METHODS
            for resolution in RESOLUTIONS
        }
        actual = set(self.cells)
        if actual != expected:
            raise ValueError(
                f"abstract report cell set differs: missing={sorted(expected-actual)}, "
                f"extra={sorted(actual-expected)}"
            )
        if any(
            not isinstance(value, AbstractCell)
            for value in self.cells.values()
        ):
            raise TypeError("every abstract report value must be AbstractCell")


def decide(report: AbstractReport) -> AbstractDecision:
    """Apply the fail-closed, non-cherry-picking decision rule."""

    report.validate()
    if not all(cell.valid for cell in report.cells.values()):
        return AbstractDecision.INVALID_EVIDENCE

    primary = report.cells[(report.selected_method, PRIMARY_RESOLUTION)]
    if primary.passes:
        return AbstractDecision.ACTIONABLE

    if any(
        report.cells[(report.selected_method, resolution)].passes
        for resolution in SENSITIVITY_RESOLUTIONS
    ):
        return AbstractDecision.RESOLUTION_SENSITIVE

    return AbstractDecision.NO_ACTIONABLE_GAP


def cells_from_bits(
    valid_bits: Iterable[bool],
    pass_bits: Iterable[bool],
) -> dict[CellKey, AbstractCell]:
    """Construct one canonical assignment in method-major order."""

    keys = tuple(
        (method, resolution)
        for method in METHODS
        for resolution in RESOLUTIONS
    )
    valid = tuple(valid_bits)
    passes = tuple(pass_bits)
    if len(valid) != len(keys) or len(passes) != len(keys):
        raise ValueError(f"expected {len(keys)} validity and pass bits")
    return {
        key: AbstractCell(bool(valid[index]), bool(passes[index]))
        for index, key in enumerate(keys)
    }


def exhaustive_assignments(selected_method: str) -> Iterable[AbstractReport]:
    """Yield all 2^12 complete finite assignments for one selected method."""

    if selected_method not in METHODS:
        raise ValueError(f"unknown selected method: {selected_method!r}")
    width = len(METHODS) * len(RESOLUTIONS)
    for bits in product((False, True), repeat=2 * width):
        yield AbstractReport(
            selected_method=selected_method,
            cells=cells_from_bits(bits[:width], bits[width:]),
        )
