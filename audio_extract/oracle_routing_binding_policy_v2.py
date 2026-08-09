"""Strict binding policy over the concrete certified-routing evidence rows.

The concrete validator in :mod:`oracle_routing_decision_v2` determines whether
one method/resolution row is complete and whether it passes all audio/artifact
thresholds. This module performs the *cross-row* decision without post-hoc
method or resolution selection.

Exactly one method and one primary resolution are preregistered. Every declared
required method/resolution row must be valid before any binding conclusion. A
pass at a sensitivity resolution is reported explicitly and cannot be promoted
to ``ACTIONABLE_ROUTING_GAP``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .oracle_binding_gate_abstract import (
    AbstractCell,
    AbstractDecision,
    AbstractReport,
    METHODS as ABSTRACT_METHODS,
    PRIMARY_RESOLUTION,
    RESOLUTIONS as ABSTRACT_RESOLUTIONS,
    decide,
)
from .oracle_routing_decision_v2 import (
    DECISION_SCHEMA as LEGACY_DECISION_SCHEMA,
    REPORT_SCHEMA,
    MethodDecision,
    ROUTED_METHODS,
    RoutingGateConfig,
    evaluate_method,
)


POLICY_SCHEMA = "audio-extract/oracle-routing-binding-policy/v2"
DECISION_SCHEMA = "audio-extract/oracle-routing-binding-decision/v3"
DEFAULT_PRIMARY_RESOLUTION = "1.0"
DEFAULT_SENSITIVITY_RESOLUTIONS = ("2.0", "0.5")


@dataclass(frozen=True)
class BindingPolicyConfig:
    schema: str = POLICY_SCHEMA
    task_id: str = "soloist_vs_rest"
    selected_method: str = "O2_global_medoid"
    primary_resolution: str = DEFAULT_PRIMARY_RESOLUTION
    sensitivity_resolutions: tuple[str, ...] = (
        DEFAULT_SENSITIVITY_RESOLUTIONS
    )
    required_methods: tuple[str, ...] = ROUTED_METHODS

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BindingPolicyConfig":
        data = dict(value)
        for name in ("sensitivity_resolutions", "required_methods"):
            if name in data:
                data[name] = tuple(str(item) for item in data[name])
        result = cls(**data)
        result.validate()
        return result

    @property
    def required_resolutions(self) -> tuple[str, ...]:
        return (self.primary_resolution, *self.sensitivity_resolutions)

    def validate(self) -> None:
        if self.schema != POLICY_SCHEMA:
            raise ValueError(f"wrong binding policy schema: {self.schema!r}")
        if not self.task_id:
            raise ValueError("binding policy task_id must be non-empty")
        if self.selected_method not in self.required_methods:
            raise ValueError("selected method must be required")
        if tuple(self.required_methods) != tuple(ROUTED_METHODS):
            raise ValueError(
                f"required methods must equal the frozen set {ROUTED_METHODS}"
            )
        if len(set(self.required_methods)) != len(self.required_methods):
            raise ValueError("required methods must be unique")
        if not self.primary_resolution:
            raise ValueError("primary resolution must be non-empty")
        if not self.sensitivity_resolutions:
            raise ValueError("at least one sensitivity resolution is required")
        if len(set(self.required_resolutions)) != len(
            self.required_resolutions
        ):
            raise ValueError("primary/sensitivity resolutions must be unique")
        if len(self.sensitivity_resolutions) != 2:
            raise ValueError(
                "the v2 finite model requires exactly two sensitivities"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["sensitivity_resolutions"] = list(
            self.sensitivity_resolutions
        )
        result["required_methods"] = list(self.required_methods)
        return result


def _abstract_method(method: str) -> str:
    mapping = {
        "O2_global_medoid": "O2",
        "O3_certified_convex": "O3",
    }
    try:
        result = mapping[method]
    except KeyError as exc:
        raise ValueError(
            f"method has no formal abstraction: {method!r}"
        ) from exc
    if result not in ABSTRACT_METHODS:
        raise AssertionError(result)
    return result


def _abstract_resolution(
    resolution: str,
    policy: BindingPolicyConfig,
) -> str:
    if resolution == policy.primary_resolution:
        return PRIMARY_RESOLUTION
    try:
        index = policy.sensitivity_resolutions.index(resolution)
    except ValueError as exc:
        raise ValueError(
            f"resolution is outside the frozen policy: {resolution!r}"
        ) from exc
    result = ("sensitivity_a", "sensitivity_b")[index]
    if result not in ABSTRACT_RESOLUTIONS:
        raise AssertionError(result)
    return result


def reduce_method_decisions(
    decisions: Sequence[MethodDecision],
    policy: BindingPolicyConfig,
) -> dict[str, Any]:
    """Reduce complete concrete row results through the verified abstraction."""

    policy.validate()
    by_key: dict[tuple[str, str], MethodDecision] = {}
    for row in decisions:
        key = (row.method, row.resolution_seconds)
        if key in by_key:
            raise ValueError(f"duplicate method decision: {key}")
        by_key[key] = row

    expected = {
        (method, resolution)
        for method in policy.required_methods
        for resolution in policy.required_resolutions
    }
    if set(by_key) != expected:
        raise ValueError(
            f"method decision set differs: missing={sorted(expected-set(by_key))}, "
            f"extra={sorted(set(by_key)-expected)}"
        )

    abstract_cells = {
        (_abstract_method(method), _abstract_resolution(resolution, policy)):
            AbstractCell(
                valid=by_key[(method, resolution)].evidence_valid,
                passes=by_key[(method, resolution)].actionable_oracle_gap,
            )
        for method, resolution in expected
    }
    abstract = AbstractReport(
        selected_method=_abstract_method(policy.selected_method),
        cells=abstract_cells,
    )
    state = decide(abstract)
    selected = by_key[(policy.selected_method, policy.primary_resolution)]
    sensitivity_hits = [
        by_key[(policy.selected_method, resolution)].to_dict()
        for resolution in policy.sensitivity_resolutions
        if by_key[(policy.selected_method, resolution)].actionable_oracle_gap
    ]

    if state is AbstractDecision.INVALID_EVIDENCE:
        decision = "INCOMPLETE_EVIDENCE"
        recommendation = "COMPLETE_CERTIFICATES_AND_CONTROLS"
        reason = "one or more frozen method/resolution rows are invalid"
        selected_payload = None
    elif state is AbstractDecision.ACTIONABLE:
        decision = "ACTIONABLE_ROUTING_GAP"
        recommendation = "TRAIN_TARGET_SINGER_FROZEN_MEMBER_GATE"
        reason = (
            f"preregistered {policy.selected_method} passes at the primary "
            f"{policy.primary_resolution}s resolution"
        )
        selected_payload = selected.to_dict()
    elif state is AbstractDecision.RESOLUTION_SENSITIVE:
        decision = "RESOLUTION_SENSITIVE_GAP"
        recommendation = "DO_NOT_PROMOTE__REVIEW_FROZEN_SCALE_SENSITIVITY"
        reason = (
            f"{policy.selected_method} fails at the primary resolution but "
            "passes at one or more frozen sensitivity resolutions"
        )
        selected_payload = None
    elif state is AbstractDecision.NO_ACTIONABLE_GAP:
        decision = "NO_ACTIONABLE_GAP_AT_TESTED_RESOLUTIONS"
        recommendation = "CHANGE_BASIS_OR_BUILD_TARGET_SINGER_CORRECTION"
        reason = (
            f"preregistered {policy.selected_method} passes neither the primary "
            "nor either frozen sensitivity resolution"
        )
        selected_payload = None
    else:  # pragma: no cover
        raise AssertionError(state)

    return {
        "schema": DECISION_SCHEMA,
        "legacy_row_schema": LEGACY_DECISION_SCHEMA,
        "decision": decision,
        "recommendation": recommendation,
        "reason": reason,
        "policy": policy.identity_dict(),
        "selected": selected_payload,
        "sensitivity_hits": sensitivity_hits,
        "method_decisions": [
            by_key[(method, resolution)].to_dict()
            for resolution in policy.required_resolutions
            for method in policy.required_methods
        ],
    }


def evaluate_report_strict(
    report: Mapping[str, Any],
    *,
    policy: BindingPolicyConfig | None = None,
    metric_config: RoutingGateConfig | None = None,
) -> dict[str, Any]:
    """Validate all frozen rows, then apply the non-cherry-picking policy."""

    selected_policy = policy or BindingPolicyConfig()
    selected_policy.validate()
    thresholds = metric_config or RoutingGateConfig()
    thresholds.validate()

    if report.get("schema") != REPORT_SCHEMA:
        return {
            "schema": DECISION_SCHEMA,
            "decision": "INCOMPLETE_EVIDENCE",
            "recommendation": "COMPLETE_CERTIFICATES_AND_CONTROLS",
            "reason": f"wrong report schema: {report.get('schema')!r}",
            "policy": selected_policy.identity_dict(),
            "selected": None,
            "sensitivity_hits": [],
            "method_decisions": [],
        }
    resolutions = report.get("resolutions")
    if not isinstance(resolutions, Mapping):
        return {
            "schema": DECISION_SCHEMA,
            "decision": "INCOMPLETE_EVIDENCE",
            "recommendation": "COMPLETE_CERTIFICATES_AND_CONTROLS",
            "reason": "report lacks a resolution mapping",
            "policy": selected_policy.identity_dict(),
            "selected": None,
            "sensitivity_hits": [],
            "method_decisions": [],
        }

    rows: list[MethodDecision] = []
    for resolution in selected_policy.required_resolutions:
        resolution_row = resolutions.get(resolution)
        works = (
            resolution_row.get("works")
            if isinstance(resolution_row, Mapping)
            else None
        )
        if not isinstance(works, Mapping):
            rows.extend(
                MethodDecision(
                    resolution_seconds=resolution,
                    method=method,
                    evidence_valid=False,
                    actionable_oracle_gap=False,
                    critical_gains_db={},
                    failures=(f"missing required resolution {resolution}",),
                )
                for method in selected_policy.required_methods
            )
            continue
        rows.extend(
            evaluate_method(resolution, works, method, thresholds)
            for method in selected_policy.required_methods
        )

    return reduce_method_decisions(rows, selected_policy)
