"""Strict, preregistered binding policy for certified oracle routing.

The concrete validator in :mod:`oracle_routing_decision_v2` determines whether
one method/resolution row is complete and whether it passes every metric,
artifact, transform, and optimizer threshold.  This module performs the
cross-row decision without post-hoc method or resolution selection.

The binding policy is intentionally *not* configurable at verification time.
Its task, selected method, primary resolution, sensitivity resolutions, and
required method panel are frozen in this source revision and protected by a
hard-coded semantic SHA-256.  A policy JSON file is only a human-readable copy
of that authority; changing whitespace or object-key order is harmless, while
changing any semantic field is refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .oracle_binding_gate_abstract import (
    METHODS as ABSTRACT_METHODS,
)
from .oracle_binding_gate_abstract import (
    PRIMARY_RESOLUTION,
    AbstractCell,
    AbstractDecision,
    AbstractReport,
    decide,
)
from .oracle_binding_gate_abstract import (
    RESOLUTIONS as ABSTRACT_RESOLUTIONS,
)
from .oracle_routing_decision_v2 import (
    DECISION_SCHEMA as LEGACY_DECISION_SCHEMA,
)
from .oracle_routing_decision_v2 import (
    REPORT_SCHEMA,
    ROUTED_METHODS,
    MethodDecision,
    RoutingGateConfig,
    evaluate_method,
)
from .oracle_routing_work_contract_v3 import (
    WORK_CONTRACT_SHA256,
)
from .oracle_routing_work_contract_v3 import (
    identity_dict as work_contract_identity,
)

POLICY_SCHEMA = "audio-extract/oracle-routing-binding-policy/v2"
DECISION_SCHEMA = "audio-extract/oracle-routing-binding-decision/v3"
CANONICAL_TASK_ID = "soloist_vs_rest"
CANONICAL_SELECTED_METHOD = "O2_global_medoid"
CANONICAL_PRIMARY_RESOLUTION = "1.0"
CANONICAL_SENSITIVITY_RESOLUTIONS = ("2.0", "0.5")
CANONICAL_REQUIRED_METHODS = tuple(ROUTED_METHODS)
# SHA-256 of sorted, compact, UTF-8 JSON for BindingPolicyConfig.identity_dict().
CANONICAL_POLICY_SHA256 = (
    "sha256:fa5c4246b858ede0eae4fcab98114abb86d91d36d67428edf78628299c679d65"
)


def _semantic_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def policy_semantic_sha256(value: Mapping[str, Any]) -> str:
    """Hash policy meaning independently of JSON whitespace/key ordering."""

    return "sha256:" + hashlib.sha256(_semantic_json(value)).hexdigest()


@dataclass(frozen=True)
class BindingPolicyConfig:
    schema: str = POLICY_SCHEMA
    task_id: str = CANONICAL_TASK_ID
    selected_method: str = CANONICAL_SELECTED_METHOD
    primary_resolution: str = CANONICAL_PRIMARY_RESOLUTION
    sensitivity_resolutions: tuple[str, ...] = CANONICAL_SENSITIVITY_RESOLUTIONS
    required_methods: tuple[str, ...] = CANONICAL_REQUIRED_METHODS

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BindingPolicyConfig:
        data = dict(value)
        for name in ("sensitivity_resolutions", "required_methods"):
            if name in data:
                raw = data[name]
                if not isinstance(raw, (list, tuple)):
                    raise ValueError(f"{name} must be an array")
                data[name] = tuple(str(item) for item in raw)
        result = cls(**data)
        result.validate()
        if policy_semantic_sha256(result.identity_dict()) != CANONICAL_POLICY_SHA256:
            raise ValueError("binding policy semantic hash is not canonical")
        return result

    @property
    def required_resolutions(self) -> tuple[str, ...]:
        return (self.primary_resolution, *self.sensitivity_resolutions)

    def validate(self) -> None:
        expected = {
            "schema": POLICY_SCHEMA,
            "task_id": CANONICAL_TASK_ID,
            "selected_method": CANONICAL_SELECTED_METHOD,
            "primary_resolution": CANONICAL_PRIMARY_RESOLUTION,
            "sensitivity_resolutions": CANONICAL_SENSITIVITY_RESOLUTIONS,
            "required_methods": CANONICAL_REQUIRED_METHODS,
        }
        actual = {
            "schema": self.schema,
            "task_id": self.task_id,
            "selected_method": self.selected_method,
            "primary_resolution": self.primary_resolution,
            "sensitivity_resolutions": tuple(self.sensitivity_resolutions),
            "required_methods": tuple(self.required_methods),
        }
        differences = {
            name: {"actual": actual[name], "expected": expected[name]}
            for name in expected
            if actual[name] != expected[name]
        }
        if differences:
            raise ValueError(
                f"binding policy differs from preregistration: {differences}"
            )
        if len(set(self.required_resolutions)) != len(self.required_resolutions):
            raise ValueError("primary/sensitivity resolutions must be unique")
        if len(set(self.required_methods)) != len(self.required_methods):
            raise ValueError("required methods must be unique")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["sensitivity_resolutions"] = list(self.sensitivity_resolutions)
        result["required_methods"] = list(self.required_methods)
        return result


def canonical_binding_policy() -> BindingPolicyConfig:
    result = BindingPolicyConfig()
    result.validate()
    digest = policy_semantic_sha256(result.identity_dict())
    if digest != CANONICAL_POLICY_SHA256:
        raise RuntimeError(
            f"compiled binding policy hash drifted: {digest} != "
            f"{CANONICAL_POLICY_SHA256}"
        )
    return result


def _abstract_method(method: str) -> str:
    mapping = {
        "O2_global_medoid": "O2",
        "O3_certified_convex": "O3",
    }
    try:
        result = mapping[method]
    except KeyError as exc:
        raise ValueError(f"method has no formal abstraction: {method!r}") from exc
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
    if policy_semantic_sha256(policy.identity_dict()) != CANONICAL_POLICY_SHA256:
        raise ValueError("noncanonical policy reached decision reduction")
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
            f"method decision set differs: missing={sorted(expected - set(by_key))}, "
            f"extra={sorted(set(by_key) - expected)}"
        )

    abstract_cells = {
        (
            _abstract_method(method),
            _abstract_resolution(resolution, policy),
        ): AbstractCell(
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
        "policy_semantic_sha256": CANONICAL_POLICY_SHA256,
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

    selected_policy = policy or canonical_binding_policy()
    selected_policy.validate()
    if policy_semantic_sha256(selected_policy.identity_dict()) != (
        CANONICAL_POLICY_SHA256
    ):
        raise ValueError("strict evaluation received a noncanonical policy")
    thresholds = metric_config or RoutingGateConfig()
    thresholds.validate()

    if report.get("schema") != REPORT_SCHEMA:
        return {
            "schema": DECISION_SCHEMA,
            "decision": "INCOMPLETE_EVIDENCE",
            "recommendation": "COMPLETE_CERTIFICATES_AND_CONTROLS",
            "reason": f"wrong report schema: {report.get('schema')!r}",
            "policy": selected_policy.identity_dict(),
            "policy_semantic_sha256": CANONICAL_POLICY_SHA256,
            "selected": None,
            "sensitivity_hits": [],
            "method_decisions": [],
        }
    if (
        report.get("work_contract") != work_contract_identity()
        or report.get("work_contract_sha256") != WORK_CONTRACT_SHA256
    ):
        return {
            "schema": DECISION_SCHEMA,
            "decision": "INCOMPLETE_EVIDENCE",
            "recommendation": "COMPLETE_CERTIFICATES_AND_CONTROLS",
            "reason": (
                "report work contract differs from the frozen hall-bearing identity"
            ),
            "policy": selected_policy.identity_dict(),
            "policy_semantic_sha256": CANONICAL_POLICY_SHA256,
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
            "policy_semantic_sha256": CANONICAL_POLICY_SHA256,
            "selected": None,
            "sensitivity_hits": [],
            "method_decisions": [],
        }

    rows: list[MethodDecision] = []
    for resolution in selected_policy.required_resolutions:
        resolution_row = resolutions.get(resolution)
        works = (
            resolution_row.get("works") if isinstance(resolution_row, Mapping) else None
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
