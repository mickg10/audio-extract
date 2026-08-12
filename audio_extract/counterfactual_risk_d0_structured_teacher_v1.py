"""Globally structured D0 imitation targets with per-cell behavioral margins.

D0 must imitate the frozen structured route, not a local unary argmin.  For a
small exact-reference grid this module exhaustively certifies the unique global
optimum route and then, for the selected candidate in every cell, measures the
*behavioral* margin: the minimum signed distance ``threshold - exact_risk`` over
the required critical constraints.  A cell is a training target only when every
required exact risk is available (eligible), every required risk is at or below
its frozen threshold (feasible), and the behavioral margin meets the frozen
minimum.  ``READY_FOR_GROUPED_ASSEMBLY`` additionally requires that every cell is
eligible and feasible and that at least one high-margin target exists; anything
else fails closed.

This is a *per-panel* offline teacher contract.  A single panel certifies exactly
one frozen group, so this teacher NEVER claims finite-sample conformal or training
coverage: one panel is one exchangeable group.  Its strongest status is
``READY_FOR_GROUPED_ASSEMBLY``.  Only the grouped training-manifest boundary
(``counterfactual_risk_grouped_training_manifest_v1``) may promote a set of
grouped-assembly-ready panels to a training-ready status after counting the
independent group identities.  It accepts an explicit exact-evidence SHA and must
never be used by the inference API.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .counterfactual_risk_exhaustive_decoder_v1 import (
    ExhaustiveRouteDecisionV1,
    decode_exhaustive_small_grid,
)
from .counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedUpperRiskPanelV1,
)
from .counterfactual_risk_routing_preflight_v1 import (
    FrozenRoutingPolicyV1,
    RoutingPreflightV1,
)

TEACHER_SCHEMA = "audio-extract/d0-structured-teacher/v1"
# The strongest status a per-panel teacher may emit.  A single panel is a single
# exchangeable group, so the teacher must never claim finite-sample conformal or
# training coverage; only the grouped training-manifest boundary promotes a set
# of grouped-assembly-ready panels to a training-ready status.
READY_FOR_GROUPED_ASSEMBLY = "READY_FOR_GROUPED_ASSEMBLY"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ELIGIBLE_STATUSES = {
    READY_FOR_GROUPED_ASSEMBLY,
    "UNAVAILABLE_NO_HIGH_MARGIN_CELLS",
}
_ALLOWED_UNAVAILABLE = {
    "UNAVAILABLE_NO_CONFIDENTLY_FEASIBLE_ROUTE",
    "UNAVAILABLE_NONUNIQUE_ROUTE",
    "UNAVAILABLE_NO_FEASIBLE_GLOBAL_ROUTE",
}


class D0StructuredTeacherError(ValueError):
    """A structured imitation target or behavioral margin certificate is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise D0StructuredTeacherError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
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


def _array_sha(value: np.ndarray, *, component: str, dtype: str) -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    header = {
        "schema": TEACHER_SCHEMA,
        "component": component,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


def _critical_metric_indices(
    panel: PartitionedUpperRiskPanelV1,
    policy: FrozenRoutingPolicyV1,
) -> tuple[tuple[int, float], ...]:
    names = tuple(panel.metric_names)
    result: list[tuple[int, float]] = []
    for name, threshold in policy.critical_thresholds:
        if name not in names:
            raise D0StructuredTeacherError(
                f"critical metric {name!r} is absent from the exact panel"
            )
        result.append((names.index(name), float(threshold)))
    return tuple(result)


def _secondary_metric_indices(
    panel: PartitionedUpperRiskPanelV1,
    policy: FrozenRoutingPolicyV1,
) -> tuple[int, ...]:
    names = tuple(panel.metric_names)
    result: list[int] = []
    for name, _weight in policy.secondary_weights:
        if name not in names:
            raise D0StructuredTeacherError(
                f"secondary metric {name!r} is absent from the exact panel"
            )
        result.append(names.index(name))
    return tuple(result)


@dataclass(frozen=True)
class D0StructuredTeacherV1:
    exact_evidence_sha256: str
    group_family_sha256: str
    panel_sha256: str
    preflight_sha256: str
    policy_sha256: str
    decoder_decision_sha256: str
    status: str
    minimum_structured_margin: float
    inference_row_sha256s: tuple[str, ...]
    labels: np.ndarray | None
    cell_available: np.ndarray | None
    cell_feasible: np.ndarray | None
    behavioral_margin: np.ndarray | None
    target_available: np.ndarray | None
    optimum_objective: float | None
    reason: str

    def validate(self, *, candidate_count: int) -> None:
        for name, value in (
            ("exact_evidence_sha256", self.exact_evidence_sha256),
            ("group_family_sha256", self.group_family_sha256),
            ("panel_sha256", self.panel_sha256),
            ("preflight_sha256", self.preflight_sha256),
            ("policy_sha256", self.policy_sha256),
            ("decoder_decision_sha256", self.decoder_decision_sha256),
        ):
            _sha(value, name)
        margin_threshold = float(self.minimum_structured_margin)
        if not math.isfinite(margin_threshold) or margin_threshold <= 0:
            raise D0StructuredTeacherError(
                "minimum structured margin must be finite and strictly positive"
            )
        if not self.inference_row_sha256s:
            raise D0StructuredTeacherError(
                "structured teacher contains no inference rows"
            )
        for index, value in enumerate(self.inference_row_sha256s):
            _sha(value, f"inference_row_sha256s[{index}]")
        if len(set(self.inference_row_sha256s)) != len(
            self.inference_row_sha256s
        ):
            raise D0StructuredTeacherError(
                "structured teacher repeats an inference row"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise D0StructuredTeacherError(
                "structured teacher reason must be non-empty"
            )
        if isinstance(candidate_count, bool) or not isinstance(
            candidate_count, int
        ) or candidate_count < 2:
            raise D0StructuredTeacherError(
                "candidate_count must be an integer of at least two"
            )

        if self.status in _ELIGIBLE_STATUSES:
            arrays = (
                self.labels,
                self.cell_available,
                self.cell_feasible,
                self.behavioral_margin,
                self.target_available,
            )
            if any(value is None for value in arrays):
                raise D0StructuredTeacherError(
                    "eligible teacher status lacks target arrays"
                )
            labels = np.asarray(self.labels)
            available = np.asarray(self.cell_available)
            feasible = np.asarray(self.cell_feasible)
            margin = np.asarray(self.behavioral_margin, dtype=np.float64)
            target = np.asarray(self.target_available)
            if labels.ndim != 2 or min(labels.shape) < 1:
                raise D0StructuredTeacherError(
                    "teacher labels must be a non-empty two-dimensional grid"
                )
            shape = labels.shape
            if not np.issubdtype(labels.dtype, np.integer):
                raise D0StructuredTeacherError("teacher labels must be integers")
            if np.any(labels < 0) or np.any(labels >= candidate_count):
                raise D0StructuredTeacherError(
                    "teacher labels exceed the candidate axis"
                )
            if len(self.inference_row_sha256s) != labels.size:
                raise D0StructuredTeacherError(
                    "teacher row identity count differs from the label grid"
                )
            for name, value in (
                ("cell_available", available),
                ("cell_feasible", feasible),
                ("target_available", target),
            ):
                if value.shape != shape or value.dtype != np.bool_:
                    raise D0StructuredTeacherError(
                        f"{name} must be a boolean grid matching labels"
                    )
            if margin.shape != shape:
                raise D0StructuredTeacherError(
                    "behavioral margin grid differs from labels"
                )
            if np.any(~np.isfinite(margin)):
                raise D0StructuredTeacherError(
                    "behavioral margins must be finite"
                )
            if self.optimum_objective is None or not math.isfinite(
                float(self.optimum_objective)
            ) or self.optimum_objective < 0:
                raise D0StructuredTeacherError(
                    "eligible teacher lacks a finite optimum objective"
                )
            # A cell is feasible only when it is eligible (available) and every
            # required exact risk is at or below its frozen threshold, i.e. the
            # behavioral margin is non-negative.
            expected_feasible = available & (margin >= 0.0)
            if not np.array_equal(feasible, expected_feasible):
                raise D0StructuredTeacherError(
                    "cell feasibility differs from availability and the behavioral margin"
                )
            expected_target = (
                available & feasible & (margin >= margin_threshold)
            )
            if not np.array_equal(target, expected_target):
                raise D0StructuredTeacherError(
                    "D0 target availability differs from the frozen behavioral-margin rule"
                )
            # READY requires complete eligibility+feasibility of the grid and a
            # satisfiable independent-cell coverage gate (>=1 high-margin cell).
            complete = bool(np.all(available) and np.all(feasible))
            expected_status = (
                READY_FOR_GROUPED_ASSEMBLY
                if complete and np.any(target)
                else "UNAVAILABLE_NO_HIGH_MARGIN_CELLS"
            )
            if self.status != expected_status:
                raise D0StructuredTeacherError(
                    "structured teacher status differs from behavioral eligibility"
                )
        elif self.status in _ALLOWED_UNAVAILABLE:
            if any(
                value is not None
                for value in (
                    self.labels,
                    self.cell_available,
                    self.cell_feasible,
                    self.behavioral_margin,
                    self.target_available,
                    self.optimum_objective,
                )
            ):
                raise D0StructuredTeacherError(
                    "unavailable structured teacher must not expose labels or margins"
                )
        else:
            raise D0StructuredTeacherError(
                f"unknown structured teacher status {self.status!r}"
            )

    def _array_hash(
        self, value: np.ndarray | None, *, component: str, dtype: str
    ) -> str | None:
        if value is None:
            return None
        return _array_sha(np.asarray(value), component=component, dtype=dtype)

    def identity_dict(self, *, candidate_count: int) -> dict[str, Any]:
        self.validate(candidate_count=candidate_count)
        return {
            "schema": TEACHER_SCHEMA,
            "exact_evidence_sha256": self.exact_evidence_sha256,
            "group_family_sha256": self.group_family_sha256,
            "panel_sha256": self.panel_sha256,
            "preflight_sha256": self.preflight_sha256,
            "policy_sha256": self.policy_sha256,
            "decoder_decision_sha256": self.decoder_decision_sha256,
            "status": self.status,
            "minimum_structured_margin": float(
                self.minimum_structured_margin
            ),
            "inference_row_sha256s": list(self.inference_row_sha256s),
            "label_shape": (
                None if self.labels is None else list(np.asarray(self.labels).shape)
            ),
            "labels_sha256": self._array_hash(
                self.labels, component="labels", dtype="<i4"
            ),
            "cell_available_sha256": self._array_hash(
                self.cell_available, component="cell_available", dtype="u1"
            ),
            "cell_feasible_sha256": self._array_hash(
                self.cell_feasible, component="cell_feasible", dtype="u1"
            ),
            "behavioral_margin_sha256": self._array_hash(
                self.behavioral_margin,
                component="behavioral_margin",
                dtype="<f8",
            ),
            "target_available_sha256": self._array_hash(
                self.target_available,
                component="target_available",
                dtype="u1",
            ),
            "optimum_objective": self.optimum_objective,
            "reason": self.reason,
        }

    def sha256(self, *, candidate_count: int) -> str:
        return _mapping_sha(self.identity_dict(candidate_count=candidate_count))

    def training_rows(self) -> tuple[dict[str, Any], ...]:
        """Return only high-margin eligible row labels for the offline D0 arm."""

        if self.labels is None or self.target_available is None:
            return ()
        labels = np.asarray(self.labels).reshape(-1)
        target = np.asarray(self.target_available).reshape(-1)
        margin = np.asarray(self.behavioral_margin).reshape(-1)
        return tuple(
            {
                "offline_inference_row_sha256": row_sha,
                "candidate_index": int(labels[index]),
                "behavioral_margin": float(margin[index]),
            }
            for index, row_sha in enumerate(self.inference_row_sha256s)
            if bool(target[index])
        )


def _unavailable_teacher(
    panel: PartitionedUpperRiskPanelV1,
    preflight: RoutingPreflightV1,
    policy: FrozenRoutingPolicyV1,
    decision: ExhaustiveRouteDecisionV1,
    *,
    exact_evidence_sha256: str,
    minimum_structured_margin: float,
) -> D0StructuredTeacherV1:
    mapping = {
        "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE": (
            "UNAVAILABLE_NO_CONFIDENTLY_FEASIBLE_ROUTE",
            "the exact grid contains a cell without a confidently feasible candidate",
        ),
        "ABSTAIN_NONUNIQUE_OPTIMUM": (
            "UNAVAILABLE_NONUNIQUE_ROUTE",
            "the exact structured route is not unique within the frozen tolerance",
        ),
        "ABSTAIN_NO_FEASIBLE_GLOBAL_ROUTE": (
            "UNAVAILABLE_NO_FEASIBLE_GLOBAL_ROUTE",
            "the exact grid has no globally feasible structured route",
        ),
    }
    if decision.status not in mapping:
        raise D0StructuredTeacherError(
            f"cannot map decoder status {decision.status!r} to a teacher refusal"
        )
    status, reason = mapping[decision.status]
    result = D0StructuredTeacherV1(
        exact_evidence_sha256=exact_evidence_sha256,
        group_family_sha256=panel.group_family_sha256,
        panel_sha256=panel.sha256,
        preflight_sha256=preflight.sha256,
        policy_sha256=policy.sha256,
        decoder_decision_sha256=decision.sha256,
        status=status,
        minimum_structured_margin=minimum_structured_margin,
        inference_row_sha256s=panel.inference_row_sha256s,
        labels=None,
        cell_available=None,
        cell_feasible=None,
        behavioral_margin=None,
        target_available=None,
        optimum_objective=None,
        reason=reason,
    )
    result.validate(candidate_count=len(panel.candidate_slot_sha256s))
    return result


def build_d0_structured_teacher(
    panel: PartitionedUpperRiskPanelV1,
    preflight: RoutingPreflightV1,
    policy: FrozenRoutingPolicyV1,
    *,
    exact_evidence_sha256: str,
    minimum_structured_margin: float,
    maximum_cells: int = 12,
) -> D0StructuredTeacherV1:
    """Build global-route labels and exact per-cell behavioral margins."""

    _sha(exact_evidence_sha256, "exact_evidence_sha256")
    threshold = float(minimum_structured_margin)
    if not math.isfinite(threshold) or threshold <= 0:
        raise D0StructuredTeacherError(
            "minimum_structured_margin must be finite and strictly positive"
        )
    decision = decode_exhaustive_small_grid(
        panel, preflight, policy, maximum_cells=maximum_cells
    )
    if decision.status != "ROUTE":
        return _unavailable_teacher(
            panel,
            preflight,
            policy,
            decision,
            exact_evidence_sha256=exact_evidence_sha256,
            minimum_structured_margin=threshold,
        )
    assert decision.labels is not None
    labels = np.asarray(decision.labels, dtype=np.int32)
    optimum = float(decision.objective)
    candidate_count = len(panel.candidate_slot_sha256s)
    critical = _critical_metric_indices(panel, policy)
    secondary = _secondary_metric_indices(panel, policy)
    upper = np.asarray(panel.upper, dtype=np.float64)
    available_mask = np.asarray(panel.available, dtype=bool)

    shape = labels.shape
    cell_available = np.ones(shape, dtype=bool)
    behavioral_margin = np.zeros(shape, dtype=np.float64)
    for time_index in range(panel.time_cell_count):
        for band_index in range(panel.band_count):
            candidate = int(labels[time_index, band_index])
            # Eligibility: every required critical and (when the policy binds
            # secondary evidence) secondary exact risk must be available.
            required = [metric for metric, _threshold in critical]
            if policy.require_secondary_evidence:
                required.extend(secondary)
            eligible = bool(
                np.all(
                    available_mask[time_index, band_index, candidate, required]
                )
            )
            cell_available[time_index, band_index] = eligible
            # Behavioral margin: the minimum signed distance to the frozen
            # critical thresholds for the selected candidate.  Negative margins
            # remain informative (a threshold was exceeded).
            signed = [
                metric_threshold
                - float(upper[time_index, band_index, candidate, metric])
                for metric, metric_threshold in critical
            ]
            behavioral_margin[time_index, band_index] = min(signed)

    cell_feasible = cell_available & (behavioral_margin >= 0.0)
    target_available = (
        cell_available & cell_feasible & (behavioral_margin >= threshold)
    )
    complete = bool(np.all(cell_available) and np.all(cell_feasible))
    if complete and np.any(target_available):
        status = READY_FOR_GROUPED_ASSEMBLY
        reason = (
            "unique global route; per-panel behavioral-margin targets are ready "
            "for grouped assembly (one exchangeable group; not training-certified "
            "at the panel boundary)"
        )
    else:
        status = "UNAVAILABLE_NO_HIGH_MARGIN_CELLS"
        reason = (
            "unique global route without a behavioral-margin training target"
        )

    for array in (
        labels,
        cell_available,
        cell_feasible,
        behavioral_margin,
        target_available,
    ):
        array.setflags(write=False)
    result = D0StructuredTeacherV1(
        exact_evidence_sha256=exact_evidence_sha256,
        group_family_sha256=panel.group_family_sha256,
        panel_sha256=panel.sha256,
        preflight_sha256=preflight.sha256,
        policy_sha256=policy.sha256,
        decoder_decision_sha256=decision.sha256,
        status=status,
        minimum_structured_margin=threshold,
        inference_row_sha256s=panel.inference_row_sha256s,
        labels=labels,
        cell_available=cell_available,
        cell_feasible=cell_feasible,
        behavioral_margin=behavioral_margin,
        target_available=target_available,
        optimum_objective=optimum,
        reason=reason,
    )
    result.validate(candidate_count=candidate_count)
    return result
