"""Globally structured D0 imitation targets with per-cell alternative-route gaps.

D0 must imitate the frozen structured route, not a local unary argmin.  For a
small exact-reference grid this module exhaustively certifies the unique global
optimum, then re-solves each cell under the constraint that its label must
change.  The resulting objective gap is the structured cell margin.  Cells are
eligible for direct imitation only when the label is forced or the frozen
minimum structured margin is met.

This is an offline teacher contract.  It accepts an explicit exact-evidence SHA
and must never be used by the inference API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import itertools
import json
import math
import re

import numpy as np

from .counterfactual_risk_exhaustive_decoder_v1 import (
    ExhaustiveRouteDecisionV1,
    decode_exhaustive_small_grid,
    recompute_route_objective,
)
from .counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedUpperRiskPanelV1,
)
from .counterfactual_risk_routing_preflight_v1 import (
    FrozenRoutingPolicyV1,
    RoutingPreflightV1,
)

TEACHER_SCHEMA = "audio-extract/d0-structured-teacher/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ALLOWED_UNAVAILABLE = {
    "UNAVAILABLE_NO_CONFIDENTLY_FEASIBLE_ROUTE",
    "UNAVAILABLE_NONUNIQUE_ROUTE",
    "UNAVAILABLE_NO_FEASIBLE_GLOBAL_ROUTE",
}


class D0StructuredTeacherError(ValueError):
    """A structured imitation target or margin certificate is invalid."""


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


def _objective_equal(first: float, second: float, tolerance: float) -> bool:
    return abs(first - second) <= tolerance * max(
        1.0, abs(first), abs(second)
    )


@dataclass(frozen=True)
class D0StructuredTeacherV1:
    exact_evidence_sha256: str
    panel_sha256: str
    preflight_sha256: str
    policy_sha256: str
    decoder_decision_sha256: str
    status: str
    minimum_structured_margin: float
    inference_row_sha256s: tuple[str, ...]
    labels: np.ndarray | None
    target_available: np.ndarray | None
    forced_label: np.ndarray | None
    alternative_available: np.ndarray | None
    alternative_objective: np.ndarray | None
    structured_margin: np.ndarray | None
    optimum_objective: float | None
    reason: str

    def validate(self, *, candidate_count: int) -> None:
        for name, value in (
            ("exact_evidence_sha256", self.exact_evidence_sha256),
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

        ready_statuses = {
            "READY_FOR_D0_TRAINING",
            "UNAVAILABLE_NO_HIGH_MARGIN_CELLS",
        }
        if self.status in ready_statuses:
            arrays = (
                self.labels,
                self.target_available,
                self.forced_label,
                self.alternative_available,
                self.alternative_objective,
                self.structured_margin,
            )
            if any(value is None for value in arrays):
                raise D0StructuredTeacherError(
                    "unique-route teacher status lacks target arrays"
                )
            labels = np.asarray(self.labels)
            target = np.asarray(self.target_available)
            forced = np.asarray(self.forced_label)
            alternative = np.asarray(self.alternative_available)
            alt_objective = np.asarray(
                self.alternative_objective, dtype=np.float64
            )
            margin = np.asarray(self.structured_margin, dtype=np.float64)
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
                ("target_available", target),
                ("forced_label", forced),
                ("alternative_available", alternative),
            ):
                if value.shape != shape or value.dtype != np.bool_:
                    raise D0StructuredTeacherError(
                        f"{name} must be a boolean grid matching labels"
                    )
            if alt_objective.shape != shape or margin.shape != shape:
                raise D0StructuredTeacherError(
                    "teacher objective/margin grids differ from labels"
                )
            if np.any(~np.isfinite(alt_objective)) or np.any(
                ~np.isfinite(margin)
            ) or np.any(alt_objective < 0) or np.any(margin < 0):
                raise D0StructuredTeacherError(
                    "teacher objectives and margins must be finite and non-negative"
                )
            if self.optimum_objective is None or not math.isfinite(
                float(self.optimum_objective)
            ) or self.optimum_objective < 0:
                raise D0StructuredTeacherError(
                    "unique-route teacher lacks a finite optimum objective"
                )
            if np.any(forced & alternative):
                raise D0StructuredTeacherError(
                    "forced labels cannot also have an alternative route"
                )
            if np.any(forced & ~target):
                raise D0StructuredTeacherError(
                    "forced labels must be available D0 targets"
                )
            if np.any(~alternative & (alt_objective != 0.0)) or np.any(
                ~alternative & (margin != 0.0)
            ):
                raise D0StructuredTeacherError(
                    "unavailable alternatives must use canonical zero storage"
                )
            expected_margin = np.where(
                alternative,
                np.maximum(
                    alt_objective - float(self.optimum_objective), 0.0
                ),
                0.0,
            )
            if not np.allclose(
                margin,
                expected_margin,
                rtol=0.0,
                atol=8.0 * np.finfo(np.float64).eps
                * max(1.0, float(self.optimum_objective)),
            ):
                raise D0StructuredTeacherError(
                    "structured margins differ from alternative-objective gaps"
                )
            expected_target = forced | (
                alternative & (margin >= margin_threshold)
            )
            if not np.array_equal(target, expected_target):
                raise D0StructuredTeacherError(
                    "D0 target availability differs from the frozen margin rule"
                )
            expected_status = (
                "READY_FOR_D0_TRAINING"
                if np.any(target)
                else "UNAVAILABLE_NO_HIGH_MARGIN_CELLS"
            )
            if self.status != expected_status:
                raise D0StructuredTeacherError(
                    "structured teacher status differs from target availability"
                )
        elif self.status in _ALLOWED_UNAVAILABLE:
            if any(
                value is not None
                for value in (
                    self.labels,
                    self.target_available,
                    self.forced_label,
                    self.alternative_available,
                    self.alternative_objective,
                    self.structured_margin,
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
            "target_available_sha256": self._array_hash(
                self.target_available,
                component="target_available",
                dtype="u1",
            ),
            "forced_label_sha256": self._array_hash(
                self.forced_label, component="forced_label", dtype="u1"
            ),
            "alternative_available_sha256": self._array_hash(
                self.alternative_available,
                component="alternative_available",
                dtype="u1",
            ),
            "alternative_objective_sha256": self._array_hash(
                self.alternative_objective,
                component="alternative_objective",
                dtype="<f8",
            ),
            "structured_margin_sha256": self._array_hash(
                self.structured_margin,
                component="structured_margin",
                dtype="<f8",
            ),
            "optimum_objective": self.optimum_objective,
            "reason": self.reason,
        }

    def sha256(self, *, candidate_count: int) -> str:
        return _mapping_sha(self.identity_dict(candidate_count=candidate_count))

    def training_rows(self) -> tuple[dict[str, Any], ...]:
        """Return only high-margin/forced row labels for the offline D0 arm."""

        if self.labels is None or self.target_available is None:
            return ()
        labels = np.asarray(self.labels).reshape(-1)
        target = np.asarray(self.target_available).reshape(-1)
        forced = np.asarray(self.forced_label).reshape(-1)
        margin = np.asarray(self.structured_margin).reshape(-1)
        return tuple(
            {
                "offline_inference_row_sha256": row_sha,
                "candidate_index": int(labels[index]),
                "forced_label": bool(forced[index]),
                "structured_margin": float(margin[index]),
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
        panel_sha256=panel.sha256,
        preflight_sha256=preflight.sha256,
        policy_sha256=policy.sha256,
        decoder_decision_sha256=decision.sha256,
        status=status,
        minimum_structured_margin=minimum_structured_margin,
        inference_row_sha256s=panel.inference_row_sha256s,
        labels=None,
        target_available=None,
        forced_label=None,
        alternative_available=None,
        alternative_objective=None,
        structured_margin=None,
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
    """Build global-route labels and exact per-cell alternative-route margins."""

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
    optimum_labels = np.asarray(decision.labels, dtype=np.int32)
    optimum = float(decision.objective)
    feasible = np.asarray(preflight.feasible, dtype=bool)
    cells = panel.time_cell_count * panel.band_count
    candidate_count = len(panel.candidate_slot_sha256s)
    alternatives: list[list[float]] = [[] for _ in range(cells)]

    for assignment in itertools.product(range(candidate_count), repeat=cells):
        labels = np.asarray(assignment, dtype=np.int32).reshape(
            panel.time_cell_count, panel.band_count
        )
        selected = np.take_along_axis(
            feasible, labels[..., None], axis=-1
        )[..., 0]
        if not np.all(selected):
            continue
        total, _, _, _ = recompute_route_objective(
            labels, panel, preflight, policy
        )
        for cell_index, (candidate, optimum_candidate) in enumerate(
            zip(labels.reshape(-1), optimum_labels.reshape(-1))
        ):
            if int(candidate) != int(optimum_candidate):
                alternatives[cell_index].append(total)

    alternative_available = np.asarray(
        [bool(values) for values in alternatives], dtype=bool
    ).reshape(optimum_labels.shape)
    forced = ~alternative_available
    alternative_objective = np.zeros(optimum_labels.shape, dtype=np.float64)
    margin = np.zeros(optimum_labels.shape, dtype=np.float64)
    for index, values in enumerate(alternatives):
        if not values:
            continue
        best = min(values)
        time_index, band_index = divmod(index, panel.band_count)
        alternative_objective[time_index, band_index] = best
        gap = max(0.0, best - optimum)
        margin[time_index, band_index] = gap
        if _objective_equal(best, optimum, policy.objective_tolerance):
            raise D0StructuredTeacherError(
                "unique decoder decision has a zero-gap cell alternative route"
            )
    target_available = forced | (
        alternative_available & (margin >= threshold)
    )
    status = (
        "READY_FOR_D0_TRAINING"
        if np.any(target_available)
        else "UNAVAILABLE_NO_HIGH_MARGIN_CELLS"
    )
    for array in (
        optimum_labels,
        target_available,
        forced,
        alternative_available,
        alternative_objective,
        margin,
    ):
        array.setflags(write=False)
    result = D0StructuredTeacherV1(
        exact_evidence_sha256=exact_evidence_sha256,
        panel_sha256=panel.sha256,
        preflight_sha256=preflight.sha256,
        policy_sha256=policy.sha256,
        decoder_decision_sha256=decision.sha256,
        status=status,
        minimum_structured_margin=threshold,
        inference_row_sha256s=panel.inference_row_sha256s,
        labels=optimum_labels,
        target_available=target_available,
        forced_label=forced,
        alternative_available=alternative_available,
        alternative_objective=alternative_objective,
        structured_margin=margin,
        optimum_objective=optimum,
        reason=(
            "unique global route; D0 labels are filtered by structured alternative-route gaps"
        ),
    )
    result.validate(candidate_count=candidate_count)
    return result
