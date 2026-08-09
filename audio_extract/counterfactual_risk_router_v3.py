"""Identity-complete, certificate-verified counterfactual-risk router v3.

This dormant research component accepts only inference-available, calibrated
upper defect risks for a frozen candidate bank. It returns either a uniquely
certified Potts route shared by both stereo channels or an exact raw-parent
bypass whose lineage is verified by a content-bearing certificate.

V3 strengthens the earlier prototypes:

* predictor and query encoders are structured weight/config/adapter bundles;
* target-singer query identity is mandatory;
* calibration is a passed, metric-specific certificate bound to the predictor,
  feature contract, candidate panel, policy, split, and empirical lower coverage;
* the conservative parent certificate binds the exact source, parent recipe,
  parent PCM, and no-transform raw-copy semantics;
* MILP certification requires an effectively zero returned gap;
* primary and alternate solutions undergo the same integrality, feasibility,
  and discrete-objective recomputation before uniqueness is claimed;
* an exhaustive small-grid mirror is available for solver-independent tests.

Clean accompaniment, vocal truth, and oracle route labels are not arguments to
any inference function in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

import numpy as np

PANEL_SCHEMA = "audio-extract/counterfactual-risk-panel/v3"
ROUTER_SCHEMA = "audio-extract/counterfactual-risk-router/v3"
ROUTE_SCHEMA = "audio-extract/counterfactual-risk-route/v3"
BUNDLE_SCHEMA = "audio-extract/verified-model-bundle/v1"
CALIBRATION_SCHEMA = "audio-extract/risk-calibration-certificate/v1"
PARENT_SCHEMA = "audio-extract/raw-parent-bypass-certificate/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class RiskRouterV3Error(RuntimeError):
    """The identity, risk panel, or structured certificate is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be canonical sha256:<64 lowercase hex>")
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be 40 lowercase hexadecimal characters")
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


@dataclass(frozen=True)
class CandidateIdentityV3:
    recipe_id: str
    artifact_pcm_sha256: str

    def validate(self) -> None:
        _sha(self.recipe_id, "candidate recipe_id")
        _sha(self.artifact_pcm_sha256, "candidate artifact_pcm_sha256")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "recipe_id": self.recipe_id,
            "artifact_pcm_sha256": self.artifact_pcm_sha256,
        }


@dataclass(frozen=True)
class VerifiedModelBundleV3:
    """Content-bearing model identity rather than an opaque bundle hash."""

    weights_sha256: str
    config_sha256: str
    adapter_sha256: str
    code_commit: str
    status: str = "verified"

    def validate(self) -> None:
        if self.status != "verified":
            raise ValueError("model bundle status must be verified")
        _sha(self.weights_sha256, "model weights SHA")
        _sha(self.config_sha256, "model config SHA")
        _sha(self.adapter_sha256, "model adapter SHA")
        _git(self.code_commit, "model code commit")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": BUNDLE_SCHEMA,
            "status": self.status,
            "weights_sha256": self.weights_sha256,
            "config_sha256": self.config_sha256,
            "adapter_sha256": self.adapter_sha256,
            "code_commit": self.code_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


def candidate_panel_sha256(
    candidates: Sequence[CandidateIdentityV3],
) -> str:
    if len(candidates) < 2:
        raise ValueError("candidate panel requires at least two candidates")
    rows = []
    seen = set()
    for candidate in candidates:
        candidate.validate()
        pair = (candidate.recipe_id, candidate.artifact_pcm_sha256)
        if pair in seen:
            raise ValueError("candidate panel contains a duplicate identity")
        seen.add(pair)
        rows.append(candidate.to_dict())
    return _mapping_sha({
        "schema": "audio-extract/counterfactual-candidate-panel/v1",
        "candidates": rows,
    })


@dataclass(frozen=True)
class MetricCoverageV3:
    metric: str
    target_coverage: float
    lower_confidence_bound: float
    independent_groups: int

    def validate(self) -> None:
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("calibration metric must be a non-empty string")
        for name in ("target_coverage", "lower_confidence_bound"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0,1]")
        if self.lower_confidence_bound < self.target_coverage:
            raise ValueError(
                f"calibration lower bound fails target for {self.metric}"
            )
        if (
            isinstance(self.independent_groups, bool)
            or not isinstance(self.independent_groups, int)
            or self.independent_groups < 1
        ):
            raise ValueError("independent_groups must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "metric": self.metric,
            "target_coverage": float(self.target_coverage),
            "lower_confidence_bound": float(self.lower_confidence_bound),
            "independent_groups": int(self.independent_groups),
        }


@dataclass(frozen=True)
class CalibrationCertificateV3:
    """Passed split/group calibration bound to every semantic dependency."""

    risk_model_bundle_sha256: str
    feature_contract_sha256: str
    candidate_panel_sha256: str
    calibration_policy_sha256: str
    calibration_data_manifest_sha256: str
    split_manifest_sha256: str
    metric_coverages: tuple[MetricCoverageV3, ...]
    status: str = "passed"

    def validate(
        self,
        *,
        risk_model_bundle_sha256: str,
        feature_contract_sha256: str,
        candidate_panel_sha256_value: str,
        calibration_policy_sha256: str,
        metric_names: Sequence[str],
    ) -> None:
        if self.status != "passed":
            raise ValueError("calibration certificate status must be passed")
        bindings = {
            "risk_model_bundle_sha256": (
                self.risk_model_bundle_sha256,
                risk_model_bundle_sha256,
            ),
            "feature_contract_sha256": (
                self.feature_contract_sha256,
                feature_contract_sha256,
            ),
            "candidate_panel_sha256": (
                self.candidate_panel_sha256,
                candidate_panel_sha256_value,
            ),
            "calibration_policy_sha256": (
                self.calibration_policy_sha256,
                calibration_policy_sha256,
            ),
        }
        for name, (actual, expected) in bindings.items():
            _sha(actual, f"calibration {name}")
            _sha(expected, f"expected {name}")
            if actual != expected:
                raise ValueError(f"calibration certificate {name} mismatch")
        _sha(
            self.calibration_data_manifest_sha256,
            "calibration data manifest SHA",
        )
        _sha(self.split_manifest_sha256, "calibration split manifest SHA")
        if not self.metric_coverages:
            raise ValueError("calibration certificate has no metric coverage")
        rows = {}
        for row in self.metric_coverages:
            row.validate()
            if row.metric in rows:
                raise ValueError("duplicate calibration metric")
            rows[row.metric] = row
        if set(rows) != set(metric_names):
            raise ValueError(
                "calibration coverage must exactly match panel metrics"
            )

    def identity_dict(self) -> dict[str, Any]:
        if not self.metric_coverages:
            raise ValueError("calibration certificate has no metric coverage")
        return {
            "schema": CALIBRATION_SCHEMA,
            "status": self.status,
            "risk_model_bundle_sha256": self.risk_model_bundle_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "calibration_policy_sha256": self.calibration_policy_sha256,
            "calibration_data_manifest_sha256": (
                self.calibration_data_manifest_sha256
            ),
            "split_manifest_sha256": self.split_manifest_sha256,
            "metric_coverages": [
                row.to_dict() for row in self.metric_coverages
            ],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class RawParentBypassCertificateV3:
    """Proof that abstention returns one frozen candidate without transforms."""

    source_pcm_sha256: str
    parent: CandidateIdentityV3
    parent_recipe_input_pcm_sha256: str
    parent_recipe_id: str
    parent_artifact_pcm_sha256: str
    raw_candidate_copy_verified: bool
    no_transform_after_candidate: bool
    verifier_commit: str
    status: str = "passed"

    def validate(
        self,
        *,
        source_pcm_sha256: str,
        parent: CandidateIdentityV3,
    ) -> None:
        if self.status != "passed":
            raise ValueError("parent bypass certificate status must be passed")
        _sha(self.source_pcm_sha256, "parent certificate source PCM")
        _sha(source_pcm_sha256, "expected source PCM")
        if self.source_pcm_sha256 != source_pcm_sha256:
            raise ValueError("parent certificate source PCM mismatch")
        self.parent.validate()
        parent.validate()
        if self.parent != parent:
            raise ValueError("parent certificate candidate mismatch")
        _sha(
            self.parent_recipe_input_pcm_sha256,
            "parent recipe input PCM",
        )
        if self.parent_recipe_input_pcm_sha256 != source_pcm_sha256:
            raise ValueError("parent recipe does not name the exact source PCM")
        _sha(self.parent_recipe_id, "parent certificate recipe ID")
        if self.parent_recipe_id != parent.recipe_id:
            raise ValueError("parent certificate recipe ID mismatch")
        _sha(
            self.parent_artifact_pcm_sha256,
            "parent certificate artifact PCM",
        )
        if self.parent_artifact_pcm_sha256 != parent.artifact_pcm_sha256:
            raise ValueError("parent certificate artifact PCM mismatch")
        if self.raw_candidate_copy_verified is not True:
            raise ValueError("raw candidate copy was not verified")
        if self.no_transform_after_candidate is not True:
            raise ValueError("parent bypass contains a transform")
        _git(self.verifier_commit, "parent certificate verifier commit")

    def identity_dict(self) -> dict[str, Any]:
        self.parent.validate()
        return {
            "schema": PARENT_SCHEMA,
            "status": self.status,
            "source_pcm_sha256": self.source_pcm_sha256,
            "parent": self.parent.to_dict(),
            "parent_recipe_input_pcm_sha256": (
                self.parent_recipe_input_pcm_sha256
            ),
            "parent_recipe_id": self.parent_recipe_id,
            "parent_artifact_pcm_sha256": (
                self.parent_artifact_pcm_sha256
            ),
            "raw_candidate_copy_verified": self.raw_candidate_copy_verified,
            "no_transform_after_candidate": self.no_transform_after_candidate,
            "verifier_commit": self.verifier_commit,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class RiskPanelIdentityV3:
    source_pcm_sha256: str
    candidates: tuple[CandidateIdentityV3, ...]
    metric_names: tuple[str, ...]
    feature_contract_sha256: str
    risk_model_bundle: VerifiedModelBundleV3
    calibration_policy_sha256: str
    calibration_certificate: CalibrationCertificateV3
    conservative_parent: CandidateIdentityV3
    conservative_parent_certificate: RawParentBypassCertificateV3
    query_encoder_bundle: VerifiedModelBundleV3
    query_condition_sha256: str

    def validate(self) -> None:
        _sha(self.source_pcm_sha256, "source_pcm_sha256")
        panel_sha = candidate_panel_sha256(self.candidates)
        if not self.metric_names or any(
            not isinstance(name, str) or not name for name in self.metric_names
        ):
            raise ValueError("metric names must be non-empty strings")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric names must be unique and ordered")
        _sha(self.feature_contract_sha256, "feature_contract_sha256")
        self.risk_model_bundle.validate()
        _sha(self.calibration_policy_sha256, "calibration_policy_sha256")
        self.calibration_certificate.validate(
            risk_model_bundle_sha256=self.risk_model_bundle.sha256,
            feature_contract_sha256=self.feature_contract_sha256,
            candidate_panel_sha256_value=panel_sha,
            calibration_policy_sha256=self.calibration_policy_sha256,
            metric_names=self.metric_names,
        )
        self.conservative_parent.validate()
        if self.conservative_parent not in self.candidates:
            raise ValueError(
                "conservative parent is not in the frozen candidate bank"
            )
        self.conservative_parent_certificate.validate(
            source_pcm_sha256=self.source_pcm_sha256,
            parent=self.conservative_parent,
        )
        if not isinstance(self.query_encoder_bundle, VerifiedModelBundleV3):
            raise ValueError("query encoder bundle is mandatory")
        self.query_encoder_bundle.validate()
        _sha(self.query_condition_sha256, "query_condition_sha256")

    @property
    def conservative_index(self) -> int:
        self.validate()
        return self.candidates.index(self.conservative_parent)

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PANEL_SCHEMA,
            "source_pcm_sha256": self.source_pcm_sha256,
            "candidate_panel_sha256": candidate_panel_sha256(
                self.candidates
            ),
            "candidates": [
                candidate.to_dict() for candidate in self.candidates
            ],
            "metric_names": list(self.metric_names),
            "feature_contract_sha256": self.feature_contract_sha256,
            "risk_model_bundle": self.risk_model_bundle.identity_dict(),
            "calibration_policy_sha256": self.calibration_policy_sha256,
            "calibration_certificate": (
                self.calibration_certificate.identity_dict()
            ),
            "calibration_certificate_sha256": (
                self.calibration_certificate.sha256
            ),
            "conservative_parent": self.conservative_parent.to_dict(),
            "conservative_parent_certificate": (
                self.conservative_parent_certificate.identity_dict()
            ),
            "conservative_parent_certificate_sha256": (
                self.conservative_parent_certificate.sha256
            ),
            "query_encoder_bundle": (
                self.query_encoder_bundle.identity_dict()
            ),
            "query_condition_sha256": self.query_condition_sha256,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class RiskRouterConfigV3:
    critical_thresholds: tuple[tuple[str, float], ...]
    secondary_weights: tuple[tuple[str, float], ...] = ()
    critical_slack_weight: float = 1.0
    temporal_switch_penalty: float = 0.05
    frequency_switch_penalty: float = 0.05
    feasibility_tolerance: float = 0.0
    milp_time_limit_seconds: float = 180.0
    milp_relative_gap: float = 0.0
    returned_gap_tolerance: float = 1e-12
    integrality_tolerance: float = 1e-7
    objective_relative_tolerance: float = 1e-9
    objective_absolute_tolerance: float = 1e-10
    uniqueness_tolerance: float = 1e-9

    def validate(self, panel: RiskPanelIdentityV3) -> None:
        panel.validate()
        critical = dict(self.critical_thresholds)
        secondary = dict(self.secondary_weights)
        if (
            not critical
            or len(critical) != len(self.critical_thresholds)
            or any(name not in panel.metric_names for name in critical)
        ):
            raise ValueError(
                "critical thresholds must be unique known metrics"
            )
        if (
            len(secondary) != len(self.secondary_weights)
            or any(name not in panel.metric_names for name in secondary)
        ):
            raise ValueError(
                "secondary weights must be unique known metrics"
            )
        if set(critical) & set(secondary):
            raise ValueError(
                "a metric cannot be both critical and secondary"
            )
        for name, value in self.critical_thresholds:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(
                    f"critical threshold {name} must be positive"
                )
        for name, value in self.secondary_weights:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(
                    f"secondary weight {name} must be strictly positive"
                )
        for name in (
            "critical_slack_weight",
            "temporal_switch_penalty",
            "frequency_switch_penalty",
            "feasibility_tolerance",
            "milp_time_limit_seconds",
            "milp_relative_gap",
            "returned_gap_tolerance",
            "integrality_tolerance",
            "objective_relative_tolerance",
            "objective_absolute_tolerance",
            "uniqueness_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.critical_slack_weight <= 0:
            raise ValueError("critical_slack_weight must be positive")
        if self.milp_time_limit_seconds <= 0:
            raise ValueError("milp_time_limit_seconds must be positive")
        if self.milp_relative_gap != 0.0:
            raise ValueError(
                "certified routing requires milp_relative_gap exactly zero"
            )
        if (
            self.returned_gap_tolerance <= 0
            or self.integrality_tolerance <= 0
            or self.objective_relative_tolerance <= 0
            or self.objective_absolute_tolerance <= 0
            or self.uniqueness_tolerance < 0
        ):
            raise ValueError("solver validation tolerances are invalid")

    def identity_dict(self, panel: RiskPanelIdentityV3) -> dict[str, Any]:
        self.validate(panel)
        return {
            "schema": ROUTER_SCHEMA,
            "panel_identity_sha256": panel.sha256,
            "critical_thresholds": [
                [name, float(value)]
                for name, value in self.critical_thresholds
            ],
            "secondary_weights": [
                [name, float(value)]
                for name, value in self.secondary_weights
            ],
            "critical_slack_weight": float(
                self.critical_slack_weight
            ),
            "temporal_switch_penalty": float(
                self.temporal_switch_penalty
            ),
            "frequency_switch_penalty": float(
                self.frequency_switch_penalty
            ),
            "feasibility_tolerance": float(
                self.feasibility_tolerance
            ),
            "milp_time_limit_seconds": float(
                self.milp_time_limit_seconds
            ),
            "milp_relative_gap": 0.0,
            "returned_gap_tolerance": float(
                self.returned_gap_tolerance
            ),
            "integrality_tolerance": float(
                self.integrality_tolerance
            ),
            "objective_relative_tolerance": float(
                self.objective_relative_tolerance
            ),
            "objective_absolute_tolerance": float(
                self.objective_absolute_tolerance
            ),
            "uniqueness_tolerance": float(
                self.uniqueness_tolerance
            ),
            "tie_policy": "abstain_on_nonunique",
        }

    def sha256(self, panel: RiskPanelIdentityV3) -> str:
        return _mapping_sha(self.identity_dict(panel))


@dataclass(frozen=True)
class RiskPanelV3:
    identity: RiskPanelIdentityV3
    upper: np.ndarray
    available: np.ndarray

    def validate(self) -> tuple[int, int, int, int]:
        self.identity.validate()
        upper = np.asarray(self.upper, dtype=np.float64)
        available = np.asarray(self.available, dtype=bool)
        if upper.ndim != 4 or upper.shape != available.shape:
            raise ValueError(
                "upper/available must share "
                "(time,band,candidate,metric)"
            )
        time, bands, candidates, metrics = upper.shape
        if min(time, bands, candidates, metrics) < 1:
            raise ValueError("risk panel dimensions must be positive")
        if candidates != len(self.identity.candidates):
            raise ValueError(
                "risk candidate axis differs from frozen identity"
            )
        if metrics != len(self.identity.metric_names):
            raise ValueError(
                "risk metric axis differs from frozen identity"
            )
        if np.any(~np.isfinite(upper[available])):
            raise ValueError("available upper risks must be finite")
        if np.any(upper[available] < 0):
            raise ValueError("available upper risks must be non-negative")
        return upper.shape

    @property
    def sha256(self) -> str:
        self.validate()
        mask = np.asarray(self.available, dtype=bool)
        upper = np.ascontiguousarray(
            np.where(
                mask,
                np.asarray(self.upper, dtype=np.float64),
                0.0,
            ),
            dtype="<f8",
        )
        available = np.ascontiguousarray(mask, dtype=np.uint8)
        header = {
            "schema": PANEL_SCHEMA,
            "identity_sha256": self.identity.sha256,
            "shape": list(upper.shape),
            "risk_dtype": upper.dtype.str,
            "availability_dtype": available.dtype.str,
        }
        return "sha256:" + hashlib.sha256(
            _canonical(header) + upper.tobytes() + available.tobytes()
        ).hexdigest()


@dataclass(frozen=True)
class RouteDecisionV3:
    status: str
    labels: np.ndarray
    raw_parent_bypass: bool
    objective: float | None
    data_objective: float | None
    temporal_switches: int
    frequency_switches: int
    selection_counts: tuple[int, ...]
    infeasible_cells: tuple[tuple[int, int], ...]
    routing_plan_sha256: str
    conservative_parent: CandidateIdentityV3
    conservative_parent_certificate_sha256: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_SCHEMA,
            "status": self.status,
            "labels_shape": list(np.asarray(self.labels).shape),
            "raw_parent_bypass": self.raw_parent_bypass,
            "objective": self.objective,
            "data_objective": self.data_objective,
            "temporal_switches": self.temporal_switches,
            "frequency_switches": self.frequency_switches,
            "selection_counts": list(self.selection_counts),
            "infeasible_cells": [
                list(value) for value in self.infeasible_cells
            ],
            "routing_plan_sha256": self.routing_plan_sha256,
            "conservative_parent": self.conservative_parent.to_dict(),
            "conservative_parent_certificate_sha256": (
                self.conservative_parent_certificate_sha256
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DecodedSolutionV3:
    labels: np.ndarray
    objective: float
    data_objective: float
    temporal_switches: int
    frequency_switches: int


def _metric_indices(
    panel: RiskPanelIdentityV3,
) -> dict[str, int]:
    return {
        name: index for index, name in enumerate(panel.metric_names)
    }


def candidate_costs_v3(
    panel: RiskPanelV3,
    config: RiskRouterConfigV3,
) -> tuple[np.ndarray, np.ndarray]:
    time, bands, candidates, _ = panel.validate()
    config.validate(panel.identity)
    upper = np.asarray(panel.upper, dtype=np.float64)
    available = np.asarray(panel.available, dtype=bool)
    indices = _metric_indices(panel.identity)
    normalized = np.zeros(
        (time, bands, candidates, len(config.critical_thresholds)),
        dtype=np.float64,
    )
    feasible = np.ones((time, bands, candidates), dtype=bool)
    for axis, (name, threshold) in enumerate(
        config.critical_thresholds
    ):
        metric = indices[name]
        present = available[..., metric]
        value = np.where(present, upper[..., metric], 0.0)
        feasible &= present
        feasible &= value <= (
            float(threshold) + float(config.feasibility_tolerance)
        )
        normalized[..., axis] = value / float(threshold)
    cost = float(config.critical_slack_weight) * np.max(
        normalized, axis=-1
    )
    for name, weight in config.secondary_weights:
        metric = indices[name]
        present = available[..., metric]
        value = np.where(present, upper[..., metric], 0.0)
        feasible &= present
        cost += float(weight) * value
    if np.any(~np.isfinite(cost)):
        raise ValueError("risk unary costs are non-finite")
    return cost, feasible


def _plan_sha(
    labels: np.ndarray,
    panel: RiskPanelV3,
    config: RiskRouterConfigV3,
    *,
    raw_parent_bypass: bool,
) -> str:
    array = np.ascontiguousarray(labels, dtype="<i4")
    header = {
        "schema": ROUTE_SCHEMA,
        "risk_panel_sha256": panel.sha256,
        "router_sha256": config.sha256(panel.identity),
        "raw_parent_bypass": bool(raw_parent_bypass),
        "conservative_parent": (
            panel.identity.conservative_parent.to_dict()
        ),
        "conservative_parent_certificate_sha256": (
            panel.identity.conservative_parent_certificate.sha256
        ),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


def _fallback(
    panel: RiskPanelV3,
    config: RiskRouterConfigV3,
    infeasible: np.ndarray,
    status: str,
    reason: str,
) -> RouteDecisionV3:
    time, bands, candidates, _ = panel.validate()
    panel.identity.conservative_parent_certificate.validate(
        source_pcm_sha256=panel.identity.source_pcm_sha256,
        parent=panel.identity.conservative_parent,
    )
    index = panel.identity.conservative_index
    labels = np.full((time, bands), index, dtype=np.int32)
    counts = np.bincount(labels.ravel(), minlength=candidates)
    cells = tuple(
        tuple(int(value) for value in row)
        for row in np.argwhere(infeasible)
    )
    return RouteDecisionV3(
        status=status,
        labels=labels,
        raw_parent_bypass=True,
        objective=None,
        data_objective=None,
        temporal_switches=0,
        frequency_switches=0,
        selection_counts=tuple(int(value) for value in counts),
        infeasible_cells=cells,
        routing_plan_sha256=_plan_sha(
            labels, panel, config, raw_parent_bypass=True
        ),
        conservative_parent=panel.identity.conservative_parent,
        conservative_parent_certificate_sha256=(
            panel.identity.conservative_parent_certificate.sha256
        ),
        reason=reason,
    )


def _problem(
    cost: np.ndarray,
    feasible: np.ndarray,
    config: RiskRouterConfigV3,
    *,
    excluded_labels: np.ndarray | None = None,
):
    from scipy.optimize import Bounds, LinearConstraint
    from scipy.sparse import coo_matrix

    time, bands, candidates = cost.shape
    cells = time * bands
    temporal_edges = [
        (t * bands + b, (t + 1) * bands + b)
        for t in range(time - 1)
        for b in range(bands)
    ]
    frequency_edges = [
        (t * bands + b, t * bands + b + 1)
        for t in range(time)
        for b in range(bands - 1)
    ]
    edges = (
        [
            (left, right, float(config.temporal_switch_penalty))
            for left, right in temporal_edges
        ]
        + [
            (left, right, float(config.frequency_switch_penalty))
            for left, right in frequency_edges
        ]
    )
    x_count = cells * candidates
    variable_count = x_count + len(edges) * candidates
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:x_count] = cost.reshape(-1) / float(cells)
    for edge_index, (_, _, penalty) in enumerate(edges):
        objective[
            x_count + edge_index * candidates:
            x_count + (edge_index + 1) * candidates
        ] = penalty / (2.0 * float(cells))

    lower_bound = np.zeros(variable_count, dtype=np.float64)
    upper_bound = np.ones(variable_count, dtype=np.float64)
    upper_bound[:x_count] = feasible.reshape(-1).astype(np.float64)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for cell in range(cells):
        for candidate in range(candidates):
            rows.append(row)
            columns.append(cell * candidates + candidate)
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        row += 1
    for edge_index, (left, right, _) in enumerate(edges):
        for candidate in range(candidates):
            d_index = x_count + edge_index * candidates + candidate
            for first, second in ((left, right), (right, left)):
                rows.extend((row, row, row))
                columns.extend((
                    first * candidates + candidate,
                    second * candidates + candidate,
                    d_index,
                ))
                values.extend((1.0, -1.0, -1.0))
                lower.append(-np.inf)
                upper.append(0.0)
                row += 1
    if excluded_labels is not None:
        labels = np.asarray(
            excluded_labels, dtype=np.int64
        ).reshape(-1)
        if labels.size != cells:
            raise ValueError("excluded route shape differs")
        if np.any(labels < 0) or np.any(labels >= candidates):
            raise ValueError("excluded route contains an invalid label")
        for cell, candidate in enumerate(labels):
            rows.append(row)
            columns.append(cell * candidates + int(candidate))
            values.append(1.0)
        lower.append(-np.inf)
        upper.append(float(cells - 1))
        row += 1

    matrix = coo_matrix(
        (values, (rows, columns)), shape=(row, variable_count)
    ).tocsr()
    return (
        objective,
        np.r_[
            np.ones(x_count, dtype=np.int8),
            np.zeros(variable_count - x_count, dtype=np.int8),
        ],
        Bounds(lower_bound, upper_bound),
        LinearConstraint(
            matrix, np.asarray(lower), np.asarray(upper)
        ),
        x_count,
    )


def _solve(problem, config: RiskRouterConfigV3):
    from scipy.optimize import milp

    objective, integrality, bounds, constraints, _ = problem
    return milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(config.milp_time_limit_seconds),
            "mip_rel_gap": 0.0,
        },
    )


def _zero_gap_optimal(
    result: Any,
    config: RiskRouterConfigV3,
) -> bool:
    if not result.success or result.x is None or int(result.status) != 0:
        return False
    gap = getattr(result, "mip_gap", None)
    return (
        gap is not None
        and math.isfinite(float(gap))
        and abs(float(gap)) <= float(config.returned_gap_tolerance)
    )


def _discrete_objective(
    labels: np.ndarray,
    cost: np.ndarray,
    config: RiskRouterConfigV3,
) -> tuple[float, float, int, int]:
    labels = np.asarray(labels, dtype=np.int64)
    time, bands, candidates = cost.shape
    if labels.shape != (time, bands):
        raise ValueError("route shape differs from cost grid")
    if np.any(labels < 0) or np.any(labels >= candidates):
        raise ValueError("route contains an invalid candidate")
    selected = np.take_along_axis(
        cost, labels[..., None], axis=-1
    )[..., 0]
    data = float(selected.mean())
    temporal = int(np.count_nonzero(labels[1:] != labels[:-1]))
    frequency = int(
        np.count_nonzero(labels[:, 1:] != labels[:, :-1])
    )
    cells = float(time * bands)
    total = (
        data
        + float(config.temporal_switch_penalty)
        * temporal / cells
        + float(config.frequency_switch_penalty)
        * frequency / cells
    )
    return total, data, temporal, frequency


def _decode_solution(
    result: Any,
    *,
    cost: np.ndarray,
    feasible: np.ndarray,
    problem: tuple,
    config: RiskRouterConfigV3,
) -> DecodedSolutionV3:
    if not _zero_gap_optimal(result, config):
        raise RiskRouterV3Error(
            "solver result is not a zero-gap optimum"
        )
    time, bands, candidates = cost.shape
    x_count = problem[-1]
    x = np.asarray(result.x[:x_count], dtype=np.float64).reshape(
        time, bands, candidates
    )
    if (
        np.max(np.abs(x.sum(axis=-1) - 1.0))
        > float(config.integrality_tolerance)
        or np.max(np.minimum(np.abs(x), np.abs(x - 1.0)))
        > float(config.integrality_tolerance)
    ):
        raise RiskRouterV3Error("MILP returned a nonintegral route")
    labels = np.argmax(x, axis=-1).astype(np.int32)
    selected_feasible = np.take_along_axis(
        feasible, labels[..., None], axis=-1
    )[..., 0]
    if not np.all(selected_feasible):
        raise RiskRouterV3Error(
            "MILP selected an infeasible candidate"
        )
    total, data, temporal, frequency = _discrete_objective(
        labels, cost, config
    )
    if not math.isclose(
        float(result.fun),
        total,
        rel_tol=float(config.objective_relative_tolerance),
        abs_tol=float(config.objective_absolute_tolerance),
    ):
        raise RiskRouterV3Error(
            f"MILP objective mismatch: {result.fun} != {total}"
        )
    return DecodedSolutionV3(
        labels=labels,
        objective=total,
        data_objective=data,
        temporal_switches=temporal,
        frequency_switches=frequency,
    )


def exhaustive_optimal_routes_v3(
    cost: np.ndarray,
    feasible: np.ndarray,
    config: RiskRouterConfigV3,
    *,
    maximum_cells: int = 10,
) -> tuple[float, tuple[np.ndarray, ...]]:
    """Solver-independent exact mirror for property tests and tiny scenes."""

    cost = np.asarray(cost, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    if cost.ndim != 3 or feasible.shape != cost.shape:
        raise ValueError(
            "cost/feasible must share (time,band,candidate)"
        )
    time, bands, candidates = cost.shape
    cells = time * bands
    if cells > maximum_cells:
        raise ValueError("exhaustive mirror grid exceeds maximum_cells")
    best = math.inf
    routes = []
    for assignment in product(range(candidates), repeat=cells):
        labels = np.asarray(
            assignment, dtype=np.int32
        ).reshape(time, bands)
        selected_feasible = np.take_along_axis(
            feasible, labels[..., None], axis=-1
        )[..., 0]
        if not np.all(selected_feasible):
            continue
        value = _discrete_objective(labels, cost, config)[0]
        if value < best - config.uniqueness_tolerance:
            best = value
            routes = [labels.copy()]
        elif math.isclose(
            value,
            best,
            rel_tol=0.0,
            abs_tol=config.uniqueness_tolerance,
        ):
            routes.append(labels.copy())
    if not routes:
        raise RiskRouterV3Error(
            "feasible cells produced no exhaustive route"
        )
    routes.sort(key=lambda route: tuple(route.ravel().tolist()))
    return best, tuple(routes)


def solve_risk_route_v3(
    panel: RiskPanelV3,
    config: RiskRouterConfigV3,
) -> RouteDecisionV3:
    """Return a unique zero-gap Potts route or the verified raw parent."""

    cost, feasible = candidate_costs_v3(panel, config)
    time, bands, candidates = cost.shape
    no_candidate = feasible.sum(axis=-1) == 0
    if np.any(no_candidate):
        return _fallback(
            panel,
            config,
            no_candidate,
            "ABSTAIN_NO_CONFIDENTLY_FEASIBLE_ROUTE",
            "one or more cells have no confidently feasible candidate",
        )

    primary_problem = _problem(cost, feasible, config)
    primary_result = _solve(primary_problem, config)
    if not _zero_gap_optimal(primary_result, config):
        return _fallback(
            panel,
            config,
            np.zeros((time, bands), dtype=bool),
            "ABSTAIN_SOLVER_UNCERTIFIED",
            "primary structured solve lacks a zero-gap certificate",
        )
    primary = _decode_solution(
        primary_result,
        cost=cost,
        feasible=feasible,
        problem=primary_problem,
        config=config,
    )

    alternate_problem = _problem(
        cost,
        feasible,
        config,
        excluded_labels=primary.labels,
    )
    alternate_result = _solve(alternate_problem, config)
    alternate_status = int(getattr(alternate_result, "status", -1))
    if alternate_status == 2:
        unique = True
    elif _zero_gap_optimal(alternate_result, config):
        alternate = _decode_solution(
            alternate_result,
            cost=cost,
            feasible=feasible,
            problem=alternate_problem,
            config=config,
        )
        if np.array_equal(alternate.labels, primary.labels):
            raise RiskRouterV3Error(
                "alternate-route exclusion failed"
            )
        unique = (
            alternate.objective
            > primary.objective + float(config.uniqueness_tolerance)
        )
        if not unique:
            return _fallback(
                panel,
                config,
                np.zeros((time, bands), dtype=bool),
                "ABSTAIN_NONUNIQUE_OPTIMUM",
                "more than one route is optimal within tolerance",
            )
    else:
        return _fallback(
            panel,
            config,
            np.zeros((time, bands), dtype=bool),
            "ABSTAIN_UNIQUENESS_UNCERTIFIED",
            "alternate solve did not prove infeasibility or a zero-gap optimum",
        )

    counts = np.bincount(
        primary.labels.ravel(), minlength=candidates
    )
    return RouteDecisionV3(
        status="ROUTE",
        labels=primary.labels,
        raw_parent_bypass=False,
        objective=primary.objective,
        data_objective=primary.data_objective,
        temporal_switches=primary.temporal_switches,
        frequency_switches=primary.frequency_switches,
        selection_counts=tuple(int(value) for value in counts),
        infeasible_cells=(),
        routing_plan_sha256=_plan_sha(
            primary.labels,
            panel,
            config,
            raw_parent_bypass=False,
        ),
        conservative_parent=panel.identity.conservative_parent,
        conservative_parent_certificate_sha256=(
            panel.identity.conservative_parent_certificate.sha256
        ),
        reason=(
            "unique zero-gap optimum; all selected upper-risk cells "
            "are feasible"
        ),
    )
