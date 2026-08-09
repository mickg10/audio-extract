"""Query-calibrated, serializable counterfactual-risk router v4.

V4 preserves the zero-gap v3 solver while strengthening the public contract:
calibration is bound to the query encoder, query-quality stratum and selected
risk quantile; JSON decisions carry the exact route labels; and the exhaustive
mirror computes the true minimum before applying uniqueness tolerance.
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

from .counterfactual_risk_router_v3 import (
    CalibrationCertificateV3,
    CandidateIdentityV3,
    MetricCoverageV3,
    RawParentBypassCertificateV3,
    RiskPanelIdentityV3,
    RiskPanelV3,
    RiskRouterConfigV3,
    RouteDecisionV3,
    VerifiedModelBundleV3,
    candidate_panel_sha256,
    solve_risk_route_v3,
)

PANEL_SCHEMA_V4 = "audio-extract/counterfactual-risk-panel/v4"
ROUTE_SCHEMA_V4 = "audio-extract/counterfactual-risk-route/v4"
CALIBRATION_SCHEMA_V4 = "audio-extract/risk-calibration-certificate/v4"
QUERY_SCOPE_SCHEMA = "audio-extract/risk-query-calibration-scope/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RiskRouterV4Error(RuntimeError):
    pass


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be canonical sha256:<64 lowercase hex>")
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _labels_sha(labels: np.ndarray) -> str:
    array = np.ascontiguousarray(labels, dtype="<i4")
    header = {
        "schema": ROUTE_SCHEMA_V4,
        "component": "labels",
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class QueryCalibrationScopeV4:
    query_encoder_bundle_sha256: str
    query_quality_stratum_sha256: str
    selected_quantile_level: float
    calibration_algorithm_sha256: str

    def validate(self) -> None:
        for name in (
            "query_encoder_bundle_sha256",
            "query_quality_stratum_sha256",
            "calibration_algorithm_sha256",
        ):
            _sha(getattr(self, name), name)
        level = float(self.selected_quantile_level)
        if not math.isfinite(level) or not 0 < level < 1:
            raise ValueError("selected_quantile_level must lie in (0,1)")

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": QUERY_SCOPE_SCHEMA,
            "query_encoder_bundle_sha256": (
                self.query_encoder_bundle_sha256
            ),
            "query_quality_stratum_sha256": (
                self.query_quality_stratum_sha256
            ),
            "selected_quantile_level": float(
                self.selected_quantile_level
            ),
            "calibration_algorithm_sha256": (
                self.calibration_algorithm_sha256
            ),
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class CalibrationCertificateV4:
    risk_model_bundle_sha256: str
    feature_contract_sha256: str
    candidate_panel_sha256: str
    calibration_policy_sha256: str
    calibration_data_manifest_sha256: str
    split_manifest_sha256: str
    query_scope: QueryCalibrationScopeV4
    metric_coverages: tuple[MetricCoverageV3, ...]
    status: str = "passed"

    def validate(
        self,
        *,
        risk_model_bundle_sha256: str,
        feature_contract_sha256: str,
        candidate_panel_sha256_value: str,
        calibration_policy_sha256: str,
        query_encoder_bundle_sha256: str,
        query_quality_stratum_sha256: str,
        selected_quantile_level: float,
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
                raise ValueError(
                    f"calibration certificate {name} mismatch"
                )
        self.query_scope.validate()
        if self.query_scope.query_encoder_bundle_sha256 != _sha(
            query_encoder_bundle_sha256,
            "expected query encoder bundle SHA",
        ):
            raise ValueError("calibration query encoder bundle mismatch")
        if self.query_scope.query_quality_stratum_sha256 != _sha(
            query_quality_stratum_sha256,
            "expected query quality stratum SHA",
        ):
            raise ValueError("calibration query quality stratum mismatch")
        if float(self.query_scope.selected_quantile_level) != float(
            selected_quantile_level
        ):
            raise ValueError("calibration selected quantile mismatch")
        _sha(
            self.calibration_data_manifest_sha256,
            "calibration data manifest SHA",
        )
        _sha(self.split_manifest_sha256, "calibration split manifest SHA")
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
        self.query_scope.validate()
        return {
            "schema": CALIBRATION_SCHEMA_V4,
            "status": self.status,
            "risk_model_bundle_sha256": self.risk_model_bundle_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "calibration_policy_sha256": self.calibration_policy_sha256,
            "calibration_data_manifest_sha256": (
                self.calibration_data_manifest_sha256
            ),
            "split_manifest_sha256": self.split_manifest_sha256,
            "query_scope": self.query_scope.identity_dict(),
            "query_scope_sha256": self.query_scope.sha256,
            "metric_coverages": [
                row.to_dict() for row in self.metric_coverages
            ],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())

    def to_v3(self) -> CalibrationCertificateV3:
        return CalibrationCertificateV3(
            risk_model_bundle_sha256=self.risk_model_bundle_sha256,
            feature_contract_sha256=self.feature_contract_sha256,
            candidate_panel_sha256=self.candidate_panel_sha256,
            calibration_policy_sha256=self.calibration_policy_sha256,
            calibration_data_manifest_sha256=(
                self.calibration_data_manifest_sha256
            ),
            split_manifest_sha256=self.split_manifest_sha256,
            metric_coverages=self.metric_coverages,
            status=self.status,
        )


@dataclass(frozen=True)
class RiskPanelIdentityV4:
    source_pcm_sha256: str
    candidates: tuple[CandidateIdentityV3, ...]
    metric_names: tuple[str, ...]
    feature_contract_sha256: str
    risk_model_bundle: VerifiedModelBundleV3
    calibration_policy_sha256: str
    calibration_certificate: CalibrationCertificateV4
    conservative_parent: CandidateIdentityV3
    conservative_parent_certificate: RawParentBypassCertificateV3
    query_encoder_bundle: VerifiedModelBundleV3
    query_condition_sha256: str
    query_quality_stratum_sha256: str
    selected_quantile_level: float

    def validate(self) -> None:
        _sha(self.source_pcm_sha256, "source_pcm_sha256")
        panel_sha = candidate_panel_sha256(self.candidates)
        if not self.metric_names or len(set(self.metric_names)) != len(
            self.metric_names
        ):
            raise ValueError("metric names must be non-empty and unique")
        _sha(self.feature_contract_sha256, "feature_contract_sha256")
        self.risk_model_bundle.validate()
        self.query_encoder_bundle.validate()
        _sha(self.calibration_policy_sha256, "calibration_policy_sha256")
        _sha(self.query_condition_sha256, "query_condition_sha256")
        _sha(
            self.query_quality_stratum_sha256,
            "query_quality_stratum_sha256",
        )
        self.calibration_certificate.validate(
            risk_model_bundle_sha256=self.risk_model_bundle.sha256,
            feature_contract_sha256=self.feature_contract_sha256,
            candidate_panel_sha256_value=panel_sha,
            calibration_policy_sha256=self.calibration_policy_sha256,
            query_encoder_bundle_sha256=self.query_encoder_bundle.sha256,
            query_quality_stratum_sha256=(
                self.query_quality_stratum_sha256
            ),
            selected_quantile_level=self.selected_quantile_level,
            metric_names=self.metric_names,
        )
        self.conservative_parent.validate()
        if self.conservative_parent not in self.candidates:
            raise ValueError(
                "conservative parent is not in the candidate bank"
            )
        self.conservative_parent_certificate.validate(
            source_pcm_sha256=self.source_pcm_sha256,
            parent=self.conservative_parent,
        )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PANEL_SCHEMA_V4,
            "source_pcm_sha256": self.source_pcm_sha256,
            "candidate_panel_sha256": candidate_panel_sha256(
                self.candidates
            ),
            "candidates": [row.to_dict() for row in self.candidates],
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
            "conservative_parent_certificate_sha256": (
                self.conservative_parent_certificate.sha256
            ),
            "query_encoder_bundle": (
                self.query_encoder_bundle.identity_dict()
            ),
            "query_condition_sha256": self.query_condition_sha256,
            "query_quality_stratum_sha256": (
                self.query_quality_stratum_sha256
            ),
            "selected_quantile_level": float(
                self.selected_quantile_level
            ),
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())

    def to_v3(self) -> RiskPanelIdentityV3:
        self.validate()
        return RiskPanelIdentityV3(
            source_pcm_sha256=self.source_pcm_sha256,
            candidates=self.candidates,
            metric_names=self.metric_names,
            feature_contract_sha256=self.feature_contract_sha256,
            risk_model_bundle=self.risk_model_bundle,
            calibration_policy_sha256=self.calibration_policy_sha256,
            calibration_certificate=self.calibration_certificate.to_v3(),
            conservative_parent=self.conservative_parent,
            conservative_parent_certificate=(
                self.conservative_parent_certificate
            ),
            query_encoder_bundle=self.query_encoder_bundle,
            query_condition_sha256=self.query_condition_sha256,
        )


@dataclass(frozen=True)
class RiskPanelV4:
    identity: RiskPanelIdentityV4
    upper: np.ndarray
    available: np.ndarray

    def validate(self) -> tuple[int, int, int, int]:
        self.identity.validate()
        upper = np.asarray(self.upper, dtype=np.float64)
        available = np.asarray(self.available, dtype=bool)
        if upper.ndim != 4 or upper.shape != available.shape:
            raise ValueError(
                "upper/available must share (time,band,candidate,metric)"
            )
        if upper.shape[2] != len(self.identity.candidates) or upper.shape[
            3
        ] != len(self.identity.metric_names):
            raise ValueError("risk panel axes differ from identity")
        if np.any(~np.isfinite(upper[available])) or np.any(
            upper[available] < 0
        ):
            raise ValueError(
                "available risks must be finite and non-negative"
            )
        return tuple(map(int, upper.shape))

    @property
    def sha256(self) -> str:
        shape = self.validate()
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
            "schema": PANEL_SCHEMA_V4,
            "identity_sha256": self.identity.sha256,
            "shape": list(shape),
            "risk_dtype": upper.dtype.str,
            "availability_dtype": available.dtype.str,
        }
        return "sha256:" + hashlib.sha256(
            _canonical(header) + upper.tobytes() + available.tobytes()
        ).hexdigest()

    def to_v3(self) -> RiskPanelV3:
        self.validate()
        return RiskPanelV3(
            self.identity.to_v3(),
            self.upper,
            self.available,
        )


@dataclass(frozen=True)
class RouteDecisionV4:
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
        labels = np.asarray(self.labels, dtype=np.int32)
        if labels.ndim != 2:
            raise RiskRouterV4Error(
                "route labels must be a two-dimensional grid"
            )
        return {
            "schema": ROUTE_SCHEMA_V4,
            "status": self.status,
            "labels": labels.tolist(),
            "labels_shape": list(labels.shape),
            "labels_dtype": "<i4",
            "labels_sha256": _labels_sha(labels),
            "raw_parent_bypass": self.raw_parent_bypass,
            "objective": self.objective,
            "data_objective": self.data_objective,
            "temporal_switches": self.temporal_switches,
            "frequency_switches": self.frequency_switches,
            "selection_counts": list(self.selection_counts),
            "infeasible_cells": [
                list(row) for row in self.infeasible_cells
            ],
            "routing_plan_sha256": self.routing_plan_sha256,
            "conservative_parent": self.conservative_parent.to_dict(),
            "conservative_parent_certificate_sha256": (
                self.conservative_parent_certificate_sha256
            ),
            "reason": self.reason,
        }


def _discrete_objective(
    labels: np.ndarray,
    cost: np.ndarray,
    config: RiskRouterConfigV3,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    selected = np.take_along_axis(
        cost,
        labels[..., None],
        axis=-1,
    )[..., 0]
    cells = float(labels.size)
    temporal = np.count_nonzero(labels[1:] != labels[:-1])
    frequency = np.count_nonzero(labels[:, 1:] != labels[:, :-1])
    return float(
        selected.mean()
        + config.temporal_switch_penalty * temporal / cells
        + config.frequency_switch_penalty * frequency / cells
    )


def exhaustive_optimal_routes_v4(
    cost: np.ndarray,
    feasible: np.ndarray,
    config: RiskRouterConfigV3,
    *,
    maximum_cells: int = 10,
) -> tuple[float, tuple[np.ndarray, ...]]:
    """Two-pass exact mirror: find the true minimum, then apply tolerance."""
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
    evaluated: list[tuple[float, np.ndarray]] = []
    for assignment in product(range(candidates), repeat=cells):
        labels = np.asarray(
            assignment,
            dtype=np.int32,
        ).reshape(time, bands)
        selected_feasible = np.take_along_axis(
            feasible,
            labels[..., None],
            axis=-1,
        )[..., 0]
        if np.all(selected_feasible):
            evaluated.append(
                (
                    _discrete_objective(labels, cost, config),
                    labels.copy(),
                )
            )
    if not evaluated:
        raise RiskRouterV4Error(
            "feasible cells produced no exhaustive route"
        )
    true_best = min(value for value, _ in evaluated)
    routes = [
        labels
        for value, labels in evaluated
        if value <= true_best + float(config.uniqueness_tolerance)
    ]
    routes.sort(key=lambda row: tuple(row.ravel().tolist()))
    return true_best, tuple(routes)


def _plan_sha_v4(
    labels: np.ndarray,
    panel: RiskPanelV4,
    config: RiskRouterConfigV3,
    *,
    raw_parent_bypass: bool,
) -> str:
    array = np.ascontiguousarray(labels, dtype="<i4")
    header = {
        "schema": ROUTE_SCHEMA_V4,
        "risk_panel_sha256": panel.sha256,
        "router_v3_sha256": config.sha256(panel.identity.to_v3()),
        "raw_parent_bypass": bool(raw_parent_bypass),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    return "sha256:" + hashlib.sha256(
        _canonical(header) + array.tobytes()
    ).hexdigest()


def solve_risk_route_v4(
    panel: RiskPanelV4,
    config: RiskRouterConfigV3,
) -> RouteDecisionV4:
    """Validate v4 query calibration, then invoke the zero-gap v3 solver."""
    panel.validate()
    v3: RouteDecisionV3 = solve_risk_route_v3(
        panel.to_v3(),
        config,
    )
    return RouteDecisionV4(
        status=v3.status,
        labels=np.asarray(v3.labels, dtype=np.int32).copy(),
        raw_parent_bypass=v3.raw_parent_bypass,
        objective=v3.objective,
        data_objective=v3.data_objective,
        temporal_switches=v3.temporal_switches,
        frequency_switches=v3.frequency_switches,
        selection_counts=v3.selection_counts,
        infeasible_cells=v3.infeasible_cells,
        routing_plan_sha256=_plan_sha_v4(
            v3.labels,
            panel,
            config,
            raw_parent_bypass=v3.raw_parent_bypass,
        ),
        conservative_parent=v3.conservative_parent,
        conservative_parent_certificate_sha256=(
            v3.conservative_parent_certificate_sha256
        ),
        reason=v3.reason,
    )
