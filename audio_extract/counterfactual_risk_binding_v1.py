"""Fail-closed bridge from identity-bound risk inference to v4 routing.

The inference and router contracts are deliberately separate. This module is
the only admissible composition boundary: it proves candidate/metric order,
model and query bundles, selected quantile, calibration policy, source PCM, and
risk arrays all describe the same inference event before routing is invoked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json

import numpy as np

from .counterfactual_risk_model_v2 import RiskPanelPayloadV2
from .counterfactual_risk_router_v3 import (
    CandidateIdentityV3,
    RawParentBypassCertificateV3,
    RiskRouterConfigV3,
    VerifiedModelBundleV3,
    candidate_panel_sha256,
)
from .counterfactual_risk_router_v4 import (
    CalibrationCertificateV4,
    RiskPanelIdentityV4,
    RiskPanelV4,
    RouteDecisionV4,
    solve_risk_route_v4,
)

BINDING_SCHEMA = "audio-extract/counterfactual-risk-inference-binding/v1"
BOUND_ROUTE_SCHEMA = "audio-extract/counterfactual-risk-bound-route/v1"
CANDIDATE_SLOT_SCHEMA = "audio-extract/counterfactual-candidate-slot/v1"


class RiskInferenceBindingError(RuntimeError):
    """Inference payload and router identity do not describe one event."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RiskInferenceBindingError(
            f"binding value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def candidate_slot_sha256(candidate: CandidateIdentityV3) -> str:
    """Return the model-axis identity for one exact recipe/PCM candidate."""

    candidate.validate()
    return _mapping_sha(
        {
            "schema": CANDIDATE_SLOT_SCHEMA,
            "candidate": candidate.to_dict(),
        }
    )


def candidate_axis_ids(
    candidates: Sequence[CandidateIdentityV3],
) -> tuple[str, ...]:
    rows = tuple(candidates)
    candidate_panel_sha256(rows)
    return tuple(candidate_slot_sha256(candidate) for candidate in rows)


def _immutable_copy(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RiskInferenceBindingV1:
    inference_payload_sha256: str
    source_pcm_sha256: str
    candidate_panel_sha256: str
    candidate_axis_ids: tuple[str, ...]
    metric_names: tuple[str, ...]
    risk_model_bundle_sha256: str
    query_encoder_bundle_sha256: str
    query_condition_sha256: str
    query_quality_stratum_sha256: str
    selected_quantile_level: float
    calibration_policy_sha256: str
    calibration_certificate_sha256: str
    risk_panel_sha256: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": BINDING_SCHEMA,
            "inference_payload_sha256": self.inference_payload_sha256,
            "source_pcm_sha256": self.source_pcm_sha256,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "candidate_axis_ids": list(self.candidate_axis_ids),
            "metric_names": list(self.metric_names),
            "risk_model_bundle_sha256": self.risk_model_bundle_sha256,
            "query_encoder_bundle_sha256": self.query_encoder_bundle_sha256,
            "query_condition_sha256": self.query_condition_sha256,
            "query_quality_stratum_sha256": (
                self.query_quality_stratum_sha256
            ),
            "selected_quantile_level": float(self.selected_quantile_level),
            "calibration_policy_sha256": self.calibration_policy_sha256,
            "calibration_certificate_sha256": (
                self.calibration_certificate_sha256
            ),
            "risk_panel_sha256": self.risk_panel_sha256,
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())


@dataclass(frozen=True)
class BoundRiskPanelV1:
    panel: RiskPanelV4
    binding: RiskInferenceBindingV1

    def validate(self) -> None:
        self.panel.validate()
        if self.binding.risk_panel_sha256 != self.panel.sha256:
            raise RiskInferenceBindingError(
                "binding names a different v4 risk panel"
            )
        if self.binding.candidate_panel_sha256 != candidate_panel_sha256(
            self.panel.identity.candidates
        ):
            raise RiskInferenceBindingError(
                "binding names a different candidate panel"
            )
        if self.binding.candidate_axis_ids != candidate_axis_ids(
            self.panel.identity.candidates
        ):
            raise RiskInferenceBindingError(
                "binding candidate-axis identities differ"
            )
        if self.binding.metric_names != self.panel.identity.metric_names:
            raise RiskInferenceBindingError("binding metric order differs")
        if self.binding.source_pcm_sha256 != (
            self.panel.identity.source_pcm_sha256
        ):
            raise RiskInferenceBindingError("binding source PCM differs")
        if self.binding.risk_model_bundle_sha256 != (
            self.panel.identity.risk_model_bundle.sha256
        ):
            raise RiskInferenceBindingError("binding risk model differs")
        if self.binding.query_encoder_bundle_sha256 != (
            self.panel.identity.query_encoder_bundle.sha256
        ):
            raise RiskInferenceBindingError("binding query encoder differs")
        if self.binding.query_condition_sha256 != (
            self.panel.identity.query_condition_sha256
        ):
            raise RiskInferenceBindingError("binding query condition differs")
        if self.binding.query_quality_stratum_sha256 != (
            self.panel.identity.query_quality_stratum_sha256
        ):
            raise RiskInferenceBindingError(
                "binding query-quality stratum differs"
            )
        if self.binding.selected_quantile_level != float(
            self.panel.identity.selected_quantile_level
        ):
            raise RiskInferenceBindingError(
                "binding selected quantile differs"
            )
        if self.binding.calibration_policy_sha256 != (
            self.panel.identity.calibration_policy_sha256
        ):
            raise RiskInferenceBindingError(
                "binding calibration policy differs"
            )
        if self.binding.calibration_certificate_sha256 != (
            self.panel.identity.calibration_certificate.sha256
        ):
            raise RiskInferenceBindingError(
                "binding calibration certificate differs"
            )

    @property
    def sha256(self) -> str:
        self.validate()
        return _mapping_sha(
            {
                "schema": BINDING_SCHEMA,
                "binding_sha256": self.binding.sha256,
                "risk_panel_sha256": self.panel.sha256,
            }
        )


@dataclass(frozen=True)
class BoundRouteDecisionV1:
    decision: RouteDecisionV4
    bound_risk_panel_sha256: str
    routing_plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = self.decision.to_dict()
        return {
            "schema": BOUND_ROUTE_SCHEMA,
            "bound_risk_panel_sha256": self.bound_risk_panel_sha256,
            "routing_plan_sha256": self.routing_plan_sha256,
            "decision": payload,
        }


def bind_inference_payload_v1(
    payload: RiskPanelPayloadV2,
    *,
    candidates: Sequence[CandidateIdentityV3],
    risk_model_bundle: VerifiedModelBundleV3,
    query_encoder_bundle: VerifiedModelBundleV3,
    query_quality_stratum_sha256: str,
    calibration_certificate: CalibrationCertificateV4,
    conservative_parent: CandidateIdentityV3,
    conservative_parent_certificate: RawParentBypassCertificateV3,
) -> BoundRiskPanelV1:
    """Prove one inference payload is admissible as one v4 risk panel."""

    payload.validate()
    rows = tuple(candidates)
    expected_axis = candidate_axis_ids(rows)
    if payload.candidate_ids != expected_axis:
        raise RiskInferenceBindingError(
            "inference candidate IDs/order do not match exact recipe/PCM slots"
        )
    if risk_model_bundle.weights_sha256 != payload.risk_model_state_sha256:
        raise RiskInferenceBindingError(
            "risk bundle weights/state differ from inference-time state"
        )
    if risk_model_bundle.config_sha256 != payload.model_config_sha256:
        raise RiskInferenceBindingError(
            "risk bundle config differs from inference-time config"
        )
    risk_model_bundle.validate()
    query_encoder_bundle.validate()
    if payload.query_encoder_sha256 != query_encoder_bundle.sha256:
        raise RiskInferenceBindingError(
            "query bundle differs from inference-time query encoder"
        )
    if payload.query_condition_sha256 is None:
        raise RiskInferenceBindingError(
            "v4 routing requires an identity-bound query condition"
        )
    if payload.calibration_policy_sha256 != (
        calibration_certificate.calibration_policy_sha256
    ):
        raise RiskInferenceBindingError(
            "inference and certificate calibration policies differ"
        )
    identity = RiskPanelIdentityV4(
        source_pcm_sha256=payload.source_pcm_sha256,
        candidates=rows,
        metric_names=payload.metric_names,
        feature_contract_sha256=payload.feature_contract_sha256,
        risk_model_bundle=risk_model_bundle,
        calibration_policy_sha256=payload.calibration_policy_sha256,
        calibration_certificate=calibration_certificate,
        conservative_parent=conservative_parent,
        conservative_parent_certificate=conservative_parent_certificate,
        query_encoder_bundle=query_encoder_bundle,
        query_condition_sha256=payload.query_condition_sha256,
        query_quality_stratum_sha256=query_quality_stratum_sha256,
        selected_quantile_level=payload.selected_quantile_level,
    )
    upper = _immutable_copy(payload.upper, dtype=np.float64)
    available = _immutable_copy(payload.available, dtype=bool)
    panel = RiskPanelV4(identity, upper, available)
    panel.validate()
    binding = RiskInferenceBindingV1(
        inference_payload_sha256=payload.sha256,
        source_pcm_sha256=payload.source_pcm_sha256,
        candidate_panel_sha256=candidate_panel_sha256(rows),
        candidate_axis_ids=expected_axis,
        metric_names=payload.metric_names,
        risk_model_bundle_sha256=risk_model_bundle.sha256,
        query_encoder_bundle_sha256=query_encoder_bundle.sha256,
        query_condition_sha256=payload.query_condition_sha256,
        query_quality_stratum_sha256=query_quality_stratum_sha256,
        selected_quantile_level=payload.selected_quantile_level,
        calibration_policy_sha256=payload.calibration_policy_sha256,
        calibration_certificate_sha256=calibration_certificate.sha256,
        risk_panel_sha256=panel.sha256,
    )
    result = BoundRiskPanelV1(panel, binding)
    result.validate()
    return result


def solve_bound_risk_route_v1(
    bound_panel: BoundRiskPanelV1,
    config: RiskRouterConfigV3,
) -> BoundRouteDecisionV1:
    """Route only after the inference-to-router binding revalidates."""

    bound_panel.validate()
    decision = solve_risk_route_v4(bound_panel.panel, config)
    labels = np.ascontiguousarray(decision.labels, dtype="<i4")
    header = {
        "schema": BOUND_ROUTE_SCHEMA,
        "bound_risk_panel_sha256": bound_panel.sha256,
        "v4_routing_plan_sha256": decision.routing_plan_sha256,
        "labels_shape": list(labels.shape),
        "labels_dtype": labels.dtype.str,
    }
    routing_plan_sha256 = "sha256:" + hashlib.sha256(
        _canonical(header) + labels.tobytes()
    ).hexdigest()
    return BoundRouteDecisionV1(
        decision=decision,
        bound_risk_panel_sha256=bound_panel.sha256,
        routing_plan_sha256=routing_plan_sha256,
    )
