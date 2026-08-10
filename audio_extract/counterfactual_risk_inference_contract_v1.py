"""Truth-free inference contract with a separate offline audit binding.

The exact dataset row ID commits clean truth, exact risks, and availability
labels; it must not be used as the model-facing row identity.  This module
projects a stable inference-safe row ID from mixture/cell/query/candidate/feature
inputs only and stores the exact-row/evidence relationship in a separate audit
binding that is never returned by ``model_input_document()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import math
import re

import numpy as np

from .counterfactual_risk_dataset_contract_v1 import CellGeometry
from .counterfactual_risk_dataset_contract_v2 import (
    CandidateArtifactBinding,
    CandidatePanel,
    DatasetManifestV2,
)
from .counterfactual_risk_feature_evidence_v1 import (
    FeatureEvidenceCertificate,
    FeatureEvidenceRegistry,
    validate_dataset_feature_evidence,
)

ROW_SCHEMA = "audio-extract/counterfactual-risk-inference-row/v1"
AUDIT_SCHEMA = "audio-extract/counterfactual-risk-inference-audit/v1"
MANIFEST_SCHEMA = "audio-extract/counterfactual-risk-inference-manifest/v1"
MODEL_INPUT_SCHEMA = "audio-extract/counterfactual-risk-model-input/v1"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class InferenceContractError(ValueError):
    """An inference row, manifest, or offline audit relationship is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise InferenceContractError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise InferenceContractError(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise InferenceContractError(
            f"{name} must be a non-empty, trim-stable string"
        )
    return value


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


def _real_tuple(value: Sequence[Any], name: str) -> tuple[float, ...]:
    result = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise InferenceContractError(
                f"{name} must contain real numbers and not booleans"
            )
        converted = float(item)
        if not math.isfinite(converted):
            raise InferenceContractError(f"{name} contains non-finite values")
        result.append(converted)
    if not result:
        raise InferenceContractError(f"{name} must be non-empty")
    return tuple(result)


def _bool_tuple(value: Sequence[Any], name: str) -> tuple[bool, ...]:
    result = tuple(value)
    if not result or any(
        not isinstance(item, (bool, np.bool_)) for item in result
    ):
        raise InferenceContractError(
            f"{name} must be a non-empty boolean sequence"
        )
    return tuple(bool(item) for item in result)


@dataclass(frozen=True)
class InferenceRowV1:
    work_id: str
    recording_session_id: str
    target_singer_id: str
    source_family_sha256: str
    query_condition_sha256: str
    mixture_pcm_sha256: str
    cell: CellGeometry
    candidate_panel_sha256: str
    candidate_artifacts: tuple[CandidateArtifactBinding, ...]
    feature_contract_sha256: str
    features: tuple[float, ...]
    feature_available: tuple[bool, ...]
    feature_required: tuple[bool, ...]
    metric_contract_sha256: str
    metric_names: tuple[str, ...]
    metric_units: tuple[str, ...]
    metric_directions: tuple[str, ...]
    route_policy_sha256: str

    def validate(self, *, candidate_panel: CandidatePanel) -> None:
        for name, value in (
            ("work_id", self.work_id),
            ("recording_session_id", self.recording_session_id),
            ("target_singer_id", self.target_singer_id),
        ):
            _text(value, name)
        for name, value in (
            ("source_family_sha256", self.source_family_sha256),
            ("query_condition_sha256", self.query_condition_sha256),
            ("mixture_pcm_sha256", self.mixture_pcm_sha256),
            ("candidate_panel_sha256", self.candidate_panel_sha256),
            ("feature_contract_sha256", self.feature_contract_sha256),
            ("metric_contract_sha256", self.metric_contract_sha256),
            ("route_policy_sha256", self.route_policy_sha256),
        ):
            _sha(value, name)
        self.cell.validate()
        candidate_panel.validate()
        if self.candidate_panel_sha256 != candidate_panel.sha256:
            raise InferenceContractError(
                "inference row names a different candidate panel"
            )
        if len(self.candidate_artifacts) != len(candidate_panel.slots):
            raise InferenceContractError(
                "candidate artifact axis differs from the candidate panel"
            )
        for slot, artifact in zip(
            candidate_panel.slots, self.candidate_artifacts
        ):
            artifact.validate(
                expected_slot=slot,
                expected_source_family_sha256=self.source_family_sha256,
            )
        features = _real_tuple(self.features, "features")
        available = _bool_tuple(
            self.feature_available, "feature_available"
        )
        required = _bool_tuple(self.feature_required, "feature_required")
        if not (len(features) == len(available) == len(required)):
            raise InferenceContractError(
                "feature values and masks differ in length"
            )
        if any(
            not is_available and value != 0.0
            for value, is_available in zip(features, available)
        ):
            raise InferenceContractError(
                "unavailable inference features must be canonical zero"
            )
        metric_count = len(self.metric_names)
        if metric_count < 1 or not (
            len(self.metric_units) == metric_count
            and len(self.metric_directions) == metric_count
        ):
            raise InferenceContractError(
                "metric names, units and directions are empty or misaligned"
            )
        if len(set(self.metric_names)) != metric_count:
            raise InferenceContractError("metric names must be unique")
        for name, unit, direction in zip(
            self.metric_names, self.metric_units, self.metric_directions
        ):
            _text(name, "metric name")
            _text(unit, "metric unit")
            if direction != "lower_is_better":
                raise InferenceContractError(
                    "inference defect metrics must declare lower_is_better"
                )

    @property
    def prediction_allowed(self) -> bool:
        available = _bool_tuple(
            self.feature_available, "feature_available"
        )
        required = _bool_tuple(self.feature_required, "feature_required")
        return all(
            is_available or not is_required
            for is_available, is_required in zip(available, required)
        )

    def identity_dict(self, *, candidate_panel: CandidatePanel) -> dict[str, Any]:
        self.validate(candidate_panel=candidate_panel)
        return {
            "schema": ROW_SCHEMA,
            "work_id": self.work_id,
            "recording_session_id": self.recording_session_id,
            "target_singer_id": self.target_singer_id,
            "source_family_sha256": self.source_family_sha256,
            "query_condition_sha256": self.query_condition_sha256,
            "mixture_pcm_sha256": self.mixture_pcm_sha256,
            "cell_sha256": self.cell.sha256,
            "cell": self.cell.identity_dict(),
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "candidate_artifacts": [
                artifact.identity_dict(
                    expected_slot=slot,
                    expected_source_family_sha256=self.source_family_sha256,
                )
                for slot, artifact in zip(
                    candidate_panel.slots, self.candidate_artifacts
                )
            ],
            "feature_contract_sha256": self.feature_contract_sha256,
            "features": list(_real_tuple(self.features, "features")),
            "feature_available": list(self.feature_available),
            "feature_required": list(self.feature_required),
            "prediction_allowed": self.prediction_allowed,
            "metric_contract_sha256": self.metric_contract_sha256,
            "metric_names": list(self.metric_names),
            "metric_units": list(self.metric_units),
            "metric_directions": list(self.metric_directions),
            "route_policy_sha256": self.route_policy_sha256,
        }

    def sha256(self, *, candidate_panel: CandidatePanel) -> str:
        return _mapping_sha(self.identity_dict(candidate_panel=candidate_panel))


@dataclass(frozen=True)
class InferenceAuditBinding:
    inference_row_sha256: str
    offline_row_id: str
    feature_evidence_sha256: str

    def validate(self) -> None:
        _sha(self.inference_row_sha256, "inference_row_sha256")
        _sha(self.offline_row_id, "offline_row_id")
        _sha(self.feature_evidence_sha256, "feature_evidence_sha256")

    def identity_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "schema": AUDIT_SCHEMA,
            "inference_row_sha256": self.inference_row_sha256,
            "offline_row_id": self.offline_row_id,
            "feature_evidence_sha256": self.feature_evidence_sha256,
        }


@dataclass(frozen=True)
class InferenceManifestV1:
    candidate_panel: CandidatePanel
    rows: tuple[InferenceRowV1, ...]
    audit_bindings: tuple[InferenceAuditBinding, ...]
    offline_dataset_sha256: str
    feature_registry_sha256: str
    source_commit: str

    @classmethod
    def build(
        cls,
        dataset: DatasetManifestV2,
        feature_registry: FeatureEvidenceRegistry,
    ) -> "InferenceManifestV1":
        validate_dataset_feature_evidence(dataset, feature_registry)
        certificates = feature_registry.by_row_id()
        candidate_panel = dataset.rows[0].candidate_panel
        pairs = []
        for offline_row in dataset.rows:
            evidence: FeatureEvidenceCertificate = certificates[
                offline_row.row_id
            ]
            inference_row = InferenceRowV1(
                work_id=offline_row.group_family.work_id,
                recording_session_id=(
                    offline_row.group_family.recording_session_id
                ),
                target_singer_id=(
                    offline_row.group_family.target_singer_id
                ),
                source_family_sha256=(
                    offline_row.group_family.source_family_sha256
                ),
                query_condition_sha256=(
                    offline_row.group_family.query_condition_sha256
                ),
                mixture_pcm_sha256=offline_row.mixture_pcm_sha256,
                cell=offline_row.cell,
                candidate_panel_sha256=candidate_panel.sha256,
                candidate_artifacts=offline_row.candidate_artifacts,
                feature_contract_sha256=(
                    offline_row.feature_contract_sha256
                ),
                features=tuple(float(value) for value in offline_row.features),
                feature_available=evidence.available,
                feature_required=evidence.required,
                metric_contract_sha256=offline_row.metric_contract_sha256,
                metric_names=offline_row.metric_names,
                metric_units=offline_row.metric_units,
                metric_directions=offline_row.metric_directions,
                route_policy_sha256=offline_row.route_policy_sha256,
            )
            inference_sha = inference_row.sha256(
                candidate_panel=candidate_panel
            )
            audit = InferenceAuditBinding(
                inference_row_sha256=inference_sha,
                offline_row_id=offline_row.row_id,
                feature_evidence_sha256=evidence.sha256,
            )
            pairs.append((inference_sha, inference_row, audit))
        pairs.sort(key=lambda value: value[0])
        result = cls(
            candidate_panel=candidate_panel,
            rows=tuple(value[1] for value in pairs),
            audit_bindings=tuple(value[2] for value in pairs),
            offline_dataset_sha256=dataset.sha256,
            feature_registry_sha256=feature_registry.sha256,
            source_commit=dataset.source_commit,
        )
        result.validate()
        return result

    def validate(self) -> None:
        self.candidate_panel.validate()
        _sha(self.offline_dataset_sha256, "offline_dataset_sha256")
        _sha(self.feature_registry_sha256, "feature_registry_sha256")
        _git(self.source_commit, "inference manifest source_commit")
        if not self.rows or len(self.rows) != len(self.audit_bindings):
            raise InferenceContractError(
                "inference rows and audit bindings are empty or misaligned"
            )
        inference_shas = []
        for row in self.rows:
            row.validate(candidate_panel=self.candidate_panel)
            inference_shas.append(
                row.sha256(candidate_panel=self.candidate_panel)
            )
        if tuple(inference_shas) != tuple(sorted(inference_shas)):
            raise InferenceContractError(
                "inference rows must be in canonical inference-SHA order"
            )
        if len(set(inference_shas)) != len(inference_shas):
            raise InferenceContractError("inference row identities are duplicated")
        for expected_sha, audit in zip(
            inference_shas, self.audit_bindings
        ):
            audit.validate()
            if audit.inference_row_sha256 != expected_sha:
                raise InferenceContractError(
                    "audit binding names a different inference row"
                )
        offline_ids = tuple(
            audit.offline_row_id for audit in self.audit_bindings
        )
        if len(set(offline_ids)) != len(offline_ids):
            raise InferenceContractError(
                "one offline exact row is bound to multiple inference rows"
            )
        first = self.rows[0]
        common = (
            first.feature_contract_sha256,
            first.metric_contract_sha256,
            first.metric_names,
            first.metric_units,
            first.metric_directions,
            first.route_policy_sha256,
            len(first.features),
        )
        for row in self.rows[1:]:
            current = (
                row.feature_contract_sha256,
                row.metric_contract_sha256,
                row.metric_names,
                row.metric_units,
                row.metric_directions,
                row.route_policy_sha256,
                len(row.features),
            )
            if current != common:
                raise InferenceContractError(
                    "inference rows disagree on model-facing contracts"
                )

    def model_input_document(self) -> dict[str, Any]:
        """Return the exact truth-free object passed across the model boundary."""

        self.validate()
        return {
            "schema": MODEL_INPUT_SCHEMA,
            "candidate_panel": self.candidate_panel.identity_dict(),
            "candidate_panel_sha256": self.candidate_panel.sha256,
            "feature_contract_sha256": self.rows[0].feature_contract_sha256,
            "metric_contract_sha256": self.rows[0].metric_contract_sha256,
            "metric_names": list(self.rows[0].metric_names),
            "metric_units": list(self.rows[0].metric_units),
            "metric_directions": list(self.rows[0].metric_directions),
            "route_policy_sha256": self.rows[0].route_policy_sha256,
            "rows": [
                {
                    "inference_row_sha256": row.sha256(
                        candidate_panel=self.candidate_panel
                    ),
                    **row.identity_dict(candidate_panel=self.candidate_panel),
                }
                for row in self.rows
            ],
        }

    @property
    def model_input_sha256(self) -> str:
        return _mapping_sha(self.model_input_document())

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": MANIFEST_SCHEMA,
            "source_commit": self.source_commit,
            "offline_dataset_sha256": self.offline_dataset_sha256,
            "feature_registry_sha256": self.feature_registry_sha256,
            "model_input_sha256": self.model_input_sha256,
            "audit_bindings": [
                binding.identity_dict() for binding in self.audit_bindings
            ],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())
